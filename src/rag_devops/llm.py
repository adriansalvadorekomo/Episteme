"""Étape 4 — Génération de réponses avec un LLM open-source (Transformers).

Charge un LLM instruct open-source et génère une réponse à partir d'un prompt
(système + question + contexte récupéré). Le device CUDA est utilisé si dispo.
"""
from __future__ import annotations

import threading

from .config import Settings, settings


class LLM:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.device = self.cfg.resolve_device()
        self._tok = None
        self._model = None
        self._lock = threading.Lock()  # un seul chargement, même si préchauffé en tâche de fond
        self.last_stats: dict | None = None  # métriques de la dernière génération

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> "LLM":
        with self._lock:
            if self._model is not None:  # déjà chargé (ex. par le préchauffage)
                return self
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            name = self.cfg.llm_model
            tok = AutoTokenizer.from_pretrained(name)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            # Chargement puis déplacement explicite sur le device (plus prévisible
            # que device_map sur une machine mono-GPU).
            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
            model.to(self.device)

            # Adaptateur LoRA optionnel (fine-tuning sur le lakehouse).
            try:
                from .finetune import resolve_adapter

                adapter = resolve_adapter(self.cfg)
                if adapter is not None:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, str(adapter))
                    print(f"🔧 Adaptateur LoRA chargé : {adapter}")
            except Exception as exc:  # peft absent ou adaptateur invalide => modèle de base
                print(f"(adaptateur ignoré : {exc})")

            model.eval()
            # publié en dernier : un autre thread ne voit jamais un modèle à moitié chargé
            self._tok, self._model = tok, model
            return self

    @property
    def tokenizer(self):
        if self._tok is None:
            self.load()
        return self._tok

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict] | None = None,
    ) -> str:
        import torch

        # système → historique (tours précédents) → question courante
        messages = [{"role": "system", "content": system_prompt}]
        for h in history or []:
            role, content = h.get("role"), h.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        # Utilise le chat template natif du modèle instruct.
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        import time

        cuda = torch.cuda.is_available() and str(self.model.device).startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        do_sample = self.cfg.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=self.cfg.temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        if cuda:
            torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0

        # ne décode que les tokens générés (au-delà du prompt)
        gen = out[0][input_len:]
        n_new = int(gen.shape[-1])
        # métriques d'inférence de la dernière génération (exposées à l'UI)
        self.last_stats = {
            "prompt_tokens": int(input_len),
            "new_tokens": n_new,
            "gen_ms": round(gen_s * 1000),
            "tok_s": round(n_new / gen_s, 1) if gen_s > 0 else None,
            "peak_vram_mb": int(torch.cuda.max_memory_allocated() // 2**20) if cuda else None,
            "device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "dtype": str(self.model.dtype).replace("torch.", ""),
            "adapter": self._model.__class__.__name__ == "PeftModel",
        }
        if cuda:
            # rend au pilote la mémoire réservée mais inutilisée (KV-cache, activations) :
            # la jauge VRAM reflète l'usage réel et les autres processus respirent.
            torch.cuda.empty_cache()
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
