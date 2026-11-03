"""Étape 4 — Génération de réponses avec un LLM open-source (Transformers).

Charge un LLM instruct open-source et génère une réponse à partir d'un prompt
(système + question + contexte récupéré). Le device CUDA est utilisé si dispo.
"""
from __future__ import annotations

from .config import Settings, settings


class LLM:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.device = self.cfg.resolve_device()
        self._tok = None
        self._model = None

    def load(self) -> "LLM":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = self.cfg.llm_model
        self._tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # Chargement puis déplacement explicite sur le device (plus prévisible
        # que device_map sur une machine mono-GPU).
        self._model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self._model.to(self.device)

        # Adaptateur LoRA optionnel (fine-tuning sur le lakehouse).
        try:
            from .finetune import resolve_adapter

            adapter = resolve_adapter(self.cfg)
            if adapter is not None:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, str(adapter))
                print(f"🔧 Adaptateur LoRA chargé : {adapter}")
        except Exception as exc:  # peft absent ou adaptateur invalide => modèle de base
            print(f"(adaptateur ignoré : {exc})")

        self._model.eval()
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
        # ne décode que les tokens générés (au-delà du prompt)
        gen = out[0][input_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
