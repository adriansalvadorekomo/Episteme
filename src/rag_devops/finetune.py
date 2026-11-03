"""Fine-tuning QLoRA du LLM sur le corpus du lakehouse (extra `finetune`).

Objectif : adapter le modèle de base au domaine (vocabulaire, style d'explication)
à partir des chunks SILVER, au format instruction→réponse du template de chat.
4-bit (QLoRA) pour tenir sur 8 Go de VRAM. L'adaptateur produit est chargé par
`LLM.load()` quand `RAG_LLM_ADAPTER` pointe dessus (ou vaut "auto").

Lourd et à la demande : le serveur et l'entraînement ne tiennent pas ensemble en
8 Go, lance de préférence `uv run rag-finetune` avec l'API arrêtée.
"""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

from .config import Settings, settings


def _latest_adapter(cfg: Settings) -> Path | None:
    d = cfg.adapters_dir
    if not d.exists():
        return None
    subs = [p for p in d.iterdir() if p.is_dir() and (p / "adapter_config.json").exists()]
    return max(subs, key=lambda p: p.stat().st_mtime) if subs else None


def resolve_adapter(cfg: Settings | None = None) -> Path | None:
    """Résout `cfg.llm_adapter` : chemin explicite, "auto" (le plus récent), ou rien."""
    cfg = cfg or settings
    val = (cfg.llm_adapter or "").strip()
    if not val:
        return None
    if val == "auto":
        return _latest_adapter(cfg)
    p = Path(val)
    return p if p.exists() else None


def _build_examples(cfg: Settings, max_examples: int) -> list[str]:
    """Construit des textes d'entraînement (template de chat) depuis les chunks."""
    from .index import VectorStore
    from .pipeline import llm_system_prompt

    store = VectorStore.load(cfg)
    chunks = [c["text"] for c in store.chunks if len(c["text"]) > 200]
    random.seed(0)
    random.shuffle(chunks)
    chunks = chunks[:max_examples]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.llm_model)
    sys_p = llm_system_prompt(cfg)
    texts = []
    for ch in chunks:
        messages = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": "Explique ce concept d'ingénierie logicielle avec rigueur."},
            {"role": "assistant", "content": ch},
        ]
        texts.append(tok.apply_chat_template(messages, tokenize=False))
    return texts


def run_finetune(cfg: Settings | None = None, max_examples: int = 300, epochs: float = 1.0) -> dict:
    """Lance un QLoRA et renvoie un résumé (chemin de l'adaptateur, etc.)."""
    cfg = cfg or settings
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    texts = _build_examples(cfg, max_examples)
    if not texts:
        return {"status": "empty", "note": "aucun chunk disponible pour l'entraînement"}
    ds = Dataset.from_dict({"text": texts})

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = cfg.adapters_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.llm_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model, quantization_config=quant, device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    args = SFTConfig(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        bf16=False, fp16=True,
        max_seq_length=1024,
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # journalise la run dans le catalogue (best-effort)
    try:
        import pyarrow as pa

        from . import lake
        from .lake import catalog as cat
        from .lake import storage

        storage.table(cat.FINETUNE_RUNS, cfg).append(pa.Table.from_pylist([{
            "run_id": ts, "created_at": datetime.utcnow(), "base_model": cfg.llm_model,
            "adapter_path": str(out_dir), "n_examples": len(texts), "status": "done",
        }], schema=cat.arrow_for(cat.FINETUNE_RUNS)))
    except Exception:
        pass

    return {"status": "done", "run_id": ts, "adapter_path": str(out_dir),
            "n_examples": len(texts), "base_model": cfg.llm_model}
