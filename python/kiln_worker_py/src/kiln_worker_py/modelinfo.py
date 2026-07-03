"""Static model metadata derived from a local model directory.

Everything here is filesystem-only (no MLX, no tokenizer load) so it can be
computed cheaply and reported in ``WorkerInfo`` regardless of engine state.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

_DTYPE_MAP = {
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
}


@dataclass(frozen=True)
class ModelInfo:
    model_path: str
    architecture: str
    dtype: str
    max_context_len: int
    vocab_size: int
    weights_bytes: int
    weights_fingerprint: str
    chat_template_hash: str


def _dtype_string(config: dict) -> str:
    quant = config.get("quantization")
    if isinstance(quant, dict) and "bits" in quant:
        return f"q{quant['bits']}_g{quant.get('group_size', 64)}"
    return _DTYPE_MAP.get(config.get("torch_dtype", ""), config.get("torch_dtype", ""))


def _chat_template(model_dir: pathlib.Path) -> str:
    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    tok_config = model_dir / "tokenizer_config.json"
    if tok_config.is_file():
        template = json.loads(tok_config.read_text(encoding="utf-8")).get(
            "chat_template", ""
        )
        if isinstance(template, str):
            return template
    return ""


def read_model_info(model_path: str) -> ModelInfo:
    """Read config.json and weight-file metadata from a local model dir."""
    model_dir = pathlib.Path(model_path).expanduser().resolve()
    config_bytes = (model_dir / "config.json").read_bytes()
    config = json.loads(config_bytes)

    weight_files = sorted(model_dir.glob("*.safetensors"))
    weights_bytes = sum(f.stat().st_size for f in weight_files)

    architecture = config.get("model_type", "unknown")
    dtype = _dtype_string(config)

    # Cheap, deterministic identity for this weight set: config content plus
    # each weight file's name and size. Not a content hash of the weights
    # (hashing ~1 GB per health-checked worker start is not worth it for the
    # fallback worker; the Rust worker's SSD tier will use its own scheme).
    fp = hashlib.sha256()
    fp.update(f"arch={architecture};dtype={dtype};".encode())
    fp.update(hashlib.sha256(config_bytes).digest())
    for f in weight_files:
        fp.update(f"{f.name}:{f.stat().st_size};".encode())

    template = _chat_template(model_dir)
    template_hash = (
        hashlib.sha256(template.encode("utf-8")).hexdigest() if template else ""
    )

    return ModelInfo(
        model_path=str(model_dir),
        architecture=architecture,
        dtype=dtype,
        max_context_len=int(config.get("max_position_embeddings", 0)),
        vocab_size=int(config.get("vocab_size", 0)),
        weights_bytes=weights_bytes,
        weights_fingerprint=fp.hexdigest(),
        chat_template_hash=template_hash,
    )
