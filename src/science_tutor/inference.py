"""Optional Transformers runtime for greedy batched local or CUDA generation."""

from collections.abc import Callable
from importlib import import_module, metadata
from pathlib import Path
from typing import Literal

from science_tutor.config import ModelSettings
from science_tutor.prompts import PromptMessage

Device = Literal["auto", "cpu", "mps", "cuda"]


def load_generator(
    settings: ModelSettings,
    requested_device: Device,
    max_new_tokens: int,
    cache_dir: Path,
) -> tuple[Callable[[list[list[PromptMessage]]], list[str]], dict[str, str]]:
    """Load the pinned tokenizer and model and decode only newly generated tokens."""
    if max_new_tokens < 1:
        raise ValueError("Max new tokens must be positive.")
    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional runtime with uv sync --locked --extra inference."
        ) from exc

    cuda, mps = torch.cuda.is_available(), torch.backends.mps.is_available()
    device = requested_device
    if requested_device == "auto":
        device = "cuda" if cuda else "mps" if mps else "cpu"
    if (device == "cuda" and not cuda) or (device == "mps" and not mps):
        raise ValueError(f"Requested device {device!r} is unavailable.")
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        settings.model_id, revision=settings.revision, cache_dir=cache_dir, token=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        cache_dir=cache_dir,
        dtype=dtype,
        token=False,
        use_safetensors=True,
    ).to(device)
    model.eval()
    generation = transformers.GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    def generate(messages: list[list[PromptMessage]]) -> list[str]:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        width = inputs["input_ids"].shape[1]
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit is not None and width + max_new_tokens > context_limit:
            raise ValueError(
                f"Prompt and generation budget exceed model context ({context_limit})."
            )
        with torch.inference_mode():
            outputs = model.generate(**inputs, generation_config=generation)
        return list(tokenizer.batch_decode(outputs[:, width:].cpu(), skip_special_tokens=True))

    return generate, {
        "device": device,
        "dtype": str(dtype),
        "torch": str(torch.__version__),
        "transformers": metadata.version("transformers"),
    }
