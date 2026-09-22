"""
Builds a BLIP image-captioning model with a LoRA adapter applied to the
attention query/key projections, following the setup used in the notebook.
"""

from peft import LoraConfig, get_peft_model
from transformers import BlipForConditionalGeneration, BlipProcessor


def build_model_and_processor(cfg: dict):
    """Load BLIP + processor and wrap the model with LoRA.

    Args:
        cfg: the "model" section of configs/config.yaml
    Returns:
        (model, processor)
    """
    base_model = cfg["base_model"]
    processor = BlipProcessor.from_pretrained(base_model)
    model = BlipForConditionalGeneration.from_pretrained(base_model)

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    return model, processor


def count_parameters(model) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.3f}% of total)")
