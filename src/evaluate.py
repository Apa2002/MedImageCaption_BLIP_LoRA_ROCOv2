"""
Generates captions on the ROCOv2 test set and scores them with
BLEU / METEOR / ROUGE-L / CIDEr, as in the original notebook.

Usage:
    python -m src.evaluate --config configs/config.yaml
"""

import argparse

import nltk
import pandas as pd
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from src.data.dataset import ImageCaptioningDataset, MergeImageAndCaption, collate_fn
from src.utils import get_device, load_config


def generate_captions(model, dataloader, processor, device, eval_cfg):
    model.eval()
    generated_captions, gt_captions_text = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Captions"):
            pixel_values = batch["pixel_values"].to(device)
            output_ids = model.generate(
                pixel_values=pixel_values,
                num_beams=eval_cfg["num_beams"],
                max_length=eval_cfg["max_length"],
                early_stopping=True,
            )
            generated_captions.extend(processor.batch_decode(output_ids, skip_special_tokens=True))
            gt_captions_text.extend(processor.batch_decode(batch["input_ids"], skip_special_tokens=True))

    return generated_captions, gt_captions_text


def score_captions(gt_captions_text, generated_captions):
    gt = {i: [c] for i, c in enumerate(gt_captions_text)}
    gen = {i: [c] for i, c in enumerate(generated_captions)}

    bleu_scores, _ = Bleu(4).compute_score(gt, gen)
    meteor_score, _ = Meteor().compute_score(gt, gen)
    rouge_score, _ = Rouge().compute_score(gt, gen)
    cider_score, _ = Cider().compute_score(gt, gen)

    return {
        "BLEU-1": bleu_scores[0],
        "BLEU-2": bleu_scores[1],
        "BLEU-3": bleu_scores[2],
        "BLEU-4": bleu_scores[3],
        "METEOR": meteor_score,
        "ROUGE-L": rouge_score,
        "CIDEr": cider_score,
    }


def main(cfg_path: str):
    nltk.download("punkt")
    cfg = load_config(cfg_path)
    paths, eval_cfg = cfg["paths"], cfg["evaluation"]
    device = get_device()

    processor = BlipProcessor.from_pretrained(cfg["model"]["base_model"])
    model = BlipForConditionalGeneration.from_pretrained(paths["output_dir"])
    model.to(device)

    df_test = pd.read_csv(paths["test_csv"])
    dataset_test = MergeImageAndCaption(df_test, paths["test_images_dir"])
    test_dataset = ImageCaptioningDataset(dataset_test, processor)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    generated, ground_truth = generate_captions(model, test_dataloader, processor, device, eval_cfg)
    scores = score_captions(ground_truth, generated)

    for name, value in scores.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
