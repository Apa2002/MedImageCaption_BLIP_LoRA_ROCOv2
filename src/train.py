"""
Fine-tunes BLIP + LoRA on (a fraction of) the ROCOv2 training set.

Usage:
    python -m src.train --config configs/config.yaml
"""

import argparse
import os

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from src.data.dataset import ImageCaptioningDataset, MergeImageAndCaption, collate_fn
from src.models.blip_lora import build_model_and_processor, count_parameters
from src.utils import get_device, load_config


def train_one_epoch(model, dataloader, optimizer, lr_scheduler, device):
    model.train()
    total_loss = 0.0
    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    device = get_device()

    df_train = pd.read_csv(paths["train_csv"])
    df_train = df_train.sample(
        frac=data_cfg["train_fraction"], random_state=data_cfg["seed"]
    ).reset_index(drop=True)

    model, processor = build_model_and_processor(cfg["model"])
    count_parameters(model)
    model.to(device)

    dataset_train = MergeImageAndCaption(df_train, paths["train_images_dir"])
    train_dataset = ImageCaptioningDataset(dataset_train, processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    optimizer = AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    lr_scheduler = get_scheduler(
        train_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_cfg["warmup_steps"],
        num_training_steps=len(train_dataloader) * train_cfg["num_epochs"],
    )

    os.makedirs(paths["output_dir"], exist_ok=True)

    for epoch in range(train_cfg["num_epochs"]):
        print(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}")
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, lr_scheduler, device)
        print(f"Average training loss: {avg_loss:.4f}")

        model.save_pretrained(paths["output_dir"])
        torch.save(optimizer.state_dict(), os.path.join(paths["output_dir"], "optimizer_state.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(paths["output_dir"], "scheduler_state.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
