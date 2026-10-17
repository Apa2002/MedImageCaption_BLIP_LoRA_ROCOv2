"""
Dataset classes for the ROCOv2 medical image captioning task.

These mirror the classes used in the training notebook
(notebooks/Part3_6Epoch_Finetune_10pct_Data.ipynb), moved here so they can be
imported/tested/reused instead of copy-pasted between notebooks.
"""

import os

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


class MergeImageAndCaption(Dataset):
    """Joins a captions dataframe with the corresponding image files on disk."""

    def __init__(self, dataframe, images_dir: str):
        self.dataset = dataframe.reset_index(drop=True)
        self.image_list = [
            os.path.join(images_dir, f"{file_id}.jpg")
            for file_id in self.dataset["ID"].tolist()
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        caption = self.dataset["Caption"][idx]
        return {"text": caption, "image": image}


class ImageCaptioningDataset(Dataset):
    """Wraps a MergeImageAndCaption dataset and runs it through BlipProcessor."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if not isinstance(image, torch.Tensor):
            image = IMAGE_TRANSFORM(image)

        encoding = self.processor(
            images=image,
            text=item["text"],
            return_tensors="pt",
            padding="max_length",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def collate_fn(batch, processor):
    """Stacks images and pads text tensors to the longest sequence in the batch."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id,
    )
    attention_masks = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }
