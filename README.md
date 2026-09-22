# Medical Image Captioning with BLIP (ROCOv2 Dataset)

This project implements **medical image captioning** using the **BLIP (Salesforce BLIP-Base)** model,
fine-tuned with **LoRA**, on the **ROCOv2** dataset. The goal is to generate textual descriptions
for medical images, including X-rays and CT scans.

---

## Repository Structure

```
medical-image-captioning-blip/
├── README.md                 # this file
├── LICENSE                   # MIT license for the code (data has its own license, see below)
├── requirements.txt          # Python dependencies
├── .gitignore
├── configs/
│   └── config.yaml           # paths + hyperparameters (single source of truth)
├── data/                     # NOT committed to git — populated by scripts/download_data.sh
│   └── .gitkeep
├── notebooks/
│   └── Part3_6Epoch_Finetune_10pct_Data.ipynb   # original exploratory notebook (Kaggle)
├── src/
│   ├── data/
│   │   ├── download.py       # downloads + extracts ROCOv2 from Zenodo
│   │   └── dataset.py        # Dataset / collate_fn classes
│   ├── models/
│   │   └── blip_lora.py      # BLIP + LoRA model builder
│   ├── train.py              # training entry point (python -m src.train)
│   ├── evaluate.py           # evaluation entry point (python -m src.evaluate)
│   └── utils.py               # config loading, device helpers
├── scripts/
│   ├── download_data.sh
│   └── run_train.sh          # train then evaluate, end to end
├── results/
│   └── metrics.md            # reported BLEU/METEOR/ROUGE-L/CIDEr scores
└── docs/
    └── images/                # figures/screenshots for the README (optional)
```

> **Note on the notebook:** only the "6-epoch, 10%-data" notebook was available when this
> repository was assembled. If you also have the earlier exploration/4-epoch notebooks,
> drop them into `notebooks/` (e.g. `Part1_Data_Preparation.ipynb`, `Part2_4Epoch_Finetune.ipynb`)
> so the full history is preserved — the scripts under `src/` already cover the same logic
> in reusable form, so the notebooks become supplementary rather than required.

---

## Project Steps

### 1. Data Loading and Preprocessing
- Load medical images and captions from **ROCOv2**.
- Resize images to **224×224 pixels**.
- Process captions using **BlipProcessor**.
- Split dataset into **training and test sets**.
- Use **10% of training data** for faster training on limited GPU.

### 2. Model Architecture (BLIP + LoRA)
- **Encoder**: Vision Transformer (BLIP)
- **Decoder**: Text generation with GPT-style decoder
- **LoRA Fine-Tuning**: Applied to attention layers (query & key) to reduce trainable parameters and memory usage.

### 3. Training
- **Optimizer**: AdamW
- **Scheduler**: Linear
- **Batch size**: 3
- **Epochs**: 4 and 6 (10% of data)
- Model and optimizer states are saved after each epoch.

### 4. Evaluation
- Generate captions for test images.
- **Metrics**: BLEU (1–4), METEOR, ROUGE-L, CIDEr.

### 5. Results

See [`results/metrics.md`](results/metrics.md):

| Metric   | BLIP 4 Epochs | BLIP 6 Epochs | Without Fine-tune |
|----------|---------------|---------------|--------------------|
| BLEU-1   | 0.0470        | 0.0545        | 0.0433             |
| BLEU-2   | 0.0222        | 0.0255        | 0.0151             |
| METEOR   | 0.0270        | 0.0278        | 0.0171             |
| ROUGE-L  | 0.0882        | 0.0892        | 0.0812             |
| CIDEr    | 0.0387        | 0.0401        | 0.0056             |

### 6. Future Improvements
- Train with **larger data (>10%)** for better results.
- Explore other **Vision Transformer encoders**.
- Tune hyperparameters and use longer epochs.
- Apply **data augmentation**.

---

## Getting Started

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/medical-image-captioning-blip.git
cd medical-image-captioning-blip
pip install -r requirements.txt
```

### 2. Download the dataset
```bash
bash scripts/download_data.sh
```
This downloads ROCOv2 from Zenodo and extracts it into `data/`.

### 3. Train and evaluate
```bash
bash scripts/run_train.sh
```
Or run each step separately:
```bash
python -m src.train --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml
```

All paths and hyperparameters (batch size, epochs, LoRA rank, etc.) live in
`configs/config.yaml`, so you don't need to touch the source code to tweak a run.

---

## Dataset
**ROCOv2: Radiology Objects in Context Version 2**
- 79,789 radiological images with captions and clinical concepts
- Seven clinical modalities, manually curated medical concepts
- Suitable for image captioning and multi-label classification

**Dataset Link**: [ROCOv2 Dataset](https://zenodo.org/records/10821435)

**Citation (APA):**
> Johannes Rückert, Louise Bloch, Raphael Brüngel, Ahmad Idrissi-Yaghir, Henning Schäfer, Cynthia S. Schmidt, Sven Koitka, Obioma Pelka, Asma Ben Abacha, Alba Garcia Seco de Herrera, Henning Müller, Peter A. Horn, Felix Nensa, & Christoph M. Friedrich. (2023). ROCOv2: Radiology Objects in COntext Version 2, An Updated Multimodal Image Dataset [Data set]. *Scientific Data (2.0.1)*. Zenodo. https://doi.org/10.5281/zenodo.10821435

### Dataset License
- **License**: Creative Commons Attribution Non-Commercial 4.0 International (CC BY-NC 4.0)
- The dataset may be used for **research and educational purposes only**; **commercial use is not allowed**.
- Users must **cite the original dataset** when using it in publications or projects.
- Official dataset page: [ROCOv2 Dataset](https://zenodo.org/records/10821435)

The **code** in this repository is released under the [MIT License](LICENSE); the dataset
itself keeps its own separate CC BY-NC 4.0 license as noted above.

---

## Requirements
See [`requirements.txt`](requirements.txt):
```
torch>=2.0
transformers>=4.10.3
datasets>=2.20.0
peft
bitsandbytes
pycocoevalcap
evaluate
sacrebleu
nltk
rouge-score
numpy
pandas
matplotlib
tqdm
Pillow
```
