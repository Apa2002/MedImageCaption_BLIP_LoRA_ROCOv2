# Medical Image Captioning with BLIP + LoRA on ROCOv2

A limited-data experiment by **Arezoo Pasand** exploring the adaptation of a pretrained vision-language model to medical image captioning.

**Status:** six training epochs and evaluation on the full 9,927-image test split are complete. This repository reports the observed results, including remaining caption errors. The experiment used 10% of the training split under a limited GPU budget.

## Method

The experiment adapts a pretrained BLIP image-captioning model using LoRA on query and key projections. RGB images are processed with BlipProcessor. Captions use dynamic padding, attention masks, and ignored padding labels. Training starts with the decoder BOS token. Evaluation reloads the saved adapter before generating captions.

## Experiment setup

| Setting | Value |
|---|---|
| Base checkpoint | `Salesforce/blip-image-captioning-base` |
| Training subset | 5,996 images; 10% of the training split |
| Test set | 9,927 images; full test split |
| Epochs / seed | 6 / 42 |
| Total / trainable parameters | 224,266,556 / 294,912 |
| Trainable scope | Query/key LoRA adapters; base model frozen |
| LoRA rank / alpha / dropout | 8 / 16 / 0.1 |
| LoRA targets | `attention.self.query`, `attention.self.key` |
| Training batch size | 3 |
| Learning rate / weight decay | 5e-5 / 0.01 |
| Training precision | FP32 |
| Training text limit | 512 tokens |
| Decoding | Beam search, 4 beams |
| Generation maximum length | 128 tokens |
| Scheduler / warm-up | Linear / 0 |
| Checkpoint selection | Last completed epoch |
| GPU | Tesla T4 |

## Results

| Metric | Score |
|---|---:|
| BLEU-1 | 0.1461 |
| BLEU-2 | 0.0655 |
| BLEU-3 | 0.0225 |
| BLEU-4 | 0.0078 |
| METEOR | 0.0475 |
| ROUGE-L | 0.1302 |
| CIDEr | 0.0351 |

Scores are reported as printed by the executed notebook, rounded to four decimal places; they are not percentages. Evaluation uses the COCO PTB tokenizer and pycocoevalcap for BLEU, METEOR, ROUGE-L, and CIDEr. Predictions are compared with the original reference captions.

### Training history

| Epoch | Training loss |
|---|---:|
| 1 | 6.466787 |
| 2 | 6.145403 |
| 3 | 6.033801 |
| 4 | 5.968374 |
| 5 | 5.915012 |
| 6 | 5.904488 |

Training loss decreased during the run. This measures fit to the training data; it does not establish generalization or rule out overfitting.

### Qualitative observations and limitations

Some displayed captions describe incorrect objects or anatomy, and several repeat phrases. One displayed prediction is "an image of a bird on a black background" for a medical image. These examples show that caption content and repetition remain limitations of this run.

- This is a single run on a 10% training subset. The reported results do not establish which factors caused the remaining errors.
- Training used a fixed six-epoch protocol, with no validation-based checkpoint selection or early stopping.
- A zero-shot baseline and repeated-seed experiments were not run. No improvement over the pretrained base model is claimed.
- The companion BLIP and ViT-GPT2 runs differ in trainable parameter count and decoding settings. Their scores are not a controlled comparison of the two architectures.

## Files

- [Executed notebook](notebooks/BLIP_LoRA_ROCOv2_Executed.ipynb): the complete English notebook, including source-writing cells, training logs, saved metric outputs, and displayed predictions.
- [Metric scores](results/metrics.json): scores transcribed from the saved notebook output at its printed precision.
- [Training history](results/training_history.csv): epoch losses transcribed from the saved notebook output.
- [Run summary](results/run_summary.json): settings, parameter counts, runtime versions, and evidence provenance.

The executed notebook is preserved byte-for-byte from the completed run. Viewing its saved outputs does not require a GPU. Dataset images, trained checkpoints, and the full per-image predictions CSV are not included in this results package.

## Notebook execution

The notebook embeds the project modules and writes them into its working directory. To reproduce the experiment, use a fresh Kaggle session, review the configuration and dependency cells, supply or download ROCOv2 and the base checkpoint, and execute the cells in order. Follow any kernel-restart instruction from the installation cell. Training is optional for readers who only want to inspect the saved results.

The reported runtime used Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.57.6, PEFT 0.18.1, Accelerate 1.12.0, and pycocoevalcap 1.2. Java was available for PTB tokenization and METEOR. A single completed run is reported; exact reproducibility across different environments has not been established.
