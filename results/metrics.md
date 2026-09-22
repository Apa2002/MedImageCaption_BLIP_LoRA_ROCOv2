# Results

Fine-tuning **BLIP-base** with **LoRA** (query/key attention adapters) on **10% of the ROCOv2 training set**.

| Metric   | BLIP 4 Epochs | BLIP 6 Epochs | Without Fine-tune |
|----------|---------------|---------------|--------------------|
| BLEU-1   | 0.0470        | 0.0545        | 0.0433             |
| BLEU-2   | 0.0222        | 0.0255        | 0.0151             |
| METEOR   | 0.0270        | 0.0278        | 0.0171             |
| ROUGE-L  | 0.0882        | 0.0892        | 0.0812             |
| CIDEr    | 0.0387        | 0.0401        | 0.0056             |

Fine-tuning consistently improves every metric over the zero-shot (no fine-tune) baseline,
with 6 epochs slightly outperforming 4 epochs on all metrics. Scores are still low in
absolute terms — see "Future Improvements" in the main README for planned next steps
(more training data, longer schedules, alternate encoders, augmentation).
