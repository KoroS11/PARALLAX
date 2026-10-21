# Model Card: deepset / deberta-v3-base-injection

- **URL**: `https://huggingface.co/deepset/deberta-v3-base-injection`
- **Raw URL**: `https://huggingface.co/deepset/deberta-v3-base-injection/raw/main/README.md`
- **Fetched Timestamp**: `2026-10-20T16:32:49Z`

---

```markdown
---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
base_model: microsoft/deberta-v3-base
model-index:
- name: deberta-v3-base-injection
  results: []
datasets:
- deepset/prompt-injections
language:
- en
- de
---

# deberta-v3-base-injection

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on the [prompt-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0673
- Accuracy: 0.9914

## Model description

This model detects prompt injection attempts and classifies them as "INJECTION". Legitimate requests are classified as "LEGIT". The dataset assumes that legitimate requests are either all sorts of questions of key word searches.

## Intended uses & limitations

If you are using this model to secure your system and it is overly "trigger-happy" to classify requests as injections, consider collecting legitimate examples and retraining the model with the [promp-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset.

## Training and evaluation data

Based in the [promp-injection](https://huggingface.co/datasets/JasperLS/prompt-injections) dataset.
```
