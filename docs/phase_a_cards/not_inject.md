# Dataset Card: leolee99 / NotInject

- **URL**: `https://huggingface.co/datasets/leolee99/NotInject`
- **Raw URL**: `https://huggingface.co/datasets/leolee99/NotInject/raw/main/README.md`
- **Fetched Timestamp**: `2026-10-20T16:22:35Z`

---

```markdown
---
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: word_list
    sequence: string
  - name: category
    dtype: string
  splits:
  - name: NotInject_one
    num_bytes: 14194
    num_examples: 113
  - name: NotInject_two
    num_bytes: 15586
    num_examples: 113
  - name: NotInject_three
    num_bytes: 19697
    num_examples: 113
  download_size: 35051
  dataset_size: 49477
configs:
- config_name: default
  data_files:
  - split: NotInject_one
    path: data/NotInject_one-*
  - split: NotInject_two
    path: data/NotInject_two-*
  - split: NotInject_three
    path: data/NotInject_three-*
license: mit
task_categories:
- text-classification
language:
- en
pretty_name: NotInject
size_categories:
- n<1K
---


## InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models

[Website](injecguard.github.io), [Paper](https://arxiv.org/pdf/2410.22770), [Code](https://github.com/SaFoLab-WISC/InjecGuard), [Demo](injecguard.github.io)

## Dataset Description

The **NotInject** is a benchmark designed to evaluate the extent of over-defense in existing prompt guard models against prompt injection. All samples in the dataset are benign but contain trigger words that may be mistakenly flagged as risky. The dataset is divided into three subsets, each consisting of prompts generated using one, two, or three trigger words respectively.

## Dataset Structure

- **prompt**: The text input containing the trigger words.
- **word_list**: A list of trigger words used to construct the prompt.
- **category**: The topic category of the prompt, with four categories——`Common Queries`, `Technique Queries`, `Virtual Creation`, and `Multilingual Queries`.

## Dataset Statistics

**Sample Number**: 113 per subset

| Category           | one-word | two-word | three-word |
|--------------------|:--------:|:--------:|:----------:|
| Common Queries     | 58       | 49       | 19         |
| Techniques Queries | 16       | 30       | 41         |
| Virtual Creation   | 14       | 4        | 24         |
| Multilingual Queries | 25       | 30       | 29         |

## Reference

If you find this work useful in your research or applications, we appreciate that if you can kindly cite:
```
@articles{InjecGuard,
  title={InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models},
  author={Hao Li and Xiaogeng Liu},
  journal={arXiv preprint arXiv:2410.22770},
  year={2024}
}
```
```
