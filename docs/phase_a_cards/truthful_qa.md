# Dataset Card / API Metadata: truthfulqa / truthful_qa

- **URL**: `https://huggingface.co/datasets/truthfulqa/truthful_qa`
- **API URL**: `https://huggingface.co/api/datasets/truthfulqa/truthful_qa`
- **Fetched Timestamp**: `2026-10-20T16:23:09Z`
- **Access Condition**: Open (`gated: false`)

---

```json
{
  "id": "truthfulqa/truthful_qa",
  "author": "truthfulqa",
  "sha": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
  "lastModified": "2024-01-04T16:36:00.000Z",
  "private": false,
  "gated": false,
  "disabled": false,
  "tags": [
    "task_categories:multiple-choice",
    "task_categories:text-generation",
    "task_categories:question-answering",
    "language:en",
    "license:apache-2.0",
    "size_categories:1K<n<10K",
    "format:parquet",
    "modality:text",
    "library:datasets",
    "arxiv:2109.07958",
    "region:us"
  ],
  "description": "Dataset Card for truthful_qa. Dataset Summary: TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts.",
  "cardData": {
    "language": ["en"],
    "license": ["apache-2.0"],
    "pretty_name": "TruthfulQA",
    "dataset_info": [
      {
        "config_name": "generation",
        "features": [
          {"name": "type", "dtype": "string"},
          {"name": "category", "dtype": "string"},
          {"name": "question", "dtype": "string"},
          {"name": "best_answer", "dtype": "string"},
          {"name": "correct_answers", "sequence": "string"},
          {"name": "incorrect_answers", "sequence": "string"},
          {"name": "source", "dtype": "string"}
        ],
        "splits": [
          {"name": "validation", "num_bytes": 473382, "num_examples": 817}
        ]
      }
    ]
  },
  "siblings": [
    {"rfilename": ".gitattributes"},
    {"rfilename": "README.md"},
    {"rfilename": "generation/validation-00000-of-00001.parquet"},
    {"rfilename": "multiple_choice/validation-00000-of-00001.parquet"}
  ],
  "usedStorage": 20547824
}
```
