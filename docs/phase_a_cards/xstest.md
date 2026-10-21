# Dataset Card / API Metadata: walledai / XSTest

- **URL**: `https://huggingface.co/datasets/walledai/XSTest`
- **API URL**: `https://huggingface.co/api/datasets/walledai/XSTest`
- **Fetched Timestamp**: `2026-10-20T16:22:56Z`
- **Access Condition**: Gated (`gated: auto` on Hugging Face API)

---

```json
{
  "id": "walledai/XSTest",
  "author": "walledai",
  "sha": "f1d713187c61b6ae64e602d74f0b3d812cc2e8e8",
  "lastModified": "2024-07-04T23:16:07.000Z",
  "private": false,
  "gated": "auto",
  "disabled": false,
  "tags": [
    "language:en",
    "license:cc-by-4.0",
    "size_categories:n<1K",
    "format:parquet",
    "modality:text",
    "library:datasets",
    "library:pandas",
    "library:mlcroissant",
    "library:polars",
    "arxiv:2308.01263",
    "region:us"
  ],
  "description": "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. Paper: XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models. Data: xstest_prompts_v2. About: Without proper safeguards, large language models will follow malicious instructions and generate toxic content. This motivates safety efforts such as red-teaming and large-scale feedback learning, which aim to make models both helpful and safe.",
  "cardData": {
    "language": ["en"],
    "license": "cc-by-4.0",
    "task_categories": ["text2text-generation"],
    "pretty_name": "exaggerated safety",
    "dataset_info": {
      "features": [
        {"name": "prompt", "dtype": "string"},
        {"name": "focus", "dtype": "string"},
        {"name": "type", "dtype": "string"},
        {"name": "note", "dtype": "string"},
        {"name": "label", "dtype": "string"}
      ],
      "splits": [
        {"name": "test", "num_bytes": 43841, "num_examples": 450}
      ],
      "download_size": 16964,
      "dataset_size": 43841
    },
    "configs": [
      {
        "config_name": "default",
        "data_files": [{"split": "test", "path": "data/train-*"}]
      }
    ]
  },
  "siblings": [
    {"rfilename": ".gitattributes"},
    {"rfilename": "README.md"},
    {"rfilename": "data/train-00000-of-00001.parquet"}
  ],
  "usedStorage": 33490
}
```
