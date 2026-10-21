# Model Card: ProtectAI / deberta-v3-base-prompt-injection-v2

- **URL**: `https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2`
- **Raw URL**: `https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2/raw/main/README.md`
- **Fetched Timestamp**: `2026-10-20T16:32:58Z`

---

```markdown
---
license: apache-2.0
base_model: microsoft/deberta-v3-base
language:
- en
datasets:
- natolambert/xstest-v2-copy
- VMware/open-instruct
- alespalla/chatbot_instruction_prompts
- HuggingFaceH4/grok-conversation-harmless
- Harelix/Prompt-Injection-Mixed-Techniques-2024
- OpenSafetyLab/Salad-Data
- jackhhao/jailbreak-classification
---

# Model Card for deberta-v3-base-prompt-injection-v2

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) specifically developed to detect and classify prompt injection attacks which can manipulate language models into producing unintended outputs.

### Dataset

The dataset used for training the model was meticulously assembled from various public open datasets to include a wide range of prompt variations. Additionally, prompt injections were crafted using insights gathered from academic research papers, articles, security competitions, and valuable LLM Guard's community feedback.

- **CC-BY-3.0:** 1 dataset (`VMware/open-instruct`)
- **MIT License:** 8 datasets
- **CC0 1.0 Universal:** 1 dataset
- **No License (public domain):** 6 datasets
- **Apache License 2.0:** 5 datasets (`alespalla/chatbot_instruction_prompts`, `HuggingFaceH4/grok-conversation-harmless`, `Harelix/Prompt-Injection-Mixed-Techniques-2024`, `OpenSafetyLab/Salad-Data`, `jackhhao/jailbreak-classification`)
- **CC-BY-4.0:** 1 dataset (`natolambert/xstest-v2-copy:1_full_compliance`)
```
