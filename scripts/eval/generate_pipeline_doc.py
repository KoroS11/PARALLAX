#!/usr/bin/env python3
"""Generate docs/PIPELINE.md and export SVG diagram (K6).

Verifies that every pipeline node maps directly to an existing repository file.
Generates:
1. docs/PIPELINE.md with Mermaid architecture diagram and stage table.
2. docs/pipeline.svg with high-resolution vector flowchart.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_MD_PATH = REPO_ROOT / "docs" / "PIPELINE.md"
PIPELINE_SVG_PATH = REPO_ROOT / "docs" / "pipeline.svg"

# Pipeline stages definition
STAGES = [
    {
        "id": "node_mask",
        "name": "Input Token Masking",
        "file": "eval/mask.py",
        "type": "Rule-based regex / Normalizer",
        "provenance": "Handcrafted rules",
        "input": "Raw untrusted prompt string",
        "output": "Masked prompt string (neutral token placeholders)",
        "evidence": "tests/test_eval_mask.py (prevents detector shortcut overfitting)",
    },
    {
        "id": "node_d2",
        "name": "D2 DeBERTa-v3 Detector",
        "file": "parallax/detectors/deberta.py",
        "type": "Pretrained Transformer + Platt scaling",
        "provenance": "protectai/deberta-v3-base-prompt-injection-v2",
        "input": "Masked prompt string",
        "output": "Calibrated probability score in [0.0, 1.0]",
        "evidence": "tests/test_hf_detectors.py (Platt calibrator: val AUC 0.7500)",
    },
    {
        "id": "node_d3",
        "name": "D3 Deepset Injection Detector",
        "file": "scripts/detector_services/third_detector_service.py",
        "type": "Pretrained Transformer + Logit-Diff Platt scaling",
        "provenance": "deepset/deberta-v3-base-injection",
        "input": "Masked prompt string",
        "output": "Calibrated probability score in [0.0, 1.0]",
        "evidence": "tests/test_promptforest.py (saturation reduced from 95.0% to 16.2%)",
    },
    {
        "id": "node_d4",
        "name": "D4 Heuristic Rule Detector",
        "file": "parallax/detectors/rules.py",
        "type": "Regex patterns / Obfuscation rules",
        "provenance": "Handcrafted in-house ruleset",
        "input": "Masked prompt string",
        "output": "Binary detection score in {0.0, 1.0}",
        "evidence": "tests/test_rules.py (100% precision on benign controls)",
    },
    {
        "id": "node_fusion",
        "name": "Primary AnyFlag Fusion",
        "file": "parallax/orchestrator/fusion.py",
        "type": "Ensemble Composition Heuristic",
        "provenance": "Trained / composed here",
        "input": "Detector score vector [d2, d3, d4]",
        "output": "Preliminary decision {ALLOW, BLOCK}, confidence",
        "evidence": "tests/test_orchestrator.py (AnyFlag conservative boundary)",
    },
    {
        "id": "node_router",
        "name": "Learned Review Router",
        "file": "parallax/router/learned_router.py",
        "type": "Supervised Error Predictor (Logistic / LightGBM)",
        "provenance": "Trained on dev/val features",
        "input": "Feature vector [d2, d3, d4, discrepancy, confidence, enc, len]",
        "output": "should_route (bool), prob_error float",
        "evidence": "no ablation run (feature selection unablated; held-out AUC 0.432 vs label)",
    },
    {
        "id": "node_judge",
        "name": "Adversarial LLM Judge",
        "file": "parallax/challenger/judge.py",
        "type": "Local 4B LLM (Ollama)",
        "provenance": "gemma3:4b (temperature 0.0, 128 max tokens)",
        "input": "Masked prompt string + review system template",
        "output": "JudgeResult: verdict {ALLOW, BLOCK, ABSTAIN}, quote, reasoning",
        "evidence": "tests/test_challenger_judge.py (timeout extended to 60.0s)",
    },
    {
        "id": "node_policy",
        "name": "Authority Policy B Mapper",
        "file": "parallax/challenger/schemas.py",
        "type": "Asymmetric Decision Mapping Logic",
        "provenance": "Handcrafted policy rules",
        "input": "Fusion decision + JudgeResult",
        "output": "Final decision {ALLOW, FLAG, BLOCK}",
        "evidence": "tests/test_challenger.py (blocks downgraded to FLAG only on certified allow)",
    },
]


def generate_pipeline_svg() -> str:
    """Generate high-quality SVG flowchart for the PARALLAX pipeline."""
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 620" width="100%" height="100%" style="background:#0f172a; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;">
  <defs>
    <linearGradient id="primaryGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#334155"/>
    </linearGradient>
    <linearGradient id="accentGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0369a1"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#38bdf8"/>
    </marker>
  </defs>

  <!-- Title -->
  <text x="480" y="35" text-anchor="middle" fill="#f8fafc" font-size="18" font-weight="bold">PARALLAX Multi-Model Detection &amp; Routing Architecture</text>
  <text x="480" y="55" text-anchor="middle" fill="#94a3b8" font-size="12">Component Lineage &amp; Execution Pipeline (Arm R with Policy B)</text>

  <!-- Stage 1: Input & Masking -->
  <rect x="50" y="90" width="220" height="70" rx="8" fill="url(#primaryGrad)" stroke="#38bdf8" stroke-width="2"/>
  <text x="160" y="120" text-anchor="middle" fill="#38bdf8" font-size="13" font-weight="bold">Input Preprocessing</text>
  <text x="160" y="140" text-anchor="middle" fill="#cbd5e1" font-size="11">eval/mask.py</text>

  <!-- Flow Arrow to Detectors -->
  <line x1="270" y1="125" x2="340" y2="125" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Stage 2: Parallel Detectors Container -->
  <rect x="340" y="80" width="270" height="230" rx="10" fill="#1e293b" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 4"/>
  <text x="475" y="102" text-anchor="middle" fill="#94a3b8" font-size="11" font-weight="bold">PARALLEL DETECTOR LAYER</text>

  <!-- D2 -->
  <rect x="360" y="115" width="230" height="45" rx="6" fill="#0f172a" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="475" y="133" text-anchor="middle" fill="#f1f5f9" font-size="11" font-weight="bold">D2: DeBERTa-v3 (Platt Scaled)</text>
  <text x="475" y="149" text-anchor="middle" fill="#94a3b8" font-size="10">parallax/detectors/deberta.py</text>

  <!-- D3 -->
  <rect x="360" y="170" width="230" height="45" rx="6" fill="#0f172a" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="475" y="188" text-anchor="middle" fill="#f1f5f9" font-size="11" font-weight="bold">D3: Deepset (Logit-Diff Scaled)</text>
  <text x="475" y="204" text-anchor="middle" fill="#94a3b8" font-size="10">scripts/detector_services/third_detector_service.py</text>

  <!-- D4 -->
  <rect x="360" y="225" width="230" height="45" rx="6" fill="#0f172a" stroke="#0ea5e9" stroke-width="1.5"/>
  <text x="475" y="243" text-anchor="middle" fill="#f1f5f9" font-size="11" font-weight="bold">D4: Heuristic Ruleset</text>
  <text x="475" y="259" text-anchor="middle" fill="#94a3b8" font-size="10">parallax/detectors/rules.py</text>

  <!-- Flow to Fusion and Router -->
  <line x1="610" y1="195" x2="680" y2="195" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Stage 3: Primary AnyFlag Fusion -->
  <rect x="680" y="115" width="230" height="65" rx="8" fill="url(#primaryGrad)" stroke="#a855f7" stroke-width="2"/>
  <text x="795" y="142" text-anchor="middle" fill="#c084fc" font-size="13" font-weight="bold">AnyFlag Primary Fusion</text>
  <text x="795" y="162" text-anchor="middle" fill="#cbd5e1" font-size="11">parallax/orchestrator/fusion.py</text>

  <!-- Stage 4: Learned Review Router -->
  <rect x="680" y="215" width="230" height="65" rx="8" fill="url(#primaryGrad)" stroke="#f59e0b" stroke-width="2"/>
  <text x="795" y="242" text-anchor="middle" fill="#fbbf24" font-size="13" font-weight="bold">Learned Review Router</text>
  <text x="795" y="262" text-anchor="middle" fill="#cbd5e1" font-size="11">parallax/router/learned_router.py</text>

  <!-- Branching: Fast Path vs Routed Review -->
  <!-- Fast Path (Unrouted) -->
  <path d="M 910 148 L 940 148 L 940 470 L 610 470" fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="4 4" marker-end="url(#arrow)"/>
  <text x="945" y="310" fill="#94a3b8" font-size="10" transform="rotate(90 945,310)">Unrouted Pass (27.7%)</text>

  <!-- Router Decision Path -->
  <line x1="795" y1="280" x2="795" y2="350" stroke="#f59e0b" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="805" y="320" fill="#fbbf24" font-size="11">should_route == True (72.3%)</text>

  <!-- Stage 5: Live LLM Judge -->
  <rect x="680" y="350" width="230" height="75" rx="8" fill="url(#primaryGrad)" stroke="#ec4899" stroke-width="2"/>
  <text x="795" y="377" text-anchor="middle" fill="#f472b6" font-size="13" font-weight="bold">Adversarial LLM Judge</text>
  <text x="795" y="395" text-anchor="middle" fill="#cbd5e1" font-size="11">gemma3:4b (Local Ollama)</text>
  <text x="795" y="411" text-anchor="middle" fill="#94a3b8" font-size="10">parallax/challenger/judge.py</text>

  <!-- Flow to Policy B -->
  <line x1="680" y1="387" x2="550" y2="387" stroke="#38bdf8" stroke-width="2"/>
  <line x1="550" y1="387" x2="550" y2="440" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>

  <!-- Stage 6: Policy B Authority -->
  <rect x="380" y="440" width="230" height="75" rx="8" fill="url(#accentGrad)" stroke="#38bdf8" stroke-width="2"/>
  <text x="495" y="468" text-anchor="middle" fill="#ffffff" font-size="13" font-weight="bold">Policy B Asymmetric Authority</text>
  <text x="495" y="488" text-anchor="middle" fill="#e0f2fe" font-size="11">Certified Downgrade (Block -> Flag)</text>
  <text x="495" y="504" text-anchor="middle" fill="#93c5fd" font-size="10">parallax/challenger/schemas.py</text>

  <!-- Final Decision -->
  <line x1="495" y1="515" x2="495" y2="560" stroke="#38bdf8" stroke-width="2" marker-end="url(#arrow)"/>
  <rect x="395" y="560" width="200" height="40" rx="6" fill="#10b981" stroke="#059669" stroke-width="2"/>
  <text x="495" y="585" text-anchor="middle" fill="#ffffff" font-size="13" font-weight="bold">FINAL: ALLOW / FLAG / BLOCK</text>
</svg>
"""
    return svg


def generate_pipeline_markdown() -> str:
    rows = []
    for s in STAGES:
        rows.append(
            f"| **{s['name']}** | [`{s['file']}`](file:///{REPO_ROOT.as_posix()}/{s['file']}) | {s['type']} | {s['input']} | {s['output']} | {s['evidence']} |"
        )

    mermaid_code = """```mermaid
flowchart TD
    In[Raw Input] --> M[eval/mask.py: Neutral Token Masking]
    M --> D2[parallax/detectors/deberta.py: D2 DeBERTa-v3]
    M --> D3[scripts/detector_services/third_detector_service.py: D3 Deepset]
    M --> D4[parallax/detectors/rules.py: D4 Heuristic Rules]

    D2 --> Fus[parallax/orchestrator/fusion.py: AnyFlag Fusion]
    D3 --> Fus
    D4 --> Fus

    D2 --> Rtr[parallax/router/learned_router.py: Learned Router]
    D3 --> Rtr
    D4 --> Rtr

    Rtr -- "should_route == True" --> Jdg[parallax/challenger/judge.py: LLM Judge gemma3:4b]
    Rtr -- "should_route == False" --> Out_Fast[Fast Decision: Preserves Fusion]

    Fus --> Pol[parallax/challenger/schemas.py: Policy B Asymmetric Mapper]
    Jdg --> Pol

    Pol --> Out[Final Decision: ALLOW / FLAG / BLOCK]
```"""

    md = rf"""# PARALLAX End-to-End System Pipeline (Stage Architecture)

## 1. Architectural Flowchart

![PARALLAX Pipeline Flowchart](pipeline.svg)

{mermaid_code}

---

## 2. Component Lineage & Implementation Registry

| Pipeline Stage | Implementation File | Component Type & Provenance | Input Specification | Output Specification | Scientific Evidence & Ablation Status |
|---|---|---|---|---|---|
{chr(10).join(rows)}

---

## 3. Evidence and Ablation Status Summary

- **Masking (`eval/mask.py`)**: Verified in `tests/test_eval_mask.py`. Strips synthetic marker shortcuts and normalizes industrial payloads.
- **D2 DeBERTa (`parallax/detectors/deberta.py`)**: Platt scaled on dev partition ($\sigma(1.4019 \cdot z - 0.3804)$). Evidence in `docs/DETECTOR_CALIBRATION.md`.
- **D3 Deepset (`scripts/detector_services/third_detector_service.py`)**: Logit-difference Platt scaled. Evidence in `docs/DETECTOR_CALIBRATION.md` (saturation reduced from 95.0% to 16.2%).
- **D4 Rules (`parallax/detectors/rules.py`)**: Heuristic patterns verified in `tests/test_rules.py`.
- **Primary Fusion (`parallax/orchestrator/fusion.py`)**: AnyFlag fusion combining calibrated detector outputs.
- **Learned Router (`parallax/router/learned_router.py`)**: Logistic regression on dev/val features. **Ablation status**: **no ablation run** (feature set chosen empirically without ablation). Held-out evaluation confirmed inverted error prediction (AUC 0.432 vs label).
- **Adversarial Judge (`parallax/challenger/judge.py`)**: Ollama `gemma3:4b` arbiter with 60.0s client timeout cap.
- **Authority Policy B (`parallax/challenger/schemas.py`)**: Asymmetric decision mapper guaranteeing that certified blocks cannot be downgraded to ALLOW without review.
"""
    return md


def main() -> int:
    print(
        "[generate_pipeline_doc] Verifying that every diagram node maps to an existing file..."
    )
    failures: list[str] = []

    for s in STAGES:
        file_path = REPO_ROOT / s["file"]
        if not file_path.exists():
            failures.append(
                f"Node '{s['name']}' specifies non-existent file: {s['file']}"
            )
        else:
            print(f"[generate_pipeline_doc] PASS: {s['name']} -> {s['file']}")

    if failures:
        print(
            "[generate_pipeline_doc] TEST FAILED: Broken node mappings found:",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("[generate_pipeline_doc] Writing docs/PIPELINE.md...")
    md_content = generate_pipeline_markdown()
    PIPELINE_MD_PATH.write_text(md_content, encoding="utf-8")
    print(f"[generate_pipeline_doc] Wrote {PIPELINE_MD_PATH}")

    print("[generate_pipeline_doc] Writing docs/pipeline.svg...")
    svg_content = generate_pipeline_svg()
    PIPELINE_SVG_PATH.write_text(svg_content, encoding="utf-8")
    print(f"[generate_pipeline_doc] Wrote {PIPELINE_SVG_PATH}")

    print("[generate_pipeline_doc] ALL K6 VERIFICATIONS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
