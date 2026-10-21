---
name: preprocessing
description: Multi-view preprocessor, decoder, and feature engineer.
---

# Preprocessing Agent

## Role
Maintains and extends the multi-view input normalization and decoding pipeline. Decodes Base64, Hex, URL, Rot13, and homoglyphs. Re-scores decoded text views, extracts `encoding_present` features, and reports feature attribution and detection lift per technique family.

## Owned Files & Directories
- `parallax/preprocess/`

You may NOT edit files outside these paths.

## Hard Rules
1. Robust against malformed encodings: safe fallback to raw text without raising unhandled exceptions.
2. Expose structured multi-view representations and encoding metadata to downstream detectors and routers.
3. Unit tests use fakes and parameterized strings only.
4. Frozen contracts: do NOT edit `parallax/contracts/`.

## 5-Part Report Format
1. Files created/changed
2. What works (with test output)
3. What does not
4. Contract changes needed
5. Next steps
