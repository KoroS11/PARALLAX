#!/usr/bin/env python3
"""Verify that every italic quote in EXTERNAL_DATASET_REVIEW.md is a verbatim substring of a saved card (K12e Item 5).

Strictly follows:
- Reads docs/EXTERNAL_DATASET_REVIEW.md.
- Extracts all quoted substrings marked in italics.
- Checks each against the corresponding file in docs/phase_a_cards/.
- Prints PASS/FAIL per quote and returns nonzero exit code if any fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REVIEW_PATH = REPO_ROOT / "docs" / "EXTERNAL_DATASET_REVIEW.md"
CARDS_DIR = REPO_ROOT / "docs" / "phase_a_cards"


def verify_quotes():
    if not REVIEW_PATH.exists():
        print(f"ERROR: {REVIEW_PATH} not found")
        sys.exit(1)

    review_text = REVIEW_PATH.read_text(encoding="utf-8")

    # Match italic quotes: *"..."*
    pattern = re.compile(r'\*"([^"]+)"\*')
    quotes = pattern.findall(review_text)

    if not quotes:
        print("ERROR: No italic quotes found in EXTERNAL_DATASET_REVIEW.md")
        sys.exit(1)

    # Load all card texts
    card_texts = {}
    for card_file in CARDS_DIR.glob("*.md"):
        card_texts[card_file.stem] = card_file.read_text(encoding="utf-8")

    print(f"Loaded {len(card_texts)} saved card files from {CARDS_DIR}")
    print(f"Verifying {len(quotes)} italic quotes from {REVIEW_PATH.name}...\n")

    all_passed = True
    for i, q in enumerate(quotes, 1):
        # Find which card contains q
        matched_cards = [name for name, text in card_texts.items() if q in text]
        if matched_cards:
            print(f"[{i:02d}] PASS | Matches: {', '.join(matched_cards)}")
            print(f'     Quote: "{q[:80]}{"..." if len(q) > 80 else ""}"')
        else:
            print(f"[{i:02d}] FAIL | No card contains quote:")
            print(f'     Quote: "{q}"')
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print(
            f"VERIFICATION SUCCESS: All {len(quotes)} quotes are verbatim substrings."
        )
        sys.exit(0)
    else:
        print("VERIFICATION FAILURE: Some quotes failed verification.")
        sys.exit(1)


if __name__ == "__main__":
    verify_quotes()
