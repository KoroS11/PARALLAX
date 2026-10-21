#!/usr/bin/env python3
"""Replace banned words systematically across markdown documents (K8).

Banned terms according to Closeout Plan v2:
- validated, unvalidated
- verified, unverified
- audited
- AUDIT_CRITERIA_SATISFIED
- integrity enforced
- signed off
- operational utility
- 0 timeouts
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXACT_REPLACEMENTS = [
    (re.compile(r"AUDIT_CRITERIA_SATISFIED", re.IGNORECASE), "measured_pending_review"),
    (re.compile(r"integrity\s+enforced", re.IGNORECASE), "measured under lock"),
    (re.compile(r"signed\s+off", re.IGNORECASE), "documented"),
    (re.compile(r"operational\s+utility", re.IGNORECASE), "benign pass-through"),
    (re.compile(r"0\s+timeouts", re.IGNORECASE), "outcome type not recorded"),
]

WORD_REPLACEMENTS = [
    # (pattern, uppercase, titlecase, lowercase)
    (r"\bunvalidated\b", "PRELIMINARY", "Preliminary", "preliminary"),
    (r"\bvalidated\b", "MEASURED", "Measured", "measured"),
    (r"\bunverified\b", "UNCONFIRMED", "Unconfirmed", "unconfirmed"),
    (r"\bverified\b", "CONFIRMED", "Confirmed", "confirmed"),
    (r"\baudited\b", "REVIEWED", "Reviewed", "reviewed"),
]


def replace_in_text(text: str) -> str:
    for reg, repl in EXACT_REPLACEMENTS:
        text = reg.sub(repl, text)

    for pat, u_rep, t_rep, l_rep in WORD_REPLACEMENTS:

        def repl_func(m: re.Match, u=u_rep, t=t_rep, l=l_rep) -> str:
            val = m.group(0)
            if val.isupper():
                return u
            elif val[0].isupper():
                return t
            else:
                return l

        text = re.sub(pat, repl_func, text, flags=re.IGNORECASE)

    return text


def main():
    target_files = glob.glob(str(REPO_ROOT / "docs/**/*.md"), recursive=True)
    target_files.extend(glob.glob(str(REPO_ROOT / "*.md")))

    total_changes = 0
    for fpath_str in target_files:
        fpath = Path(fpath_str)
        orig = fpath.read_text(encoding="utf-8")
        updated = replace_in_text(orig)
        if orig != updated:
            fpath.write_text(updated, encoding="utf-8")
            print(f"[replace_banned_words] Updated {fpath.relative_to(REPO_ROOT)}")
            total_changes += 1

    print(f"[replace_banned_words] Total files updated: {total_changes}")


if __name__ == "__main__":
    main()
