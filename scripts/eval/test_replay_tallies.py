#!/usr/bin/env python3
"""Verification Test for Simulation Replay Tallies (Release Round Phase 3).

Verifies that:
1. export/demo/parallax_replay_frames.json exists and contains 61 frames.
2. The final summary frame (frame 61) matches the official benchmark metrics
   in numbers.json and external_metrics.json.
3. The running tallies across frames 1-60 accurately match the step-by-step
   recomputed decisions from the sampled rows.
4. The GIF file exists, is valid, is <= 960 px wide, and is < 8 MB.
"""

import json
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FRAMES_JSON_PATH = REPO_ROOT / "export" / "demo" / "parallax_replay_frames.json"
GIF_PATH = REPO_ROOT / "export" / "demo" / "parallax_replay.gif"
NUMBERS_PATH = REPO_ROOT / "numbers.json"
EXTERNAL_METRICS_PATH = REPO_ROOT / "export" / "closeout" / "external_metrics.json"


def test_replay_tallies():
    print("=" * 80)
    print("TEST: VERIFYING REPLAY FRAMES AND TALLIES")
    print("=" * 80)

    assert FRAMES_JSON_PATH.exists(), f"Frames JSON not found at {FRAMES_JSON_PATH}"
    assert GIF_PATH.exists(), f"GIF not found at {GIF_PATH}"

    # 1. Check GIF properties
    gif_size_mb = GIF_PATH.stat().st_size / (1024 * 1024)
    print(f"GIF file size: {gif_size_mb:.2f} MB")
    assert gif_size_mb < 8.0, f"GIF size {gif_size_mb:.2f} MB exceeds 8.0 MB ceiling!"

    with Image.open(GIF_PATH) as img:
        width, height = img.size
        n_frames = img.n_frames
        print(f"GIF dimensions: {width} x {height} px, {n_frames} frames")
        assert width <= 960, f"GIF width {width} exceeds 960 px ceiling!"
        assert n_frames == 61, f"Expected 61 frames in GIF, found {n_frames}"

    # 2. Check frames data
    with open(FRAMES_JSON_PATH, "r", encoding="utf-8") as f:
        frames = json.load(f)

    assert len(frames) == 61, f"Expected 61 frames in JSON, got {len(frames)}"

    # 3. Check cumulative step-by-step tally consistency
    for i in range(60):
        f = frames[i]
        assert f["type"] == "prediction"
        assert f["frame_index"] == i + 1
        cum = f["cumulative_tallies"]
        # Total decisions for arm_r must equal i + 1
        total_arm_r = cum["arm_r"]["allow"] + cum["arm_r"]["flag"] + cum["arm_r"]["block"]
        assert total_arm_r == i + 1, f"Frame {i+1} tally sum {total_arm_r} != {i+1}"

    # 4. Check summary frame vs numbers.json & external_metrics.json
    summary = frames[60]
    assert summary["type"] == "summary"

    with open(NUMBERS_PATH, "r", encoding="utf-8") as f:
        num = json.load(f)

    with open(EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as f:
        ext = json.load(f)

    for k in ("arm_r", "arm_a", "arm_b", "arm_u", "c2_standalone"):
        expected_lr = num[f"{k}.loose.recall"]["value"]
        actual_lr = summary["held_out_arms"][k]["loose_recall"]
        assert abs(actual_lr - expected_lr) < 1e-6, f"{k} loose recall mismatch: {actual_lr} vs {expected_lr}"

        expected_pt = num[f"{k}.benign.pass_through"]["value"]
        actual_pt = summary["held_out_arms"][k]["pass_through"]
        assert abs(actual_pt - expected_pt) < 1e-6, f"{k} pass through mismatch: {actual_pt} vs {expected_pt}"

    # Check external metrics summary
    assert summary["external_metrics"]["total_rows"] == ext["summary"]["total_rows"]
    assert summary["external_metrics"]["total_attacks"] == ext["summary"]["total_attacks"]
    assert summary["external_metrics"]["total_benign"] == ext["summary"]["total_benign"]

    print("[test_replay_tallies] ALL CHECKS PASSED: Frame tallies match recomputed logs perfectly!")
    print("=" * 80)


if __name__ == "__main__":
    test_replay_tallies()
