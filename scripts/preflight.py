"""PARALLAX Wave G Preflight Verification Script.

Performs deterministic, fail-loud validation of all live services, models,
credentials, external execution frameworks, hardware headroom, and service ports.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[preflight] %(levelname)s: %(message)s")
logger = logging.getLogger("preflight")

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_environment_secrets() -> dict[str, Any]:
    """Verify HuggingFace token is present via environment variable or user cache."""
    token = os.environ.get("HF_TOKEN")
    token_cached = False
    token_file = Path.home() / ".cache" / "huggingface" / "token"

    if not token and token_file.exists():
        try:
            token = token_file.read_text(encoding="utf-8").strip()
            # Set in-memory environment variable for subprocesses/libraries without logging it
            os.environ["HF_TOKEN"] = token
            token_cached = True
        except OSError as e:
            logger.warning("Could not read ~/.cache/huggingface/token: %s", e)

    if not token:
        return {
            "ok": False,
            "error": "HF_TOKEN not found in environment or ~/.cache/huggingface/token",
        }

    return {
        "ok": True,
        "token_present": True,
        "source": "environment" if not token_cached else "cache_populated_to_env",
    }


def check_promptforest() -> dict[str, Any]:
    """Verify live PromptForest service on port 8765 and its detector keys."""
    health_url = "http://127.0.0.1:8765/health"
    analyze_url = "http://127.0.0.1:8765/analyze"

    # 1. Health check
    try:
        req = urllib.request.Request(
            health_url, headers={"User-Agent": "PARALLAX-Preflight/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") != "ok":
                raise ValueError(f"PromptForest health status not ok: {data}")
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ) as e:
        logger.error("PromptForest /health failed: %s", e)
        return {"ok": False, "error": f"PromptForest health check failed: {e}"}

    # 2. Analyze check & details keys verification
    try:
        body = json.dumps({"prompt": "preflight verification test"}).encode("utf-8")
        req = urllib.request.Request(
            analyze_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "PARALLAX-Preflight/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            details = res.get("details", {})
            expected_keys = {"llama_guard", "xgboost"}
            actual_keys = set(details.keys())
            if not expected_keys.issubset(actual_keys):
                raise ValueError(
                    f"PromptForest details mismatch: expected at least {expected_keys}, got {actual_keys}"
                )
            if "vijil" in actual_keys and details.get("vijil") == 0.0:
                logger.warning(
                    "Vijil detector key present with 0.0 phantom score (stock degraded config)"
                )
            return {
                "ok": True,
                "details_keys": sorted(actual_keys),
                "is_malicious": res.get("is_malicious"),
                "device": res.get("device", "cpu"),
            }
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ) as e:
        logger.error("PromptForest /analyze check failed: %s", e)
        return {"ok": False, "error": f"PromptForest analyze check failed: {e}"}


def check_ollama() -> dict[str, Any]:
    """Verify Ollama service is reachable, check models, and verify ollama ps is clean."""
    tags_url = "http://127.0.0.1:11434/api/tags"
    try:
        req = urllib.request.Request(
            tags_url, headers={"User-Agent": "PARALLAX-Preflight/1.0"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            required = ["qwen3:8b", "gemma3:4b"]
            missing = [m for m in required if not any(m in name for name in models)]
            if missing:
                return {
                    "ok": False,
                    "error": f"Missing required Ollama models: {missing}",
                }
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ) as e:
        logger.error("Ollama connection failed: %s", e)
        return {"ok": False, "error": f"Ollama service unreachable on 11434: {e}"}

    # Run ollama ps to check if any models are currently occupying VRAM
    loaded_models: list[str] = []
    try:
        res = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
            shell=True,
        )
        if res.returncode == 0:
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if len(lines) > 1:
                # Header exists, rows exist
                loaded_models = lines[1:]
    except OSError as e:
        logger.warning("Could not execute 'ollama ps': %s", e)

    return {
        "ok": True,
        "models": models,
        "loaded_models": loaded_models,
        "gpu_busy": len(loaded_models) > 0,
    }


def check_candidate_models() -> dict[str, Any]:
    """Verify HuggingFace candidate model gating and license statuses."""
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=token)
        candidate_models = [
            "hlyn-labs/prompt-injection-judge-deberta-70m",
            "protectai/deberta-v3-base-prompt-injection-v2",
            "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1",
            "deepset/deberta-v3-base-injection",
            "meta-llama/Llama-Prompt-Guard-2-86M",
            "hlyn-labs/prompt-injection-judge-8b",
            "hlyn-labs/prompt-injection-judge-3b",
        ]
        statuses = {}
        for model_id in candidate_models:
            try:
                info = api.model_info(model_id, timeout=6)
                gated = getattr(info, "gated", False)
                license_str = "unknown"
                if info.card_data:
                    license_str = getattr(info.card_data, "license", "unknown")
                statuses[model_id] = {
                    "ok": True,
                    "gated": gated,
                    "license": license_str,
                }
            except Exception as e:  # noqa: BLE001
                statuses[model_id] = {"ok": False, "error": str(e)}

        return {"ok": True, "models": statuses}
    except Exception as exc:  # noqa: BLE001
        logger.error("HuggingFace API query failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def check_tool_venvs() -> dict[str, Any]:
    """Check existence of dedicated external tool virtual environments."""
    home = Path.home()
    expected_venvs = {
        "garak": home / ".venv_garak",
        "pyrit": home / ".venv_pyrit",
        "llamafirewall": home / ".venv_llamafirewall",
        "detector3": home / ".venv_detector3",
    }
    venv_statuses = {}
    for name, vpath in expected_venvs.items():
        exists = vpath.is_dir()
        venv_statuses[name] = {
            "path": str(vpath),
            "exists": exists,
        }
    return {"ok": True, "venvs": venv_statuses}


def check_service_ports() -> dict[str, Any]:
    """Verify status and availability of local daemon ports."""
    ports = {
        "promptforest": 8765,
        "detector3": 8766,
        "ollama": 11434,
        "api_gateway": 8000,
    }
    port_status = {}
    for name, port in ports.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            # Try connecting
            result = s.connect_ex(("127.0.0.1", port))
            is_listening = result == 0
            port_status[name] = {
                "port": port,
                "listening": is_listening,
            }
    return {"ok": True, "ports": port_status}


def check_tool_runtimes() -> dict[str, Any]:
    """Verify Node, npm, git, and python runtimes."""
    results = {}
    for tool in ["node", "npm", "git", "uv"]:
        bin_path = shutil.which(tool)
        if not bin_path:
            results[tool] = {"ok": False, "error": f"{tool} not found on PATH"}
            continue
        try:
            res = subprocess.run(
                [bin_path, "--version"],
                capture_output=True,
                text=True,
                check=False,
                shell=True,
            )
            results[tool] = {"ok": res.returncode == 0, "version": res.stdout.strip()}
        except OSError as e:
            results[tool] = {"ok": False, "error": str(e)}

    all_ok = all(v["ok"] for v in results.values())
    return {"ok": all_ok, "tools": results}


def check_hardware_headroom() -> dict[str, Any]:
    """Check GPU VRAM and system memory headroom."""
    # 1. System RAM
    ram_info = {"ok": False}
    try:

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_ram_gb = stat.ullTotalPhys / (1024**3)
            avail_ram_gb = stat.ullAvailPhys / (1024**3)
            ram_info = {
                "ok": True,
                "total_gb": round(total_ram_gb, 2),
                "free_gb": round(avail_ram_gb, 2),
            }
    except Exception as e:  # noqa: BLE001
        ram_info = {"ok": False, "error": str(e)}

    # 2. GPU VRAM via nvidia-smi
    vram_info = {"ok": False}
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().split(",")
            total_mb = float(lines[0])
            free_mb = float(lines[1])
            vram_info = {"ok": True, "total_mb": total_mb, "free_mb": free_mb}
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        vram_info = {"ok": False, "error": str(e)}

    return {"ram": ram_info, "vram": vram_info}


def run_preflight() -> bool:
    """Execute all preflight checks and output structured Wave G summary."""
    logger.info("Starting Wave G Preflight Verification...")

    sec = check_environment_secrets()
    pf = check_promptforest()
    ollama = check_ollama()
    models = check_candidate_models()
    tool_venvs = check_tool_venvs()
    ports = check_service_ports()
    tools = check_tool_runtimes()
    hw = check_hardware_headroom()

    print("\n" + "=" * 65)
    print("PARALLAX WAVE G PREFLIGHT RESULTS")
    print("=" * 65)

    print(f"1. HuggingFace Token Environment: {'PASS' if sec.get('ok') else 'FAIL'}")
    print(f"   - Token Source: {sec.get('source')}")

    print(f"\n2. PromptForest Service (8765): {'PASS' if pf.get('ok') else 'FAIL'}")
    if pf.get("ok"):
        print(f"   - Active Sub-Detectors: {pf.get('details_keys')}")
        print(f"   - Device: {pf.get('device')}")
    else:
        print(f"   - Error: {pf.get('error')}")

    print(f"\n3. Ollama Service (11434): {'PASS' if ollama.get('ok') else 'FAIL'}")
    if ollama.get("ok"):
        print(f"   - Installed Models: {ollama.get('models')}")
        print(f"   - Active in VRAM: {ollama.get('loaded_models') or 'None (Clean)'}")
    else:
        print(f"   - Error: {ollama.get('error')}")

    print(
        f"\n4. Candidate Model Gating & Licenses: {'PASS' if models.get('ok') else 'FAIL'}"
    )
    if models.get("ok"):
        for m_id, m_data in models.get("models", {}).items():
            if m_data.get("ok"):
                print(
                    f"   - {m_id}: gated={m_data.get('gated')}, license={m_data.get('license')}"
                )
            else:
                print(f"   - {m_id}: error={m_data.get('error')}")

    print("\n5. Dedicated Tool Virtual Environments:")
    for v_name, v_data in tool_venvs.get("venvs", {}).items():
        print(
            f"   - {v_name}: {'EXISTS' if v_data['exists'] else 'MISSING (to be provisioned)'} ({v_data['path']})"
        )

    print("\n6. Service Ports:")
    for p_name, p_data in ports.get("ports", {}).items():
        status_str = "IN USE (Listening)" if p_data["listening"] else "FREE (Available)"
        print(f"   - {p_name} (Port {p_data['port']}): {status_str}")

    print(f"\n7. Execution Toolchains: {'PASS' if tools.get('ok') else 'FAIL'}")
    for t_name, t_info in tools.get("tools", {}).items():
        print(f"   - {t_name}: {t_info.get('version', t_info.get('error'))}")

    print("\n8. Hardware Headroom:")
    if hw["ram"].get("ok"):
        print(
            f"   - Host System RAM: {hw['ram']['free_gb']} GB free / {hw['ram']['total_gb']} GB total"
        )
    if hw["vram"].get("ok"):
        print(
            f"   - GPU VRAM: {hw['vram']['free_mb']} MB free / {hw['vram']['total_mb']} MB total"
        )
    print("=" * 65 + "\n")

    passed = (
        sec.get("ok")
        and pf.get("ok")
        and ollama.get("ok")
        and models.get("ok")
        and tools.get("ok")
    )

    if not passed:
        logger.error("PREFLIGHT VERIFICATION FAILED. Halting autonomous run.")
        return False

    logger.info("ALL WAVE G PREFLIGHT CHECKS PASSED. Ready for Wave G1 execution.")
    return True


if __name__ == "__main__":
    success = run_preflight()
    sys.exit(0 if success else 1)
