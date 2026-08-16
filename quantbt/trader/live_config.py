"""Configuration loader — reads .env file for live trading settings."""

import os
import json
from pathlib import Path

EXPECTED_KEYS = [
    "THS_EXE_PATH", "THS_ACCOUNT", "THS_PASSWORD",
    "STRATEGY_NAME", "STRATEGY_PARAMS", "STOCK_POOL",
    "INITIAL_CAPITAL", "MAX_POSITIONS", "REBALANCE_FREQ",
]


def load_config(env_path: str | None = None) -> dict:
    """Load live trading configuration.

    Priority (highest wins):
    1. OS environment variables
    2. .env file
    """
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"

    # Read .env file
    env_vars: dict[str, str] = {}
    env_file = Path(env_path)
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]  # strip one pair of wrapping quotes
            elif " #" in value:
                value = value.split(" #", 1)[0].rstrip()  # strip inline comment
            env_vars[key] = value

    # Environment variables override .env
    raw = {}
    for key in EXPECTED_KEYS:
        raw[key] = os.environ.get(key) or env_vars.get(key) or ""

    # Parse typed config
    try:
        strategy_params = json.loads(raw.get("STRATEGY_PARAMS") or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"STRATEGY_PARAMS in the .env file is not valid JSON: {e}"
        ) from e

    parsed = {
        "ths_exe_path": raw["THS_EXE_PATH"],
        "ths_account": raw["THS_ACCOUNT"],
        "ths_password": raw["THS_PASSWORD"],
        "strategy_name": raw.get("STRATEGY_NAME") or "momentum",
        "strategy_params": strategy_params,
        "stock_pool": [
            s.strip()
            for s in (raw.get("STOCK_POOL") or "").split(",")
            if s.strip()
        ],
        "initial_capital": float(raw.get("INITIAL_CAPITAL") or "1000000"),
        "max_positions": int(raw.get("MAX_POSITIONS") or "5"),
        "rebalance_freq": raw.get("REBALANCE_FREQ") or "daily",
    }

    if not parsed["stock_pool"]:
        raise ValueError(
            "STOCK_POOL is empty. Set at least one stock code in .env"
        )

    return parsed