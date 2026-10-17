"""Crusoe Managed Inference: endpoint and pinned per-token prices.

Prices are USD per 1M tokens from the Serverless Inference table at
https://crusoe.ai/cloud/pricing, and exact model IDs from the authenticated
https://api.inference.crusoecloud.com/v1/models listing, both read on 2026-10-17.
Only models with a published price are allowed. Spending limits are only as
accurate as this table, so recheck both pages before a live run.
"""

from decimal import Decimal
import os
from urllib.parse import urlsplit

BASE_URL = "https://api.inference.crusoecloud.com/v1"
API_KEY_ENV = "CRUSOE_API_KEY"
# Tests point the proxy at a local fake; only loopback http is accepted.
BASE_URL_OVERRIDE_ENV = "DISCOVERY_CRUSOE_BASE_URL"
PRICE_SOURCE = "https://crusoe.ai/cloud/pricing"
MODEL_SOURCE = BASE_URL + "/models"
PRICES_RETRIEVED = "2026-10-17"
# Conservative ceiling across the pinned catalog (all advertised contexts <= 1M).
# Reserve the full ceiling, rather than estimating the provider's chat template.
MAX_INPUT_TOKENS = 1048576

# Exact model IDs verified against authenticated /v1/models on 2026-10-17.
# Model ID: (input, output, cached input), USD per 1M tokens.
_PRICES = {
    "deepseek-ai/Deepseek-V4-Flash": ("0.14", "0.28", "0.03"),
    "deepseek-ai/DeepSeek-V4.1-Flash": ("0.29", "1.20", "0.007"),
    "deepseek-ai/DeepSeek-V4-Pro": ("1.74", "3.48", "0.15"),
    "google/gemma-4-31b-it": ("0.14", "0.40", "0.14"),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": ("0.05", "0.20", "0.03"),
    "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B": ("0.30", "1.83", "0.30"),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B": ("0.30", "2.40", "0.15"),
    "nvidia/Nemotron-3.5-Lightning-30B-A3B": ("0.05", "0.20", "0.03"),
    "openai/gpt-oss-120b": ("0.05", "0.25", "0.05"),
    "zai-org/GLM-5.3": ("1.40", "4.40", "0.26"),
    "zai-org/GLM-5.3-Flash": ("0.15", "0.50", "0.03"),
}
MODELS = tuple(sorted(_PRICES))
_ALIASES = {
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek-ai/Deepseek-V4-Flash",
    "nvidia/Nemotron-3-Nano-30B-A3B": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3-Super-120B-A12B": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    "nvidia/nemotron-3.5-lightning-30b-a3b": "nvidia/Nemotron-3.5-Lightning-30B-A3B",
    "zai/GLM-5.3": "zai-org/GLM-5.3",
    "zai/GLM-5.3-Flash": "zai-org/GLM-5.3-Flash",
}


def canonical_model(model):
    """Keep earlier documented commands usable, but send exact API identifiers."""
    if not isinstance(model, str):
        raise ValueError("The model identifier must be a string.")
    return _ALIASES.get(model, model)


def prices(model):
    """Decimal USD per 1M tokens for a priced model; KeyError otherwise."""
    input_price, output_price, cached_price = _PRICES[canonical_model(model)]
    return {"input": Decimal(input_price), "output": Decimal(output_price),
            "cached_input": Decimal(cached_price)}


def price_record(model):
    record = {key: str(value) for key, value in prices(model).items()}
    record.update(unit="USD per 1M tokens", source=PRICE_SOURCE, retrieved=PRICES_RETRIEVED,
                  input_token_ceiling=MAX_INPUT_TOKENS, model_source=MODEL_SOURCE)
    return record


def upstream_url():
    """The Crusoe base URL, or a loopback test override."""
    override = os.environ.get(BASE_URL_OVERRIDE_ENV, "").strip()
    if not override:
        return BASE_URL
    parts = urlsplit(override)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1":
        raise ValueError(f"{BASE_URL_OVERRIDE_ENV} may only name http://127.0.0.1 test servers.")
    return override
