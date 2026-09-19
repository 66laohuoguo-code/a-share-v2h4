"""Component weights, loaded from a local file that is never committed.

Why this module exists
----------------------
The production weighting scheme used to be hard-coded in the strategy engines,
which published the numbers along with the source. The weights now live in

    config/component_weights.local.json      (git-ignored)

and this module is the only place that knows how to read them.

On a fresh clone that file does not exist, so `load()` falls back to the
PLACEHOLDER values below. Those placeholders are **equal weights** on purpose:
they let the pipeline run end-to-end for demonstration, and they are obviously
not a tuned production scheme, so a demo run cannot be mistaken for a real one.

Point the loader somewhere else with the environment variable
`ASHARE_COMPONENT_WEIGHTS`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PATH = os.environ.get(
    "ASHARE_COMPONENT_WEIGHTS", "config/component_weights.local.json"
)


def _equal(*names: str) -> Dict[str, float]:
    return {name: round(1.0 / len(names), 4) for name in names}


# Demonstration-only weights: equal-weighted, deliberately not tuned.
PLACEHOLDERS: Dict[str, Dict[str, float]] = {
    "static": _equal(
        "low_beta_score", "low_volatility_score", "low_turnover_score",
        "reversal_score", "lower_drawdown_score", "industry_trend_score",
    ),
    "small_account_v3": _equal(
        "low_beta_score", "low_volatility_score", "low_turnover_score",
        "reversal_score", "lower_drawdown_score", "industry_trend_score",
        "earnings_yield_score", "residual_momentum_score",
    ),
    "v31_alpha": _equal("earnings_yield_score", "quality_score_v31"),
    "v31_default": _equal(
        "low_beta_score", "low_volatility_score", "low_turnover_score",
        "lower_drawdown_score", "industry_trend_score", "earnings_yield_score",
        "quality_score_v31", "growth_score_v31", "residual_momentum_score",
    ),
}


def load(section: str, path: Optional[Path] = None) -> Dict[str, float]:
    """Return one weight set.

    Reads `path` (default `DEFAULT_PATH`) and returns the requested section.
    When the file is absent the equal-weighted placeholder for that section is
    returned, so a fresh clone still runs. A malformed file raises rather than
    silently falling back, because quietly using demo weights in production
    would be far worse than failing loudly.
    """
    resolved = Path(path) if path is not None else Path(DEFAULT_PATH)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        if section not in PLACEHOLDERS:
            raise SystemExit(
                "no weights file at %s and no placeholder for %r" % (resolved, section)
            )
        return dict(PLACEHOLDERS[section])
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit("cannot read component weights from %s: %s" % (resolved, error))

    weights = payload.get(section)
    if not isinstance(weights, dict) or not weights:
        raise SystemExit("%s: missing or empty section %r" % (resolved, section))
    return {str(key): float(value) for key, value in weights.items()}


def load_all(path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """Return every section; used by report and validation tooling."""
    return {section: load(section, path) for section in PLACEHOLDERS}


def is_placeholder(section: str, path: Optional[Path] = None) -> bool:
    """True when `section` is currently running on demo values.

    Callers that must not emit production output from demo weights can gate on
    this and warn or abort.
    """
    return load(section, path) == dict(PLACEHOLDERS.get(section, {}))
