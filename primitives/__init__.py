"""Pure-numpy primitives extracted from mentis_ai.

Each module is a line-level copy of a specific source in mentis_ai, with
framework coupling (dispatch tables, mentis_* imports) removed. Behaviour
is bit-exact vs the source.
"""
from .similarity import kl_divergence, jsd
from .lexical import lexical_jsd, lexical_predictive_kl
from .policy import gated_kl, hard_gated_kl, policy_select_projection
from .peaks import quantile_peak_select
from .segment import regime_segment

__all__ = [
    "kl_divergence", "jsd",
    "lexical_jsd", "lexical_predictive_kl",
    "gated_kl", "hard_gated_kl", "policy_select_projection",
    "quantile_peak_select",
    "regime_segment",
]
