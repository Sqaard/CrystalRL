"""Static sector map for the current 29-ticker Dow-style universe."""

from __future__ import annotations


DOW30_STATIC_SECTOR_MAP: dict[str, str] = {
    "AAPL": "technology",
    "AMGN": "healthcare",
    "AMZN": "consumer_discretionary",
    "AXP": "financials",
    "BA": "industrials",
    "CAT": "industrials",
    "CRM": "technology",
    "CSCO": "technology",
    "CVX": "energy",
    "DIS": "communication_services",
    "GS": "financials",
    "HD": "consumer_discretionary",
    "HON": "industrials",
    "IBM": "technology",
    "INTC": "technology",
    "JNJ": "healthcare",
    "JPM": "financials",
    "KO": "consumer_staples",
    "MCD": "consumer_discretionary",
    "MMM": "industrials",
    "MRK": "healthcare",
    "MSFT": "technology",
    "NKE": "consumer_discretionary",
    "PG": "consumer_staples",
    "TRV": "financials",
    "UNH": "healthcare",
    "V": "financials",
    "VZ": "communication_services",
    "WMT": "consumer_staples",
}


# ---------------------------------------------------------------------------
# Runtime-registered sector maps (for arbitrary universes, e.g. Qlib csi300).
#
# The env validates EVERY panel ticker against the configured sector map before any
# custom-group config is consulted, so to drive the env on a universe whose tickers are
# not in DOW30_STATIC (e.g. Qlib's SH600000/SZ000001 csi300 names) you register a matching
# map here first. The generic case is a single "all-U" group: every ticker maps to one
# sector "U", which the env then treats as a single stock group. See DATA_ADAPTERS.md.
# ---------------------------------------------------------------------------
_REGISTERED_SECTOR_MAPS: dict[str, dict[str, str]] = {}


def register_single_group_map(name: str, tickers: list[str], group: str = "U") -> dict[str, str]:
    """Register (and return) a sector map that assigns every ticker in ``tickers`` to one group.

    This is the generic, universe-agnostic map DATA_ADAPTERS.md describes for an arbitrary
    Qlib universe: all names land in a single sector ``group`` (default "U"), so the env's
    group layer is a single stock group with no sector partition. Re-registering the same
    ``name`` overwrites it.
    """
    mapping = {str(t): group for t in tickers}
    _REGISTERED_SECTOR_MAPS[name] = mapping
    return dict(mapping)


def get_sector_map(name: str) -> dict[str, str]:
    """Return a ticker -> sector map by name (static ``dow30_static`` or a registered map)."""
    if name == "dow30_static":
        return dict(DOW30_STATIC_SECTOR_MAP)
    if name in _REGISTERED_SECTOR_MAPS:
        return dict(_REGISTERED_SECTOR_MAPS[name])
    raise ValueError(
        f"Unknown sector map: {name}. Known: 'dow30_static' plus registered "
        f"{sorted(_REGISTERED_SECTOR_MAPS)}. Register one with register_single_group_map(name, tickers)."
    )
