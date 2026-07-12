"""CRYSTAL-1 B0 — the role-contract layer (the portability fix the DNA audit demanded).

Lessons this encodes (CRYSTAL_AGENT_BLUEPRINT.md §1, ironclad):
  * The legacy R6c risk gates were FEATURE-NAME-keyed with silent absent→0 defaults — on any universe missing
    those exact Dow-30 columns the gates read 0 < threshold and silently blocked all investment. FORBIDDEN.
  * Universe/group metadata was registered at runtime in the parent process → broke subprocess vectorization.
  * Breadth knobs were absolute counts tuned to Dow-30 scale (top_k_buy=8) and had to be hand-rescaled per
    universe.

The contract here:
  UniverseSpec  — a SERIALIZABLE (JSON round-trip) description of a universe: tickers, groups, the semantic
                  role→column binding, and breadth expressed as FRACTIONS of N (absolute counts derived at
                  build time).
  RoleAdapter   — binds semantic roles ("vix_shock", "regime_prob", …) to actual panel columns and FAILS
                  LOUD (raises RoleContractError) on any missing role or column — never a silent default.
  RoleGate      — the gate primitive re-expressed role-keyed: evaluates threshold conditions against ROLES
                  via the adapter; a gate whose role is unbound raises at CONSTRUCTION, not mid-episode.

Run: python -m src.crystal.universe   (selftest: loud failure, fraction breadth, JSON round-trip,
                                       binds to the real csi500 panel AND a polygon spec)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


class RoleContractError(RuntimeError):
    """A semantic role is unbound or its bound column is absent — always raised, never defaulted."""


# The registry of semantic roles CRYSTAL-1 layers may reference. Adding a role here is a design act
# (T0-config visible), not an incidental column name.
KNOWN_ROLES = {
    # market/regime roles (L1/L3 inputs)
    "vix_level": "volatility index level",
    "vix_shock": "short-horizon volatility shock",
    "regime_prob": "hidden-regime probability (e.g. HMM P(bear))",
    "market_trend": "broad-market trend indicator",
    "turbulence": "cross-sectional turbulence",
    "realized_vol": "trailing realized volatility (the legitimate risk driver)",
    # per-asset roles (L2 worker inputs)
    "asset_return_1d": "per-asset 1-day return",
    "asset_momentum": "per-asset medium-horizon momentum",
    "asset_volume_ratio": "per-asset volume ratio",
    # synthetic-polygon roles
    "observation_signal": "the env's raw partial observation of the hidden regime",
}


@dataclass
class UniverseSpec:
    """Serializable (JSON round-trip) universe: tickers, groups, role->column bindings, and fractional breadth."""
    name: str
    tickers: list
    role_bindings: dict                     # role -> column name in the panel (or env attribute)
    groups: dict = field(default_factory=dict)          # ticker -> group label
    breadth_fractions: dict = field(default_factory=lambda: {"top_buy_frac": 1 / 3, "top_sell_frac": 1 / 2})
    cash_symbol: str = "CASH"

    # ---------------- serialization (no runtime registration; child processes just load the file)
    def to_json(self, path: str | Path) -> Path:
        """Write the spec to `path` as JSON and return the path (subprocess-safe: no runtime registration)."""
        p = Path(path)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @staticmethod
    def from_json(path: str | Path) -> "UniverseSpec":
        """Load and return a UniverseSpec from a JSON file written by to_json."""
        return UniverseSpec(**json.loads(Path(path).read_text(encoding="utf-8")))

    # ---------------- derived breadth (fractions -> counts at build time)
    def breadth_counts(self) -> dict:
        """Derive absolute breadth counts (top_buy_k/top_sell_k) from the fractions and universe size N (half-up)."""
        n = len(self.tickers)
        # deterministic half-up (int(x+0.5)), NOT Python's banker's rounding — config derivation must never surprise
        return {k.replace("_frac", "_k"): max(1, int(v * n + 0.5)) for k, v in self.breadth_fractions.items()}

    def validate_roles(self) -> None:
        """Raise RoleContractError if any bound role is not in KNOWN_ROLES (adding a role must be deliberate)."""
        unknown = [r for r in self.role_bindings if r not in KNOWN_ROLES]
        if unknown:
            raise RoleContractError(f"unknown roles {unknown}; add to KNOWN_ROLES deliberately (a design act)")


class RoleAdapter:
    """Binds a UniverseSpec's roles to an actual panel (DataFrame columns). LOUD on any gap."""

    def __init__(self, spec: UniverseSpec, panel_columns: list):
        spec.validate_roles()
        self.spec = spec
        missing = {r: c for r, c in spec.role_bindings.items() if c not in panel_columns}
        if missing:
            raise RoleContractError(
                f"universe '{spec.name}': roles bound to ABSENT columns {missing} — refusing to run "
                f"(the legacy silent absent→0 default blocked investment on foreign universes)")
        self._map = dict(spec.role_bindings)

    def column(self, role: str) -> str:
        """Return the panel column bound to `role`, raising RoleContractError if unbound (never a silent default)."""
        if role not in self._map:
            raise RoleContractError(f"universe '{self.spec.name}': role '{role}' is UNBOUND — bind it or "
                                    f"disable the consumer gate explicitly (no silent default)")
        return self._map[role]

    def series(self, df, role: str):
        """Return the DataFrame column for `role` via the role->column binding."""
        return df[self.column(role)]


@dataclass
class RoleGate:
    """A role-keyed threshold gate: fires when role's value crosses threshold in `direction`.
    Constructing it against an adapter VERIFIES the binding immediately (fail at build, not mid-episode)."""
    name: str
    role: str
    threshold: float
    direction: str = "above"                 # 'above' | 'below'

    def bind(self, adapter: RoleAdapter) -> "BoundGate":
        """Resolve the gate's role to a column via `adapter` and return a BoundGate (raises if unbound at build)."""
        col = adapter.column(self.role)      # raises if unbound/absent
        return BoundGate(self.name, col, self.threshold, self.direction)


@dataclass
class BoundGate:
    """A RoleGate resolved to a concrete panel column, ready to evaluate against panel rows."""
    name: str
    column: str
    threshold: float
    direction: str

    def fires(self, row) -> bool:
        """Return whether the gate fires on `row` (value crosses threshold in `direction`); raises on NaN."""
        v = row[self.column]
        if v != v:                           # NaN: loud, not silent
            raise RoleContractError(f"gate '{self.name}': NaN in column '{self.column}'")
        return bool(v > self.threshold) if self.direction == "above" else bool(v < self.threshold)


def _selftest() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import pandas as pd
    out = []
    # 1. loud failure on absent column
    spec_bad = UniverseSpec("bad", ["A", "B"], {"vix_shock": "VIX_change_5d"})
    try:
        RoleAdapter(spec_bad, panel_columns=["close", "VIX"])
        out.append(("loud-failure", "FAIL — did not raise"))
    except RoleContractError:
        out.append(("loud-failure", "PASS (raises, no silent 0)"))
    # 2. unbound role at gate construction
    spec_ok = UniverseSpec("ok", ["A", "B", "C"], {"vix_level": "VIX"})
    ad = RoleAdapter(spec_ok, panel_columns=["VIX", "close"])
    try:
        RoleGate("crash_arm", "vix_shock", 2.0).bind(ad)
        out.append(("unbound-gate", "FAIL — did not raise"))
    except RoleContractError:
        out.append(("unbound-gate", "PASS (gate fails at BUILD)"))
    # 3. fraction breadth derives counts from N
    counts = UniverseSpec("n29", [f"T{i}" for i in range(29)], {}).breadth_counts()
    counts500 = UniverseSpec("n344", [f"T{i}" for i in range(344)], {}).breadth_counts()
    ok = counts == {"top_buy_k": 10, "top_sell_k": 15} and counts500 == {"top_buy_k": 115, "top_sell_k": 172}
    out.append(("fraction-breadth", f"{'PASS' if ok else 'FAIL'} 29→{counts}, 344→{counts500}"))
    # 4. JSON round-trip (serializable universe: subprocess-safe by construction)
    p = Path(__file__).parent / "_selftest_universe.json"
    spec_ok.to_json(p)
    rt = UniverseSpec.from_json(p)
    out.append(("json-roundtrip", "PASS" if rt == spec_ok else "FAIL"))
    p.unlink(missing_ok=True)
    # 5. binds to the REAL csi500 panel (same adapter code, different universe)
    csi = Path(__file__).parents[2] / "data/adapters/_csi500_wide/csi300_model_ready.csv"
    if csi.exists():
        cols = list(pd.read_csv(csi, nrows=1).columns)
        spec_csi = UniverseSpec("csi500_wide", ["placeholder"],
                                {"vix_level": "VIX", "regime_prob": "Regime_1_Prob",
                                 "market_trend": "SP500_Trend", "turbulence": "turbulence"})
        RoleAdapter(spec_csi, cols)          # raises if the contract is wrong
        gate = RoleGate("derisk_on_regime", "regime_prob", 0.5, "above").bind(RoleAdapter(spec_csi, cols))
        out.append(("real-panel-bind", f"PASS (csi500: 4 roles bound; gate→{gate.column})"))
    else:
        out.append(("real-panel-bind", "SKIP (panel absent)"))
    # 6. polygon spec (the synthetic universe speaks the same contract)
    spec_poly = UniverseSpec("series_g_polygon", ["VENUE_0"], {"observation_signal": "y"})
    RoleAdapter(spec_poly, ["y", "t", "inv"])
    out.append(("polygon-bind", "PASS"))
    print("=== crystal.universe selftest ===")
    for k, v in out:
        print(f"  {k:18s}: {v}")
    ok_all = all("PASS" in v or "SKIP" in v for _, v in out)
    print(f"VERDICT: {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    _selftest()
