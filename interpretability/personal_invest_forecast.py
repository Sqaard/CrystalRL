"""W3/P-5 — the INDEPENDENT forecast/scenario engine (v1), separate from any policy/PPO layer.

North-star role: expected future return and its probability are estimated by THIS calibrated
wealth-distribution engine, never by a PPO critic. Work-order W3 requirements implemented at v1:

  * ENSEMBLE of scenario members, each producing h-year net-nominal annualized-return samples:
      m1 unconditional PIT stationary bootstrap (Politis-Romano, E[block]=63d) — the P-2 kernel;
      m2 regime-conditional bootstrap (two vol states, empirical block-level persistence);
      m3 building-block CMA recentering (P-5): equity central = implied dividend yield (from the
         TR-vs-price pair) + sourced real-growth anchor + expected inflation; bond/gold/cash anchors
         from current yields — m1 paths recentered to the CMA central;
    plus a PARAMETER-UNCERTAINTY layer: every sample carries a mean-shift draw ~ N(0, se(mu_hat))
    (block-bootstrap SE; the shift does NOT shrink with horizon — Pastor-Stambaugh's point).
  * JOINT extras: annual-grain inflation paths (World Bank FP.CPI.TOTL.ZG, cached+hashed) give net REAL
    outcomes; dated fee layers (CN_ACCESS_V1 / US ETF fees); the CN cash leg is the net-of-tax CN_CASH_V1
    path, the US cash leg is ^IRX.
  * POINT-IN-TIME discipline: every member sees only data <= as_of (annual inflation only <= as_of.year-1).
  * FORECAST LEDGER: every forward forecast is written BEFORE realization (LOCKED_PENDING_REALIZATION).
  * STRESSES: named historical windows replayed + a hypothetical joint stress (-30% equity, +5% inflation).
  * Honest v1 limitations (ledgered): inflation resampled independently of returns at annual grain;
    growth anchors are sourced constants (approx-flagged); DY-reversion is the active valuation proxy;
    latest-vintage Shiller CAPE was tested and rejected, not promoted as PIT evidence;
    CN dividend yield proxied by the CSI300 TR/price pair; daily constant-mix books without rebalance cost.

Calibration/exit gates live in exp_w3_calibration.py; until they PASS, every probability from this module
is labeled UNCALIBRATED_RESEARCH_ESTIMATES. Effective sample sizes are reported, not path counts.
"""
from __future__ import annotations
import hashlib, json, sys, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

REG = ROOT / "data" / "_personal_invest_registry"
APPROVED = REG / "approved_daily_returns.csv"
AUDIT = REG / "w2_data_gate_audit.json"
INFL_CSV = REG / "inflation_annual_worldbank.csv"
CAPE_CSV = REG / "shiller_cape_monthly.csv"
CAPE_XLS = REG / "_shiller_ie_data.xls"
CAPE_LAG_MONTHS = 3          # diagnostic earnings/publication lag; latest-vintage history is not PIT-clean
LEDGER_DIR = REG / "forecast_ledger"
US_CACHE = ROOT / "data" / "_dow_extended" / "full2000"

ENGINE_VERSION = "W3_FORECAST_V1_4"
# v1.4 (W3.4, 2026-07-11) — ONE structural fix from the HL-PROFILES-B backtest diagnosis, declared pre-run:
# the CASH leg of every sampled path now resamples rate CHANGES cumulated from the CURRENT rate (floored
# at 0), instead of resampling historical rate LEVELS — rf is a persistent random walk, and level
# resampling made ZIRP-era cash promises structurally overconfident (48.5% exceedance on the conservative
# profile). Block indices are SHARED between the risky part and the rate changes, preserving co-movement.
# The risky part, members, weights, conformal, CMA anchors: all unchanged from v1.3.
# v1.3 (W3.3, 2026-07-11) — the real Shiller CAPE was tested as the US valuation anchor and FALSIFIED on
# the reproducible diagnostic (G4 MAE 0.0677 vs DY-reversion 0.0451 vs sample-mean 0.0444): expanding-mean CAPE
# reversion under-forecasts modern US equity — the well-known "CAPE has been expensive for 30 years"
# failure; the implied-DY expanding mean adapts faster. VERDICT: the DY-reversion term is RETAINED for
# both universes; the CAPE series stays in the registry (load_cape) as a tested-and-rejected alternative
# and a diagnostic. Scenario equations stay v1.2-equivalent; v1.3 additionally makes the forward-ledger
# path use the same prequential weights and widen-only conformal rule as the calibration harness.
# v1.2 (W3.2, 2026-07-11) — the second (and per pre-commitment, LAST constant-free) structural iteration:
#   (d) VALUATION-REVERSION term in the equity anchor (Bogle's speculative return with the implied dividend
#       yield as the valuation metric): drag = (DY_now / DY_expanding_mean)^(1/10) - 1 — cheap markets
#       (high DY) get a positive term, expensive ones negative; symmetric across universes, PIT-clean,
#       from the OFFICIAL TR-vs-price pair (Shiller CAPE fetch works but its legacy format + earnings
#       lag make it a W3.3 option, ledgered);
#   (e) PREQUENTIAL MEMBER WEIGHTS: the mixture learns its weights online — softmax over each member's
#       trailing resolved pinball (archive/harness-supplied); equal weights until >=10 resolved scores;
#   (f) the harness's PRIMARY proper score is now DOWNSIDE-weighted pinball (taus 05/10/25/50 — the
#       product decides on risk quantiles), declared BEFORE the run; full pinball stays reported.
# v1.1 (W3.1, 2026-07-11) — three NAMED fixes from the v1 gate failures, structural not tuned:
#   (a) m4 stress member (US only): a crisis-shock tail so 1y bands carry a crash even when the trailing
#       history lacks one (v1's 2008-cohort undercoverage: prequential conformal cannot know 2008 in 2007);
#   (b) residual-variance parameter layer: member-central disagreement and the se(mu) draw DOUBLE-COUNTED
#       mean uncertainty (CN overcoverage) -> eps ~ N(0, sqrt(max(se^2 - var(member centrals), 0)));
#   (c) adaptive-conformal coverage wrapper (widen-only, prequential): factor from ARCHIVED resolved
#       nonconformity scores; sanctioned by the work order as a wrapper, never a personal CDF.
EBLOCK = 63
STRESS_WEIGHT = {"US": 0.10, "CN": 0.0}   # CN failed by OVERcoverage; widening it would be anti-diagnosis
CALIB_ARCHIVE = HERE / "exp_w3_calibration_report.json"
GOALS = {"US": {"inflation": 0.03, "deposits": 0.045, "6%": 0.06, "10%": 0.10},
         "CN": {"inflation": 0.022, "deposits": 0.025, "6%": 0.06, "10%": 0.10}}
# sourced constants, approx-flagged (v1): long-run REAL earnings-growth anchors
REAL_GROWTH = {"US": 0.018, "CN": 0.030}          # US: ~Shiller-era real EPS growth; CN: conservative haircut of GDP
# v1.1: US equity cash returned to holders is DY + NET BUYBACKS (~1.5%/yr net of issuance post-2000,
# approx-flagged) — plain DY structurally underestimated US equity and lost the G4 backcast in v1.
# CN stays 0: A-share net issuance is NEGATIVE-to-zero payout wedge, and CN's CMA already passed G4 —
# touching a passing gate would be tuning, not a named fix.
NET_BUYBACK = {"US": 0.015, "CN": 0.0}
US_FEES = {"SPY": 0.0009, "SLEEVE": 0.0015}       # approx expense ratios, flagged
CN_FEE_CUT = pd.Timestamp("2024-11-22"); CN_FEE_BEFORE, CN_FEE_AFTER = 0.0060, 0.0020
STRESS_WINDOWS = {"GFC_2008": ("2008-01-01", "2008-12-31"), "CN_2015_crash": ("2015-06-15", "2016-02-01"),
                  "COVID_2020": ("2020-02-15", "2020-04-15"), "inflation_2022": ("2022-01-01", "2022-12-31")}


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ data ------------------------
def load_inflation() -> pd.DataFrame:
    """Annual CPI inflation (%, YoY) for US/CN from the World Bank; cached + hashed in the registry."""
    if INFL_CSV.exists():
        return pd.read_csv(INFL_CSV)
    url = "https://api.worldbank.org/v2/country/CHN;USA/indicator/FP.CPI.TOTL.ZG?format=json&per_page=400"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (W3 engine)"})
    j = json.loads(urllib.request.urlopen(req, timeout=60).read())
    rows = [{"country": ("CN" if r["country"]["id"] == "CN" else "US"), "year": int(r["date"]),
             "cpi_yoy_pct": r["value"]} for r in j[1] if r["value"] is not None]
    d = pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)
    d["source"] = "World Bank FP.CPI.TOTL.ZG"; d["fetched_at_utc"] = _utc_now()
    d.to_csv(INFL_CSV, index=False, lineterminator="\n")
    return d


def load_cape() -> pd.Series:
    """Latest-vintage monthly CAPE under a +3m diagnostic lag convention; not a PIT-vintage archive."""
    if not CAPE_CSV.exists():
        if not CAPE_XLS.exists():
            req = urllib.request.Request("http://www.econ.yale.edu/~shiller/data/ie_data.xls",
                                         headers={"User-Agent": "Mozilla/5.0 (W3.3)"})
            CAPE_XLS.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        raw = pd.read_excel(CAPE_XLS, sheet_name="Data", header=None, skiprows=8)
        dates, capes = [], []
        for _, row in raw.iterrows():
            try:
                x = float(row[0]); cape = float(row[12])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x) and np.isfinite(cape)):
                continue
            year = int(x); month = int(round((x - year) * 100))
            if 1 <= month <= 12:
                dates.append(pd.Timestamp(year=year, month=month, day=1)); capes.append(cape)
        d = pd.DataFrame({"month": dates, "cape": capes}).drop_duplicates("month").sort_values("month")
        d["available_at"] = d["month"] + pd.DateOffset(months=CAPE_LAG_MONTHS)
        d["source"] = "Shiller ie_data.xls (econ.yale.edu)"; d["xls_sha256"] = _sha(CAPE_XLS)
        d.to_csv(CAPE_CSV, index=False, lineterminator="\n")
    d = pd.read_csv(CAPE_CSV, parse_dates=["month", "available_at"])
    return d.set_index("available_at")["cape"].sort_index()


def _yahoo_cached(sym: str) -> pd.Series:
    from interpretability.exp_p2_longhistory import fetch          # cached one-response-per-series fetch
    return fetch(sym).set_index("date")["adjclose" if sym[0] != "^" else "close"].sort_index()


def _official_price_index(code: str) -> pd.Series:
    """Official CSI PRICE index (e.g. 000905) from the same endpoint/provenance as H00905; cached+identity-
    checked. Needed for the TR-vs-price dividend-yield pair (the Yahoo mirror only starts ~2021)."""
    f = REG / "official_indices" / f"P{code}.csv"
    if not f.exists():
        params = urllib.parse.urlencode({"indexCode": code, "startDate": "20041231",
                                         "endDate": pd.Timestamp.now().strftime("%Y%m%d")})
        req = urllib.request.Request(
            f"https://www.csindex.com.cn/csindex-home/perf/index-perf?{params}",
            headers={"Accept": "application/json", "Referer": "https://www.csindex.com.cn/",
                     "User-Agent": "Mozilla/5.0 (W3 engine)", "X-Requested-With": "XMLHttpRequest"})
        j = json.loads(urllib.request.urlopen(req, timeout=120).read())
        rows = j.get("data") or []
        if str(j.get("code")) != "200" or not rows:
            raise RuntimeError(f"official price index {code} unavailable: {j.get('code')}")
        if {str(r["indexCode"]) for r in rows} != {code}:
            raise RuntimeError(f"identity mismatch fetching price index {code}")
        d = pd.DataFrame({"date": pd.to_datetime([r["tradeDate"] for r in rows], format="mixed"),
                          "index_code": code,
                          "index_name_en": [r["indexNameEnAll"] for r in rows],
                          "close": pd.to_numeric([r["close"] for r in rows], errors="coerce")})
        d = d.dropna().drop_duplicates("date").sort_values("date")
        d.to_csv(f, index=False, lineterminator="\n")
    return pd.read_csv(f, parse_dates=["date"]).set_index("date")["close"].sort_index()


def _implied_div_yield(tr: pd.Series, px: pd.Series) -> pd.Series:
    """Trailing-1y dividend yield implied by the total-return-vs-price pair."""
    both = pd.DataFrame({"tr": tr, "px": px}).dropna()
    rel = (both["tr"] / both["tr"].iloc[0]) / (both["px"] / both["px"].iloc[0])
    return (rel / rel.shift(252) - 1.0).clip(lower=0.0)


def load_universe(universe: str) -> dict:
    """Daily component returns + rf + dividend-yield series + inflation, all indexed by date."""
    infl = load_inflation()
    if universe == "CN":
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        if audit.get("gate_verdict") != "PASS" or _sha(APPROVED) != audit["approved_daily_returns"]["sha256"]:
            raise SystemExit("REFUSED: the W2 gate has not signed the CN panel (or the panel was tampered)")
        p = pd.read_csv(APPROVED, parse_dates=["date"]).set_index("date")
        comp = {"CSI500": p["csi500_total_return"], "CSI300": p["csi300_total_return"]}
        # W2-verified issuer-NAV sleeves (ISSUER_NAV_V1): the CN diversifiers for profile books.
        # cum_nav carries distributions; history starts 2013 — books using them are younger, handled
        # by the engine's 5y-history guard. Sleeve-specific fees live in CN_ACCESS_V1 (the index-ETF
        # fee layer is applied uniformly here — an approximation, noted).
        for sleeve, code in (("BOND_CN", "511010"), ("GOLD_CN", "518880"), ("QDII_CN", "513100")):
            f_nav = REG / "issuer_nav" / f"{code}.csv"
            if f_nav.exists():
                nav = pd.read_csv(f_nav, parse_dates=["date"]).set_index("date")["cum_nav"]
                comp[sleeve] = nav.sort_index().reindex(p.index).ffill().pct_change()
        rf = p["rf_cn"]
        h905 = pd.read_csv(REG / "official_indices" / "H00905.csv", parse_dates=["date"]).set_index("date")["close"]
        px905 = _official_price_index("000905")                          # same official provenance
        dy = _implied_div_yield(h905, px905).reindex(p.index).ffill()
        fee = pd.Series(np.where(p.index >= CN_FEE_CUT, CN_FEE_AFTER, CN_FEE_BEFORE), index=p.index)
    elif universe == "US":
        spy = _yahoo_cached("SPY"); spypx = _yahoo_cached("SPY").rename("a")  # adjclose
        from interpretability.exp_p2_longhistory import fetch
        spy_close = fetch("SPY").set_index("date")["close"].sort_index()
        comp = {"SPY": spy.pct_change(), "EFA": _yahoo_cached("EFA").pct_change(),
                "IEF": _yahoo_cached("IEF").pct_change(), "TLT": _yahoo_cached("TLT").pct_change(),
                "TIP": _yahoo_cached("TIP").pct_change(), "GLD": _yahoo_cached("GLD").pct_change()}
        idx = comp["SPY"].dropna().index
        rf = (_yahoo_cached("^IRX").reindex(idx).ffill() / 100 / 252).fillna(0.0)
        dy = _implied_div_yield(spy, spy_close).reindex(idx).ffill()
        fee = pd.Series(US_FEES["SPY"], index=idx)
    else:
        raise ValueError(universe)
    tnx = _yahoo_cached("^TNX") if universe == "US" else None
    return {"universe": universe, "components": comp, "rf": rf, "div_yield": dy, "fee": fee,
            "tnx": tnx,
            "inflation": infl[infl["country"] == universe].set_index("year")["cpi_yoy_pct"] / 100}


# ------------------------------------------------------------------ books -----------------------
def book_daily_net(uni: dict, book: dict, as_of: pd.Timestamp) -> np.ndarray:
    """Daily net returns of a constant-mix book up to as_of (fees subtracted daily; v1: no rebal cost)."""
    parts = []
    for key, w in book["weights"].items():
        s = uni["rf"] if key == "CASH" else uni["components"][key]
        parts.append(w * s)
    r = sum(parts).dropna()
    r = r[r.index <= as_of]
    eq_w = sum(w for k, w in book["weights"].items() if k != "CASH")
    fee = uni["fee"].reindex(r.index).ffill().to_numpy() / 252.0 * eq_w
    return (r.to_numpy() - fee)


# ------------------------------------------------------------------ members ---------------------
class RateAwareSeries:
    """v1.4: a book's daily history decomposed into the risky part and the cash leg, so sampled paths
    resample rate CHANGES cumulated from the CURRENT rate (persistent process) while the risky part is
    resampled by LEVELS as before. Books without a CASH weight behave EXACTLY as plain arrays."""
    def __init__(self, total, x_ex, drf, rf0, w_cash):
        self.total = np.asarray(total, dtype=float)
        self.x_ex = np.asarray(x_ex, dtype=float)
        self.drf = np.asarray(drf, dtype=float)
        self.rf0 = float(rf0)
        self.w_cash = float(w_cash)

    def __len__(self):
        return len(self.total)


RF_DAILY_CAP = 0.15 / 252


def _as_total(x):
    return x.total if isinstance(x, RateAwareSeries) else np.asarray(x, dtype=float)


def _stat_boot(x, n_days: int, n_paths: int, rng, pool_mask: np.ndarray | None = None):
    base = x.x_ex if isinstance(x, RateAwareSeries) else np.asarray(x, dtype=float)
    n = len(base)
    starts_pool = np.flatnonzero(pool_mask) if pool_mask is not None else np.arange(n)
    out = np.empty((n_paths, n_days))
    idx = np.empty(n_days, dtype=int)
    for p in range(n_paths):
        t = 0
        while t < n_days:
            start = int(rng.choice(starts_pool))
            L = min(int(rng.geometric(1.0 / EBLOCK)), n_days - t)
            idx[t:t + L] = (start + np.arange(L)) % n
            t += L
        out[p] = base[idx]
        if isinstance(x, RateAwareSeries) and x.w_cash > 0:
            # SHARED indices: the cash leg cumulates the SAME blocks' rate CHANGES from the current rate
            rf_path = np.clip(x.rf0 + np.cumsum(x.drf[idx]), 0.0, RF_DAILY_CAP)
            out[p] += x.w_cash * rf_path
    return out


def _ann(paths: np.ndarray, h: float) -> np.ndarray:
    return np.prod(1 + paths, axis=1) ** (1.0 / h) - 1


def m1_unconditional(x, h, n, rng):
    return _ann(_stat_boot(x, int(252 * h), n, rng), h)


def m2_regime(x, h, n, rng):
    """Two vol states (trailing 63d vol vs expanding median); blocks start in state-matched days;
    the state flips between blocks per its empirical block-level persistence."""
    v = pd.Series(_as_total(x)).rolling(EBLOCK).std().to_numpy()
    med = pd.Series(v).expanding(min_periods=252).median().to_numpy()
    state = (v > med).astype(int)
    state[np.isnan(v) | np.isnan(med)] = 0
    flips = np.abs(np.diff(state[252:])).mean() * EBLOCK if len(state) > 504 else 0.5
    p_flip = min(max(flips, 0.05), 0.95)
    cur = int(state[-1])
    n_days = int(252 * h)
    out = np.empty((n_paths := n, n_days))
    pools = {s: (state == s) for s in (0, 1)}
    base = x.x_ex if isinstance(x, RateAwareSeries) else np.asarray(x, dtype=float)
    idxbuf = np.empty(n_days, dtype=int)
    for p in range(n_paths):
        t, s = 0, cur
        while t < n_days:
            pool = pools[s] if pools[s].sum() > 300 else None
            start = int(rng.choice(np.flatnonzero(pool))) if pool is not None else int(rng.integers(0, len(x)))
            L = min(int(rng.geometric(1.0 / EBLOCK)), n_days - t)
            idxbuf[t:t + L] = (start + np.arange(L)) % len(x)
            t += L
            if rng.random() < p_flip:
                s = 1 - s
        out[p] = base[idxbuf]
        if isinstance(x, RateAwareSeries) and x.w_cash > 0:
            rf_path = np.clip(x.rf0 + np.cumsum(x.drf[idxbuf]), 0.0, RF_DAILY_CAP)
            out[p] += x.w_cash * rf_path
    return _ann(out, h)


def cma_central(uni: dict, book: dict, as_of: pd.Timestamp) -> float:
    """Building-block expected NET nominal return (P-5): equity = DY + real growth + expected inflation;
    bonds = current 10y yield; TIP = 10y yield (approx); GLD = expected inflation; cash = current rf."""
    infl = uni["inflation"]
    exp_infl = float(infl[infl.index <= as_of.year - 1].tail(5).mean())
    dy = float(uni["div_yield"][uni["div_yield"].index <= as_of].iloc[-1])
    rf_now = float(uni["rf"][uni["rf"].index <= as_of].iloc[-1]) * 252
    y10 = (float(uni["tnx"][uni["tnx"].index <= as_of].iloc[-1]) / 100) if uni["tnx"] is not None else rf_now + 0.01
    anchors = {"CASH": rf_now, "IEF": y10, "TLT": y10, "TIP": y10, "GLD": exp_infl}
    # valuation reversion (Bogle's speculative-return term): the implied-DY expanding-mean proxy for BOTH
    # universes. The real Shiller CAPE was tested here (W3.3) and REJECTED on the backcast — expanding-mean
    # CAPE reversion drags modern US centrals ~5%/yr low (the "expensive for 30 years" failure);
    # load_cape() remains available as a diagnostic.
    dy_hist = uni["div_yield"][uni["div_yield"].index <= as_of].dropna()
    dy_mean = float(dy_hist.expanding(min_periods=504).mean().iloc[-1]) if len(dy_hist) > 504 else dy
    val_drag = ((dy / dy_mean) ** 0.1 - 1.0) if dy_mean > 1e-6 and dy > 1e-6 else 0.0
    eq_anchor = dy + NET_BUYBACK[uni["universe"]] + REAL_GROWTH[uni["universe"]] + exp_infl + val_drag
    total = 0.0
    for k, w in book["weights"].items():
        total += w * anchors.get(k, eq_anchor)
    eq_w = sum(w for k, w in book["weights"].items() if k != "CASH")
    fee_now = float(uni["fee"][uni["fee"].index <= as_of].iloc[-1]) * eq_w
    return total - fee_now


def m3_building_block(x, h, n, rng, central):
    ann = m1_unconditional(x, h, n, rng)
    return ann + (central - float(ann.mean()))


def m4_stress(x, h, n, rng, eq_w):
    """Structural crisis member: m1 paths hit by ONE crisis inside the horizon — the equity leg loses
    20-45% (uniform; the named-stress magnitude range: 2008 −50%, CN-2015 −45%, 2020 −34%, 2022 −25%
    equity legs). Exists so the left tail carries a crash even when the trailing window lacks one."""
    ann = m1_unconditional(x, h, n, rng)
    shock = rng.uniform(0.20, 0.45, n) * eq_w
    return ((1 + ann) ** h * (1 - shock)) ** (1.0 / h) - 1


def conformal_factor(universe: str, h: int, as_of=None, archive_path: Path = CALIB_ARCHIVE) -> float:
    """Adaptive-conformal WIDEN-ONLY factor from archived resolved forecasts (prequential: only origins
    resolved by as_of). Nonconformity s = |realized - q50| / (0.5*(q90-q10)); the factor rescales the
    sample spread so ~80% of past scores fall inside the 80% band. Never narrows (floor 1.0)."""
    if not archive_path.exists():
        return 1.0
    try:
        rows = json.loads(archive_path.read_text(encoding="utf-8")).get("rows", [])
    except Exception:
        return 1.0
    cutoff = pd.Timestamp(as_of) if as_of is not None else None
    scores = []
    for r in rows:
        if r.get("universe") != universe or r.get("h") != h:
            continue
        if cutoff is not None and pd.Timestamp(r["origin"]) + pd.DateOffset(years=h) > cutoff:
            continue
        if "q50_raw" in r and "half80_raw" in r and r["half80_raw"] > 1e-9:
            scores.append(abs(r["realized"] - r["q50_raw"]) / r["half80_raw"])
    if len(scores) < 10:
        return 1.0
    return float(max(1.0, np.quantile(scores, 0.80)))


def prequential_member_weights(universe: str, h: int, as_of, archive_path: Path = CALIB_ARCHIVE):
    """Operational v1.2 member-weight rule, using only member scores resolved by `as_of`.

    None means the declared equal-weight warm-up.  This closes the prior harness/runtime mismatch: locked
    forecasts and product research now use the same score-weight rule as the calibration backcast.
    """
    if not Path(archive_path).exists():
        return None
    try:
        rows = json.loads(Path(archive_path).read_text(encoding="utf-8")).get("rows", [])
    except Exception:
        return None
    cutoff = pd.Timestamp(as_of)
    resolved = [r for r in rows if r.get("universe") == universe and r.get("h") == h
                and "pin_m1_dn" in r
                and pd.Timestamp(r["origin"]) + pd.DateOffset(years=h) <= cutoff]
    if len(resolved) < 10:
        return None
    pins = {m: float(np.mean([r[f"pin_{m}_dn"] for r in resolved])) for m in ("m1", "m2", "m3")}
    scale = max(float(np.mean(list(pins.values()))), 1e-9)
    raw = {m: float(np.exp(-pins[m] / (0.5 * scale))) for m in pins}
    total = sum(raw.values())
    return {m: raw[m] / total for m in raw}


def apply_conformal(samples: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return samples
    med = float(np.median(samples))
    return med + (samples - med) * factor


def se_of_mean(x, rng, n_boot=200) -> float:
    """Block-bootstrap SE of the ANNUALIZED mean return (the parameter-uncertainty scale)."""
    x = _as_total(x)
    n = len(x)
    nb = max(n // EBLOCK, 2)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(EBLOCK)[None, :]).ravel() % n
        means[i] = x[idx].mean() * 252
    return float(means.std())


# ------------------------------------------------------------------ the engine ------------------
def forecast(uni: dict, book: dict, as_of, h: int, n_per_member=600, seed=311,
             members=("m1", "m2", "m3"), param_uncertainty=True, conformal=None,
             member_weights=None) -> dict:
    """conformal: None = raw; "archive" = prequential factor from the committed calibration archive;
    a float = an externally computed (harness-prequential) factor. Widen-only either way.
    member_weights: None = equal; a dict {m1:w,...} of prequential score-based weights (v1.2) —
    the mixture composition, not the samples, changes."""
    as_of = pd.Timestamp(as_of)
    rng = np.random.default_rng(seed + h * 7)
    x_total = book_daily_net(uni, book, as_of)
    if len(x_total) < 252 * 5:
        raise ValueError(f"under 5y of history before {as_of.date()} for {book['name']}")
    # v1.4: decompose the book so sampled paths carry a PERSISTENT cash leg (changes from the current
    # rate), not resampled historical rate levels; books without CASH are bit-identical to v1.3
    w_cash = float(book["weights"].get("CASH", 0.0))
    rf_hist = uni["rf"][uni["rf"].index <= as_of].to_numpy()[-len(x_total):]
    x = RateAwareSeries(x_total, x_total - w_cash * rf_hist,
                        np.diff(rf_hist, prepend=rf_hist[:1]), rf_hist[-1], w_cash)
    central = cma_central(uni, book, as_of)
    parts = {}
    if "m1" in members:
        parts["m1"] = m1_unconditional(x, h, n_per_member, rng)
    if "m2" in members:
        parts["m2"] = m2_regime(x, h, n_per_member, rng)
    if "m3" in members:
        parts["m3"] = m3_building_block(x, h, n_per_member, rng, central)
    if member_weights:
        total = sum(member_weights.get(k, 1.0) for k in parts) or 1.0
        pooled = [p[:max(int(round(len(p) * len(parts) * member_weights.get(k, 1.0) / total)), 20)]
                  for k, p in parts.items()]
    else:
        pooled = list(parts.values())
    # the stress member joins the FULL ensemble only; the m1-only baseline stays the pure P-2 kernel
    sw = STRESS_WEIGHT.get(uni["universe"], 0.0) if len(parts) >= 3 else 0.0
    if sw > 0:
        eq_w = sum(w for k, w in book["weights"].items() if k != "CASH")
        n4 = int(sw / (1 - sw) * sum(len(p) for p in pooled))
        parts["m4"] = m4_stress(x, h, n4, rng, eq_w)
        pooled.append(parts["m4"])
    member_centrals = np.array([float(p.mean()) for p in pooled])
    nom = np.concatenate(pooled)
    if param_uncertainty:
        # residual-variance rule (v1.1): member disagreement IS mean uncertainty — only the remainder
        # of se(mu)^2 beyond the member-central variance is added, killing the CN double count
        se = se_of_mean(x, rng)
        se_resid = float(np.sqrt(max(se ** 2 - member_centrals.var(), 0.0)))
        nom = nom + rng.normal(0.0, se_resid, len(nom))
    q50_raw = float(np.median(nom))
    half80_raw = float(0.5 * (np.quantile(nom, 0.90) - np.quantile(nom, 0.10)))
    factor = (conformal if isinstance(conformal, (int, float)) else
              conformal_factor(uni["universe"], h, as_of) if conformal == "archive" else 1.0)
    nom = apply_conformal(nom, float(factor))
    # inflation paths: consecutive h-year runs resampled from annual CPI known at as_of (<= year-1)
    infl_hist = uni["inflation"][uni["inflation"].index <= as_of.year - 1].to_numpy()
    starts = rng.integers(0, max(len(infl_hist) - h, 1), len(nom))
    infl_path = np.array([infl_hist[s:s + h].mean() if len(infl_hist) >= h else infl_hist.mean()
                          for s in starts])
    real = (1 + nom) / (1 + infl_path) - 1
    qs = {f"q{int(q*100):02d}": round(float(np.quantile(nom, q)), 4)
          for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}
    goals = GOALS[uni["universe"]]
    return {"engine_version": ENGINE_VERSION, "universe": uni["universe"], "book": book["name"],
            "as_of": str(as_of.date()), "horizon_years": h,
            "n_samples": int(len(nom)), "members": list(members),
            "stress_member_weight": sw, "conformal_factor": round(float(factor), 4),
            "q50_raw": round(q50_raw, 4), "half80_raw": round(half80_raw, 4),
            "cma_central_net_nominal": round(central, 4),
            "quantiles_net_nominal_ann": qs,
            "median_net_real_ann": round(float(np.median(real)), 4),
            "real_q10": round(float(np.quantile(real, 0.10)), 4),
            "P_nominal": {g: round(float((nom > v).mean()), 3) for g, v in goals.items()},
            "P_real_positive": round(float((real > 0).mean()), 3),
            "member_weights_used": ({k: round(member_weights.get(k, 1.0), 3) for k in parts if k != "m4"}
                                     if member_weights else "equal"),
            "ess": {"independent_h_windows_in_history": int(len(x) / 252 / h),
                    "bootstrap_block_ess": int(len(x) / EBLOCK),
                    "note": "path count is NOT sample size; these are the honest effective sizes"},
            "calibration_status": "UNCALIBRATED_RESEARCH_ESTIMATES (until exp_w3_calibration gates PASS)",
            "_samples_nominal": nom, "_samples_real": real,
            "_member_samples": {k: p for k, p in parts.items()}}


def operational_forecast(uni: dict, book: dict, as_of, h: int, **kwargs) -> dict:
    """The sole forward/ledger path: prequential weights + archived widen-only conformal wrapper."""
    if "member_weights" in kwargs or "conformal" in kwargs:
        raise ValueError("operational_forecast owns member_weights and conformal; use forecast() for research")
    weights = prequential_member_weights(uni["universe"], h, as_of)
    out = forecast(uni, book, as_of, h, member_weights=weights, conformal="archive", **kwargs)
    out["operational_pipeline"] = "prequential_member_weights+archive_widen_only_conformal"
    return out


def stress_report(uni: dict, book: dict, as_of) -> dict:
    as_of = pd.Timestamp(as_of)
    x = book_daily_net(uni, book, as_of)
    idx = (uni["rf"].index if uni["universe"] == "CN" else uni["components"]["SPY"].dropna().index)
    idx = idx[idx <= as_of][-len(x):]
    s = pd.Series(x, index=idx)
    out = {}
    for name, (a, b) in STRESS_WINDOWS.items():
        w = s[(s.index >= a) & (s.index <= b)]
        if len(w) > 20:
            eq = np.concatenate([[1.0], np.cumprod(1 + w.to_numpy())])
            out[name] = {"total_return": round(float(eq[-1] - 1), 4),
                         "maxDD": round(float((eq / np.maximum.accumulate(eq) - 1).min()), 4)}
    # hypothetical joint stress: equity sleeve -30% over a year with +5% inflation
    eq_w = sum(w for k, w in book["weights"].items() if k != "CASH")
    rf_now = float(uni["rf"][uni["rf"].index <= as_of].iloc[-1]) * 252
    nom = eq_w * (-0.30) + (1 - eq_w) * rf_now
    out["hypothetical_eq-30_infl+5"] = {"net_nominal": round(nom, 4),
                                        "net_real": round((1 + nom) / 1.05 - 1, 4)}
    return out


def write_ledger(fc: dict, stresses: dict | None = None) -> Path:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    rec = {k: v for k, v in fc.items() if not k.startswith("_")}
    rec["status"] = "LOCKED_PENDING_REALIZATION"
    rec["locked_at_utc"] = _utc_now()
    rec["input_hashes"] = {"approved_panel": _sha(APPROVED) if APPROVED.exists() else None,
                           "inflation": _sha(INFL_CSV) if INFL_CSV.exists() else None}
    if stresses:
        rec["stress_report"] = stresses
    f = LEDGER_DIR / (f"{fc['universe']}_{fc['book']}_{fc['horizon_years']}y_asof_{fc['as_of']}"
                      f"_{fc['engine_version']}.json")
    if f.exists():
        raise FileExistsError(f"ledger entry exists — a locked forecast is never overwritten: {f.name}")
    f.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    return f


BOOKS_W3 = {
    "US": [{"name": "SPY100", "weights": {"SPY": 1.0}},
           {"name": "SPY6040", "weights": {"SPY": .45, "EFA": .15, "IEF": .20, "TLT": .10, "TIP": .05, "GLD": .05}},
           {"name": "SPY_c40", "weights": {"SPY": 0.6, "CASH": 0.4}},
           {"name": "SPY_c80", "weights": {"SPY": 0.2, "CASH": 0.8}}],
    "CN": [{"name": "CSI500_TR", "weights": {"CSI500": 1.0}},
           {"name": "CSI500_c40", "weights": {"CSI500": 0.6, "CASH": 0.4}},
           {"name": "CSI500_c80", "weights": {"CSI500": 0.2, "CASH": 0.8}}],
}


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    print(f"=== {ENGINE_VERSION} — first LOCKED forward forecasts (research-only) ===")
    for universe in ("US", "CN"):
        uni = load_universe(universe)
        as_of = (uni["rf"].index.max() if universe == "CN"
                 else uni["components"]["SPY"].dropna().index.max())
        for book in BOOKS_W3[universe][:2]:
            st = stress_report(uni, book, as_of)
            for h in (1, 3, 5, 10):
                fc = operational_forecast(uni, book, as_of, h)
                try:
                    f = write_ledger(fc, st if h == 1 else None)
                    tag = f"locked -> {f.name}"
                except FileExistsError:
                    tag = "already locked (kept)"
                print(f"[{universe} {book['name']:9s} {h:2d}y] median nom {fc['quantiles_net_nominal_ann']['q50']:+.2%} "
                      f"(q10 {fc['quantiles_net_nominal_ann']['q10']:+.2%}) | median real "
                      f"{fc['median_net_real_ann']:+.2%} | P(>infl) {fc['P_nominal']['inflation']:.0%} | {tag}")


if __name__ == "__main__":
    main()
