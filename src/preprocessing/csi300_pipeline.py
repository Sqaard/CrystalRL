"""PROJECT-FAITHFUL preprocessing pipeline ported to the Qlib csi300 universe.

WHY THIS EXISTS
---------------
The canonical preprocessing recipe lives in ``Preprocessing/Data_preprocessing.ipynb`` (48 cells,
7 stages) and was written for FinRL's Dow-30(+AMZN) Yahoo universe with WRDS US fundamentals. The
interpretability thesis wants the SAME feature contract reproduced on a public CN benchmark
(Qlib csi300) so a third party can re-run it on data they can download. This module is the
notebook turned into a reusable, runnable script -- the modularization is itself a deliverable.

It is the PROJECT-FAITHFUL feature contract (the user chose this over Qlib-native Alpha158). It is
kept SEPARATE from the FSDS-PIT panel and is NOT merged with it.

THE 7 STAGES (faithful port; per-stage status documented in PREPROCESSING_CSI300.md)
------------------------------------------------------------------------------------
1. Prices            -- ADAPTED: FinRL YahooDownloader(Dow-30+AMZN) -> Qlib csi300 OHLCV via
                        qlib.data.D. Same downstream long-frame (date, tic, open/high/low/close/
                        volume, day).
2. Fundamentals      -- ADAPTED (CN, best-effort): WRDS has no CN coverage. Sourced from akshare
                        (free, ``stock_financial_abstract_ths`` per symbol, by report period).
                        Same LTM rolling-4Q sums + ``rdq``-style PIT filing-date merge
                        (``pd.merge_asof`` backward by ticker). If akshare/network is unavailable
                        this stage is SKIPPED and flagged as a documented GAP (never fabricated).
3. Macro + HMM       -- PORTED (CN macro): VIX/^GSPC/^TNX -> CN macro via Qlib csi300 index proxy
                        (the bundle has no ^VIX), StandardScaler + hmmlearn 2-state GaussianHMM,
                        causal (past-only) filtering -> ``Market_Regime`` exactly as cell 26.
4. Tech indicators   -- ADAPTED: pandas_ta is uninstallable on Py-3.9 (PyPI now requires >=3.12 and
                        the old repo is gone) -> the 3 pandas_ta indicators (ATR/OBV/volume_ratio)
                        AND the 4 FinRL FeatureEngineer indicators (macd, rsi_30, cci_30, dx_30) +
                        turbulence are re-implemented with the IDENTICAL formulas in pure
                        pandas/numpy. Heavy FinRL is NOT installed.
5. RL transforms     -- PORTED: cell 34 ``prepare_rl_features`` (day_sin/day_cos, atr_rel,
                        obv_pct_change) verbatim.
6. GRU forecasts     -- PORTED: cell 39 ``add_gru_forecasts_safe`` (bidirectional GRU + attention,
                        predicts RETURNS, train_end-gated, MinMaxScaler fit on train only, no
                        lookahead) + cell 46 multi-horizon aggregation. Epochs kept SMALL (it is a
                        feature, not the result).
7. Impute            -- PORTED: cell 43 ``safe_impute_missing_data`` (ticker ffill then train-only
                        median, no leak) -> a processed_final-shaped CSV.

LEAK-SAFETY PRESERVED: the HMM is fit on train-only macro; the GRU scaler/training is gated at
``train_end``; impute medians are computed on train-only rows. ``train_end`` defaults preserve the
notebook's split concept (TRAIN_END_DATE = '2021-10-01').

RUN
---
Run with the Py-3.9 venv that has pyqlib:
    C:/Users/ivanp/.qlib_venv39/Scripts/python.exe -m src.preprocessing.csi300_pipeline --help
    C:/Users/ivanp/.qlib_venv39/Scripts/python.exe -m src.preprocessing.csi300_pipeline \
        --n-names 40 --start 2018-01-01 --end 2021-12-31 --gru-epochs 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable when run as a file (not just `-m`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Notebook-faithful defaults (cells 12, 40, 43). The notebook trained on
# 2010..2021-10 and tested 2021-10..2023-03; for a tractable csi300 proof we
# default to a ~4y window that includes 2018-2021 and keep the TRAIN_END split.
# ---------------------------------------------------------------------------
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2021-12-31"
DEFAULT_TRAIN_END = "2021-10-01"  # == notebook TRAIN_END_DATE
DEFAULT_N_NAMES = 40
REGION = "cn"
MARKET = "csi300"

_OUT_DIR = _REPO_ROOT / "data" / "adapters" / "_csi300_processed"


def _resolve_bundle() -> Path:
    """Locate the CN qlib bundle (``~/.qlib/qlib_data/cn`` or ``cn_data``).

    Reuses the existing adapter's ``bundle_path`` (which accepts both the ``cn`` and ``cn_data``
    naming) so the pipeline points at the same bundle the real-smoke uses.
    """
    from data.adapters.qlib_adapter import bundle_path

    b = bundle_path(REGION)
    if b is None:
        raise FileNotFoundError(
            f"No CN qlib bundle under ~/.qlib/qlib_data/{REGION}[_data]. See DATA_ADAPTERS.md."
        )
    return b


# ===========================================================================
# STAGE 1 -- PRICES (Qlib csi300 OHLCV)   [ADAPTED from FinRL YahooDownloader]
# ===========================================================================
def stage1_prices(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    n_names: int = DEFAULT_N_NAMES,
    provider_uri: str | Path | None = None,
) -> pd.DataFrame:
    """Pull a tractable csi300 OHLCV slice from Qlib as a FinRL-shaped long frame.

    Replaces ``YahooDownloader(...).fetch_data()`` (cell 12). Emits the same columns the rest of
    the notebook expects: ``date, tic, open, high, low, close, volume, day`` (``day`` = weekday,
    matching FinRL's convention used by cell 34's cyclical encoding).
    """
    import qlib
    from qlib.data import D

    if provider_uri is None:
        provider_uri = _resolve_bundle()
    qlib.init(provider_uri=str(provider_uri), region=REGION)

    insts = D.instruments(market=MARKET)
    names = sorted(
        D.list_instruments(instruments=insts, start_time=start, end_time=end, as_list=True)
    )[:n_names]

    fields = ["$open", "$high", "$low", "$close", "$volume"]
    raw = D.features(names, fields, start_time=start, end_time=end)
    raw.columns = ["open", "high", "low", "close", "volume"]
    raw = raw.reset_index().rename(columns={"datetime": "date", "instrument": "tic"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    # FinRL's `day` is the weekday index (0=Mon..4=Fri); cell 34 maps it onto a 5-day cycle.
    raw["day"] = raw["date"].dt.dayofweek
    raw = raw.dropna(subset=["close"]).sort_values(["date", "tic"]).reset_index(drop=True)
    return raw[["date", "tic", "open", "high", "low", "close", "volume", "day"]]


def _qlib_code_to_akshare(tic: str) -> str:
    """``SH600000`` / ``SZ000001`` -> ``600000`` / ``000001`` (akshare 6-digit symbol)."""
    return tic[2:] if tic[:2] in ("SH", "SZ") else tic


def stage1_prices_akshare(start: str = DEFAULT_START, end: str = DEFAULT_END,
                          n_names: int = DEFAULT_N_NAMES, provider_uri: str | Path | None = None,
                          names: list[str] | None = None) -> pd.DataFrame:
    """OHLCV THROUGH 2022-2023 via akshare (the free Qlib cn bundle ends 2021-06).

    If ``names`` is given (e.g. csi500 constituents), uses it directly (no Qlib needed); else reads the csi300 membership
    list from Qlib. Prices are fetched per-name from ``ak.stock_zh_a_hist`` (hfq). Columns taken POSITIONALLY (akshare
    returns Chinese headers): 0=date,1=open,2=close,3=high,4=low,5=volume. Same long-frame schema as ``stage1_prices``."""
    import akshare as ak
    if names is None:
        import qlib
        from qlib.data import D
        if provider_uri is None:
            provider_uri = _resolve_bundle()
        qlib.init(provider_uri=str(provider_uri), region=REGION)
        insts = D.instruments(market=MARKET)
        names = sorted(D.list_instruments(instruments=insts, start_time=start, end_time="2021-06-11", as_list=True))[:n_names]
    else:
        names = list(names)[:n_names]
    sd, ed = start.replace("-", ""), end.replace("-", "")
    frames, ok = [], 0
    for tic in names:
        try:
            h = ak.stock_zh_a_hist(symbol=_qlib_code_to_akshare(tic), period="daily", start_date=sd, end_date=ed, adjust="hfq")
        except Exception:
            continue
        if h is None or len(h) == 0 or h.shape[1] < 6:
            continue
        g = pd.DataFrame({"date": pd.to_datetime(h.iloc[:, 0], errors="coerce"), "tic": tic,
                          "open": pd.to_numeric(h.iloc[:, 1], errors="coerce"),
                          "close": pd.to_numeric(h.iloc[:, 2], errors="coerce"),
                          "high": pd.to_numeric(h.iloc[:, 3], errors="coerce"),
                          "low": pd.to_numeric(h.iloc[:, 4], errors="coerce"),
                          "volume": pd.to_numeric(h.iloc[:, 5], errors="coerce")})
        frames.append(g); ok += 1
    if not frames:
        raise RuntimeError("akshare returned no price data for any csi300 name")
    raw = pd.concat(frames, ignore_index=True)
    raw["day"] = raw["date"].dt.dayofweek
    raw = (raw.dropna(subset=["close"]).drop_duplicates(subset=["date", "tic"], keep="last")
              .sort_values(["date", "tic"]).reset_index(drop=True))
    print(f"[stage1-akshare] fetched {ok}/{len(names)} names; date max = {raw['date'].max().date()}")
    return raw[["date", "tic", "open", "high", "low", "close", "volume", "day"]]


# ===========================================================================
# STAGE 2 -- FUNDAMENTALS (akshare CN, best-effort)  [ADAPTED from WRDS]
# ===========================================================================
def _parse_cn_number(x) -> float:
    """Parse akshare CN financial strings: '5.42亿', '83.2万', '14.45%', False/'' -> float.

    亿 = 1e8, 万 = 1e4, % -> /100. Sentinel ``False``/empty -> NaN. Already-numeric passes through.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, bool):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s in ("", "False", "--", "nan", "None"):
        return np.nan
    mult = 1.0
    if s.endswith("%"):
        s = s[:-1]
        mult = 0.01
    elif s.endswith("亿"):
        s = s[:-1]
        mult = 1e8
    elif s.endswith("万"):
        s = s[:-1]
        mult = 1e4
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


# CN quarterly reports are CUMULATIVE YTD (Q2=H1, Q3=9mo, Q4=FY). A single-quarter flow is the
# YTD diff within the same fiscal year; Q1 == its own YTD. We de-cumulate flow items so the
# notebook's rolling-4Q LTM sums behave like the US single-quarter ``*q`` fields.
def _decumulate_ytd(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = group.sort_values("report_period").copy()
    g["_year"] = g["report_period"].dt.year
    g["_q"] = g["report_period"].dt.quarter
    for c in cols:
        prev = g.groupby("_year")[c].shift(1)
        q_flow = g[c] - prev
        # First quarter of each fiscal year keeps its YTD (== single-quarter) value.
        q_flow = q_flow.where(g["_q"] != 1, g[c])
        g[c] = q_flow
    return g.drop(columns=["_year", "_q"])


# CN regulatory filing deadlines (conservative PIT lag from fiscal quarter-end to public filing).
# Q1 -> Apr 30 (~30d), Q2/H1 -> Aug 31 (~60d), Q3 -> Oct 31 (~30d), Q4/Annual -> Apr 30 next yr.
# Using the deadline (not the actual rdq, which akshare's by-period table omits) is leak-SAFE
# because the report is provably public by then. Mirrors the notebook's rdq-or-datadate+60d.
_FILING_LAG_DAYS = {1: 30, 2: 62, 3: 31, 4: 120}


def _fetch_pit_announce_dates(periods, retries: int = 4) -> dict:
    """PROPER point-in-time: the report's ACTUAL first-disclosure date (实际披露时间), not the deadline.

    ``ak.stock_yysj_em(date=YYYYMMDD)`` is the regulatory disclosure CALENDAR for a report period:
    its ``实际披露时间`` column is the date each company's report was actually first published. This is
    the genuine PIT availability date (typically BEFORE the statutory deadline), which is what proper
    point-in-time requires. We deliberately avoid ``stock_yjbb_em.最新公告日期`` — that is the *latest*
    (amended) disclosure date and can be a year or more after the original (FY2021 → 2023+ for restated
    names), which would push fundamentals *staler*, the opposite of proper-PIT.

    Keyed by ``(akshare 6-digit code, report-period Timestamp)``. Transient eastmoney truncation errors
    are retried; a period/name that never resolves is absent (stage-2 falls back to the leak-safe
    deadline). Train-only periods are unaffected by test data.
    """
    import time as _time

    import akshare as ak

    out: dict = {}
    for p in sorted({pd.Timestamp(x).normalize() for x in periods}):
        ymd = p.strftime("%Y%m%d")
        df = None
        for attempt in range(retries):
            try:
                df = ak.stock_yysj_em(date=ymd)
                break
            except Exception as exc:  # transient eastmoney truncation / rate limit
                if attempt == retries - 1:
                    print(f"  [PIT] yysj {ymd} failed after {retries} tries: {type(exc).__name__}: {exc}")
                else:
                    _time.sleep(1.5 * (attempt + 1))
        if df is None or df.empty or "实际披露时间" not in df.columns or "股票代码" not in df.columns:
            continue
        adt = pd.to_datetime(df["实际披露时间"], errors="coerce")
        for sym6, a in zip(df["股票代码"].astype(str).str.zfill(6), adt):
            if pd.notna(a):
                out[(sym6, p)] = a
        print(f"  [PIT] {ymd}: {adt.notna().sum()} actual-disclosure dates")
    return out


def stage2_fundamentals(
    price_df: pd.DataFrame,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    pit: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Best-effort CN fundamentals from akshare, ported to the notebook's LTM + PIT-merge logic.

    Returns ``(merged_df, status)``. ``status['stage2'] in {'PORTED','GAP'}``. On any akshare /
    network failure the price frame is returned unchanged and stage2 is flagged GAP (no fabrication).
    """
    status = {"stage2": "GAP", "n_funda_tickers": 0, "reason": ""}
    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover - environment-dependent
        status["reason"] = f"akshare import failed: {type(exc).__name__}: {exc}"
        return price_df.copy(), status

    tics = sorted(price_df["tic"].unique().tolist())
    # akshare's report-period table maps Chinese column names -> the notebook's renamed fields.
    col_map = {
        "净利润": "net_inc_q",        # net income (YTD; de-cumulated below)
        "营业总收入": "rev_q",         # total revenue (YTD; de-cumulated below)
        "基本每股收益": "eps_incl_ex",  # basic EPS
        "每股净资产": "BPS_raw",        # book value per share (stock value, not flow)
        "净资产收益率": "ROE_raw",      # ROE (%)
        "销售净利率": "NPM_raw",        # net profit margin (%)
        "流动比率": "cur_ratio",        # current ratio
        "速动比率": "quick_ratio",      # quick ratio
        "资产负债率": "debt_ratio_raw", # debt-to-asset (%)
    }
    rows = []
    n_ok = 0
    fetch_start = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    for tic in tics:
        sym = _qlib_code_to_akshare(tic)
        try:
            fa = ak.stock_financial_abstract_ths(symbol=sym, indicator="按报告期")
        except Exception:
            continue
        if fa is None or fa.empty or "报告期" not in fa.columns:
            continue
        fa = fa.copy()
        fa["report_period"] = pd.to_datetime(fa["报告期"], errors="coerce")
        fa = fa[(fa["report_period"] >= fetch_start) & (fa["report_period"] <= end)]
        if fa.empty:
            continue
        keep = {"report_period": fa["report_period"]}
        for cn, en in col_map.items():
            keep[en] = fa[cn].map(_parse_cn_number) if cn in fa.columns else np.nan
        sub = pd.DataFrame(keep)
        sub["tic"] = tic
        rows.append(sub)
        n_ok += 1

    if not rows:
        status["reason"] = "akshare returned no usable report-period data for any ticker"
        return price_df.copy(), status

    fund = pd.concat(rows, ignore_index=True)
    # De-cumulate the YTD flow items into single-quarter flows (US-style *q semantics).
    fund = (
        fund.groupby("tic", group_keys=False)
        .apply(lambda g: _decumulate_ytd(g, ["net_inc_q", "rev_q"]))
        .reset_index(drop=True)
    )

    # --- LTM ratios (cell 21 calculate_financial_ratios), the subset we have inputs for ---
    fund = fund.sort_values(["tic", "report_period"])

    def _ltm_sum(g, col):
        return g[col].rolling(window=4, min_periods=1).sum()

    def _ltm_ratio(g, num, den):
        return g[num].rolling(window=4, min_periods=1).sum() / g[den].rolling(window=4, min_periods=1).sum()

    fund = (
        fund.groupby("tic")
        .apply(
            lambda x: x.assign(
                NPM=_ltm_ratio(x, "net_inc_q", "rev_q"),  # LTM net profit margin
                # ROE/BPS come pre-computed by the vendor (point-in-time levels); keep them.
                ROE=x["ROE_raw"],
                BPS=x["BPS_raw"],
                EPS=x["eps_incl_ex"],
                debt_ratio=x["debt_ratio_raw"],
                revenue_growth=x["rev_q"].pct_change(4),
                eps_growth=x["eps_incl_ex"].pct_change(4),
            )
        )
        .reset_index(drop=True)
    )

    # --- PIT filing date: real announcement date (公告日期) if pit=True, else statutory deadline ---
    deadline = fund.apply(
        lambda r: r["report_period"] + pd.Timedelta(days=_FILING_LAG_DAYS[r["report_period"].quarter]),
        axis=1,
    )
    if pit:
        try:
            amap = _fetch_pit_announce_dates(fund["report_period"].unique())
        except Exception as exc:
            print(f"  [PIT] announce-date fetch unavailable ({type(exc).__name__}: {exc}); using deadline proxy")
            amap = {}
        sym6 = fund["tic"].map(_qlib_code_to_akshare).astype(str)
        announce = pd.Series(
            [amap.get((s, pd.Timestamp(p).normalize())) for s, p in zip(sym6, fund["report_period"])],
            index=fund.index,
        )
        announce = pd.to_datetime(announce, errors="coerce")
        # Leak-safe: where a real announce date exists use it; otherwise fall back to the deadline.
        fund["date_available"] = announce.fillna(deadline)
        status["pit"] = "REAL_ANNOUNCE"
        status["n_pit_announce"] = int(announce.notna().sum())
        status["n_pit_deadline_fallback"] = int(announce.isna().sum())
    else:
        fund["date_available"] = deadline
        status["pit"] = "DEADLINE_PROXY"

    funda_cols = [
        "NPM", "ROE", "BPS", "EPS", "cur_ratio", "quick_ratio",
        "debt_ratio", "revenue_growth", "eps_growth",
    ]
    fund_merge = fund[["tic", "date_available", *funda_cols]].dropna(subset=["date_available"])

    # --- PIT merge_asof backward by ticker (cell 21 merge_with_lag_and_calculate_valuations) ---
    px = price_df.copy().sort_values("date")
    merged = pd.merge_asof(
        px,
        fund_merge.sort_values("date_available"),
        left_on="date",
        right_on="date_available",
        by="tic",
        direction="backward",
    )
    # valuation ratios that need price (cell 21 calculate_valuation_ratios), clipped as in notebook
    merged["PB_ratio"] = (merged["close"] / merged["BPS"]).replace([np.inf, -np.inf], np.nan).clip(0, 10)
    merged["PE_ratio"] = (merged["close"] / merged["EPS"]).replace([np.inf, -np.inf], np.nan).clip(0, 100)

    # forward fill within ticker (cell 21), keep tic intact
    merged = merged.sort_values(["tic", "date"])
    tic_col = merged["tic"]
    merged = merged.groupby("tic").ffill()
    merged["tic"] = tic_col
    merged = merged.drop(columns=["date_available"], errors="ignore")

    status["stage2"] = "PORTED"
    status["n_funda_tickers"] = n_ok
    status["funda_columns"] = [*funda_cols, "PB_ratio", "PE_ratio"]
    return merged.reset_index(drop=True), status


# ===========================================================================
# STAGE 3 -- MACRO + HMM REGIME   [PORTED from cell 26, CN macro proxy]
# ===========================================================================
def _infer_causal_hmm_states(hmm_model, X: np.ndarray):
    """Past-only HMM filtering: posterior at t uses observations up to t only (cell 26)."""
    filtered = np.zeros((len(X), hmm_model.n_components))
    for idx in range(len(X)):
        _, posteriors = hmm_model.score_samples(X[: idx + 1])
        filtered[idx] = posteriors[-1]
    return filtered.argmax(axis=1), filtered


def _cn_macro_from_qlib(start: str, end: str, provider_uri: str | Path | None) -> pd.DataFrame:
    """CN macro context built from the csi300 index proxy in the Qlib bundle.

    The notebook used ^VIX/^GSPC/^TNX from Yahoo. The CN bundle ships no VIX/yield series, so we
    build the SAME three HMM inputs (trend, weekly return, smoothed-vol) from the csi300 index
    (``SH000300``), using realized volatility as the VIX analogue. Documented as ADAPTED macro.
    """
    import qlib
    from qlib.data import D

    if provider_uri is None:
        provider_uri = _resolve_bundle()
    qlib.init(provider_uri=str(provider_uri), region=REGION)

    # csi300 index instrument in the CN bundle.
    idx_candidates = ["SH000300", "000300", "sh000300"]
    px = None
    for code in idx_candidates:
        try:
            px = D.features([code], ["$close"], start_time=start, end_time=end)
            if px is not None and not px.empty:
                break
        except Exception:
            continue
    if px is None or px.empty:
        # Fallback: equal-weight csi300 constituents as the index proxy.
        insts = D.instruments(market=MARKET)
        names = sorted(
            D.list_instruments(instruments=insts, start_time=start, end_time=end, as_list=True)
        )
        feat = D.features(names, ["$close"], start_time=start, end_time=end)
        feat.columns = ["close"]
        idx = feat.groupby(level="datetime")["close"].mean()
        m = idx.to_frame("SP500")
    else:
        px.columns = ["SP500"]
        m = px.reset_index().rename(columns={"datetime": "date"}).set_index("date")[["SP500"]]

    m.index = pd.to_datetime(m.index).tz_localize(None)
    m = m.sort_index()
    # VIX analogue: 20d annualized realized vol of the index (the CN bundle has no ^VIX).
    ret = m["SP500"].pct_change()
    m["VIX"] = ret.rolling(20).std() * np.sqrt(252) * 100.0
    # A 10Y-yield analogue is unavailable in the bundle; emit NaN so the merge keeps the column
    # shape but the feature is honestly absent (ffill leaves it NaN -> imputed in stage 7).
    m["10Y_Yield"] = np.nan
    return m


def stage3_macro_hmm(
    df: pd.DataFrame,
    train_end: str,
    start: str,
    end: str,
    provider_uri: str | Path | None = None,
) -> pd.DataFrame:
    """Add VIX-analogue macro + a 2-state causal Gaussian-HMM ``Market_Regime`` (cell 26)."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    print("Fetching CN macro, generating features, training HMM...")
    macro = _cn_macro_from_qlib(start, end, provider_uri)

    # Same trend-based HMM features as cell 26 (200d trend, 5d return, 10d smoothed VIX).
    macro["SMA_200"] = macro["SP500"].rolling(window=200).mean()
    macro["SP500_Trend"] = (macro["SP500"] - macro["SMA_200"]) / macro["SMA_200"]
    macro["SP500_Ret_5d"] = macro["SP500"].pct_change(periods=5)
    macro["VIX_SMA_10"] = macro["VIX"].rolling(window=10).mean()
    macro = macro.dropna(subset=["SP500_Trend", "SP500_Ret_5d", "VIX_SMA_10"]).copy()

    hmm_features = ["SP500_Trend", "SP500_Ret_5d", "VIX_SMA_10"]
    train_mask = macro.index <= pd.to_datetime(train_end)
    train_macro = macro.loc[train_mask].copy()
    if train_macro.empty:
        raise ValueError("Train macro dataset is empty after feature engineering.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_macro[hmm_features].values)
    X_all = scaler.transform(macro[hmm_features].values)

    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    hmm_model.fit(X_train)
    print("Transition Matrix:\n", hmm_model.transmat_.round(3))
    print("Running causal HMM filtering (past-only posteriors)...")

    causal_regimes, regime_probs = _infer_causal_hmm_states(hmm_model, X_all)
    macro["Market_Regime"] = causal_regimes
    macro["Regime_0_Prob"] = regime_probs[:, 0]
    macro["Regime_1_Prob"] = regime_probs[:, 1]

    macro = macro.reset_index().rename(columns={"index": "date", "Date": "date"})
    if "date" not in macro.columns:
        macro = macro.rename(columns={macro.columns[0]: "date"})
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    cols_to_merge = ["date", "VIX", "10Y_Yield", "Market_Regime",
                     "Regime_0_Prob", "Regime_1_Prob", "SP500_Trend"]
    existing = [c for c in cols_to_merge if c in df.columns and c != "date"]
    df = df.drop(columns=existing, errors="ignore")
    df = pd.merge(df, macro[cols_to_merge], on="date", how="left")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    fill_cols = ["VIX", "10Y_Yield", "Market_Regime", "Regime_0_Prob", "Regime_1_Prob", "SP500_Trend"]
    df[fill_cols] = df[fill_cols].ffill()
    return df


# ===========================================================================
# STAGE 4 -- TECH INDICATORS   [ADAPTED: manual TA == pandas_ta / FinRL formulas]
# ===========================================================================
def _wilder_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (RMA) -- the exact recursion pandas_ta/TA-Lib use for ATR/RSI/DX."""
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _atr(high, low, close, length=14):
    """Average True Range (Wilder), identical to ``pandas_ta.atr``."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return _wilder_rma(tr, length)


def _obv(close, volume):
    """On-Balance Volume, identical to ``pandas_ta.obv``."""
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def _macd(close, fast=12, slow=26, signal=9):
    """MACD line == FinRL/stockstats ``macd`` (EMA(fast) - EMA(slow), then minus signal EMA)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # stockstats `macd` column = histogram (FinRL default)


def _rsi(close, length=30):
    """RSI (Wilder), == FinRL ``rsi_30``."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_rma(gain, length)
    avg_loss = _wilder_rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _cci(high, low, close, length=30):
    """Commodity Channel Index, == FinRL ``cci_30`` (mean-deviation form)."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad)


def _dx(high, low, close, length=30):
    """Directional Index (Wilder), == FinRL ``dx_30``."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = _wilder_rma(tr, length)
    plus_di = 100.0 * _wilder_rma(pd.Series(plus_dm, index=high.index), length) / atr
    minus_di = 100.0 * _wilder_rma(pd.Series(minus_dm, index=high.index), length) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx


def stage4_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """ATR/OBV/volume_ratio (cell 31) + macd/rsi_30/cci_30/dx_30 + turbulence (cell 36).

    Re-implements the pandas_ta indicators and FinRL FeatureEngineer's 4 indicators with the
    SAME formulas, in pure pandas/numpy (pandas_ta uninstallable on Py-3.9; FinRL not installed).
    """
    print("Adding Technical Indicators (ATR, OBV, returns, macd, rsi_30, cci_30, dx_30)...")
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    df["daily_return"] = df.groupby("tic")["close"].pct_change()

    def _apply(group):
        g = group.sort_values("date").copy()
        g["atr"] = _atr(g["high"], g["low"], g["close"], length=14)
        g["obv"] = _obv(g["close"], g["volume"])
        g["volume_ratio"] = g["volume"] / g["volume"].rolling(window=20).mean()
        g["macd"] = _macd(g["close"])
        g["rsi_30"] = _rsi(g["close"], length=30)
        g["cci_30"] = _cci(g["high"], g["low"], g["close"], length=30)
        g["dx_30"] = _dx(g["high"], g["low"], g["close"], length=30)
        return g

    df = df.groupby("tic", group_keys=False).apply(_apply)
    df = _add_turbulence(df)
    return df.reset_index(drop=True)


def _add_turbulence(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """FinRL-style turbulence: Mahalanobis distance of each day's cross-sectional returns from the
    trailing-window mean/cov (FeatureEngineer use_turbulence=True). Same definition, leak-safe
    (uses only history up to t for the cov)."""
    piv = df.pivot(index="date", columns="tic", values="daily_return").sort_index()
    dates = piv.index
    turb = pd.Series(0.0, index=dates)
    arr = piv.values
    for i in range(lookback, len(dates)):
        hist = arr[max(0, i - lookback):i]
        hist = hist[~np.isnan(hist).any(axis=1)]
        cur = arr[i]
        if len(hist) < lookback // 2 or np.isnan(cur).any():
            continue
        mu = hist.mean(axis=0)
        cov = np.cov(hist, rowvar=False)
        try:
            inv = np.linalg.pinv(cov)
        except Exception:
            continue
        diff = (cur - mu).reshape(1, -1)
        turb.iloc[i] = float(diff @ inv @ diff.T)
    turb_df = turb.reset_index()
    turb_df.columns = ["date", "turbulence"]
    return df.merge(turb_df, on="date", how="left")


# ===========================================================================
# STAGE 5 -- RL-READY TRANSFORMS   [PORTED from cell 34 prepare_rl_features]
# ===========================================================================
def stage5_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical weekday encoding + stationary ATR/OBV transforms (cell 34, verbatim)."""
    data = df.copy().sort_values(["tic", "date"]).reset_index(drop=True)
    data["day_sin"] = np.sin(2 * np.pi * data["day"] / 5)
    data["day_cos"] = np.cos(2 * np.pi * data["day"] / 5)
    data["atr_rel"] = data["atr"] / data["close"]
    data["obv_pct_change"] = (
        data.groupby("tic")["obv"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    return data


# ===========================================================================
# STAGE 6 -- GRU RETURN-FORECAST FEATURES   [PORTED from cell 39 add_gru_forecasts_safe]
# ===========================================================================
def _build_gru_forecaster():
    import torch.nn as nn

    class GRUForecaster(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, forecast_steps=5):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.forecast_steps = forecast_steps
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim * forecast_steps),
            )

        def forward(self, x):
            import torch

            batch_size = x.size(0)
            h0 = torch.zeros(self.gru.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            gru_out, _ = self.gru(x, h0)
            attn = __import__("torch").softmax(self.attention(gru_out), dim=1)
            context = __import__("torch").sum(attn * gru_out, dim=1)
            return self.fc(context).view(batch_size, self.forecast_steps, 1)

    return GRUForecaster


def stage6_gru_forecasts(
    df: pd.DataFrame,
    train_end: str,
    lookback: int = 30,
    forecast_steps: int = 5,
    epochs: int = 8,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Bidirectional GRU+attention return forecasts, train_end-gated, no lookahead (cell 39).

    Predicts RETURNS (not price), fits the MinMaxScaler and trains strictly on rows up to
    ``train_end``, then aggregates the 5 horizons (cell 46). Epochs default SMALL (it's a feature).
    """
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import DataLoader, TensorDataset

    print("Starting SAFE GRU forecasting (predicting returns, no lookahead)...")
    GRUForecaster = _build_gru_forecaster()

    exclude = ["date", "date_available", "tic"]
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    df = df.copy()
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tics = df["tic"].unique()
    input_dim = len(features)
    all_forecasts = {}

    for idx, tic in enumerate(tics, 1):
        tdf = df[df["tic"] == tic].sort_values("date").reset_index(drop=True)
        prices = tdf["close"].values
        dates = tdf["date"].values
        data = tdf[features].values

        train_mask = pd.to_datetime(dates) <= pd.to_datetime(train_end)
        train_len = int(train_mask.sum())
        if train_len < lookback + forecast_steps:
            print(f"Skipping {tic}: not enough training data.")
            continue

        scaler = MinMaxScaler()
        scaler.fit(data[:train_len])
        norm = scaler.transform(data)

        X, y = [], []
        for i in range(lookback, len(norm) - forecast_steps):
            X.append(norm[i - lookback:i])
            base = prices[i - 1]
            y.append((prices[i:i + forecast_steps] - base) / base)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        train_y_len = train_len - lookback - forecast_steps
        X_train, y_train = X[:train_y_len], y[:train_y_len]
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size, shuffle=True,
        )
        model = GRUForecaster(input_dim, 64, 1, forecast_steps=forecast_steps).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                loss = crit(model(bx).squeeze(-1), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        Xt = torch.tensor(X).to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xt), batch_size):
                preds.extend(model(Xt[i:i + batch_size]).squeeze(-1).cpu().numpy())
        arr = np.full((len(tdf), forecast_steps), np.nan)
        arr[lookback:len(norm) - forecast_steps] = preds
        all_forecasts[tic] = arr
        print(f"Processed {tic} ({idx}/{len(tics)})")

    for step in range(1, forecast_steps + 1):
        df[f"gru_return_forecast_{step}d"] = np.nan
    for tic in tics:
        if tic in all_forecasts:
            mask = df["tic"] == tic
            for step in range(1, forecast_steps + 1):
                df.loc[mask, f"gru_return_forecast_{step}d"] = all_forecasts[tic][:, step - 1]

    # --- cell 46 multi-horizon aggregation ---
    fcols = [f"gru_return_forecast_{s}d" for s in range(1, forecast_steps + 1)]
    df["forecast_mean"] = df[fcols].mean(axis=1)
    df["forecast_std"] = df[fcols].std(axis=1)
    df["forecast_trend"] = df[f"gru_return_forecast_{forecast_steps}d"] - df["gru_return_forecast_1d"]
    return df


# ===========================================================================
# STAGE 7 -- TRAIN-ONLY MEDIAN IMPUTE   [PORTED from cell 43 safe_impute_missing_data]
# ===========================================================================
def stage7_impute(df: pd.DataFrame, train_end: str) -> pd.DataFrame:
    """Ticker ffill, then fill remaining NaNs with TRAIN-ONLY medians (cell 43, no leak)."""
    print("Imputing missing data securely (ticker ffill + train-only median)...")
    df = df.sort_values(["tic", "date"])
    df = df.groupby("tic", group_keys=False).apply(lambda x: x.ffill())
    train_df = df[df["date"] <= pd.to_datetime(train_end)]
    medians = train_df.select_dtypes(include=[np.number]).median()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(medians[col] if col in medians and pd.notna(medians[col]) else 0)
    return df


# ===========================================================================
# END-TO-END RUNNER
# ===========================================================================
def run_pipeline(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    train_end: str = DEFAULT_TRAIN_END,
    n_names: int = DEFAULT_N_NAMES,
    gru_epochs: int = 8,
    gru_lookback: int = 30,
    provider_uri: str | Path | None = None,
    out_dir: Path = _OUT_DIR,
    skip_fundamentals: bool = False,
    prices_source: str = "qlib",
    names: list[str] | None = None,
    pit: bool = False,
) -> dict:
    """Run all 7 stages end-to-end and write a processed_final-shaped CSV.

    Returns a status dict with per-stage outcome, the final panel shape, and the output path.
    """
    from data.adapters.panel_schema import build_model_ready_frame, write_model_ready

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"window": [start, end], "train_end": train_end, "n_names_requested": n_names, "stages": {}}

    # Stage 1
    df = (stage1_prices_akshare(start, end, n_names, provider_uri, names=names) if prices_source == "akshare"
          else stage1_prices(start, end, n_names, provider_uri))
    status["stages"]["1_prices"] = {"status": "ADAPTED", "source": prices_source, "rows": int(len(df)),
                                    "tickers": int(df["tic"].nunique()), "dates": int(df["date"].nunique())}
    print(f"[stage1] prices: {len(df)} rows, {df['tic'].nunique()} tickers, {df['date'].nunique()} dates")

    # Stage 2 (best-effort CN fundamentals)
    if skip_fundamentals:
        status["stages"]["2_fundamentals"] = {"status": "GAP", "reason": "skipped by flag"}
    else:
        df, fstat = stage2_fundamentals(df, start, end, pit=pit)
        status["stages"]["2_fundamentals"] = fstat
        print(f"[stage2] fundamentals: {fstat['stage2']} ({fstat.get('n_funda_tickers', 0)} tickers) "
              f"PIT={fstat.get('pit')} announce={fstat.get('n_pit_announce', '-')}/deadline_fallback={fstat.get('n_pit_deadline_fallback', '-')}")

    # Stage 3
    df = stage3_macro_hmm(df, train_end, start, end, provider_uri)
    status["stages"]["3_macro_hmm"] = {"status": "PORTED",
                                       "regimes": sorted(pd.Series(df["Market_Regime"]).dropna().unique().tolist())}
    print(f"[stage3] macro+HMM: regimes={status['stages']['3_macro_hmm']['regimes']}")

    # Stage 4
    df = stage4_tech_indicators(df)
    status["stages"]["4_tech"] = {"status": "ADAPTED",
                                  "added": ["atr", "obv", "volume_ratio", "macd", "rsi_30", "cci_30", "dx_30", "turbulence"]}
    print("[stage4] tech indicators added")

    # Stage 5
    df = stage5_rl_features(df)
    status["stages"]["5_rl_transforms"] = {"status": "PORTED",
                                           "added": ["day_sin", "day_cos", "atr_rel", "obv_pct_change"]}
    print("[stage5] RL stationary transforms added")

    # Stage 6
    df = stage6_gru_forecasts(df, train_end, lookback=gru_lookback, epochs=gru_epochs)
    status["stages"]["6_gru"] = {"status": "PORTED", "epochs": gru_epochs, "lookback": gru_lookback,
                                 "added": ["gru_return_forecast_1d..5d", "forecast_mean", "forecast_std", "forecast_trend"]}
    print("[stage6] GRU forecasts added")

    # Stage 7
    df = stage7_impute(df, train_end)
    status["stages"]["7_impute"] = {"status": "PORTED"}
    print("[stage7] train-only median imputation done")

    # Persist the raw processed_final-shaped frame (notebook's cell 47 output shape).
    raw_csv = out_dir / "processed_final_csi300.csv"
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df.to_csv(raw_csv, index=False)
    status["processed_final_csv"] = str(raw_csv)
    status["processed_final_shape"] = [int(df.shape[0]), int(df.shape[1])]

    # Also emit a strict model_ready rectangle (env-loadable) via the shared schema builder.
    feature_cols = [c for c in df.columns
                    if c not in ("date", "tic", "close", "open", "high", "low", "volume", "day")]
    model_ready = build_model_ready_frame(df, feature_columns=feature_cols, require_complete=True)
    mr_csv = out_dir / "csi300_model_ready.csv"
    write_model_ready(model_ready, mr_csv)
    status["model_ready_csv"] = str(mr_csv)
    status["model_ready_shape"] = [int(model_ready.shape[0]), int(model_ready.shape[1])]
    status["model_ready_tickers"] = sorted(model_ready["tic"].unique().tolist())
    status["n_features"] = len(feature_cols)
    print(f"[done] processed_final={df.shape}  model_ready={model_ready.shape}  -> {mr_csv}")
    return status


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="PROJECT-FAITHFUL csi300 preprocessing pipeline (7 stages)")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    ap.add_argument("--n-names", type=int, default=DEFAULT_N_NAMES)
    ap.add_argument("--gru-epochs", type=int, default=8)
    ap.add_argument("--gru-lookback", type=int, default=30)
    ap.add_argument("--skip-fundamentals", action="store_true", help="force stage-2 GAP (offline)")
    ap.add_argument("--prices", choices=["qlib", "akshare"], default="qlib",
                    help="price source; 'akshare' reaches 2022-2023 (Qlib free bundle ends 2021-06)")
    ap.add_argument("--names-file", default=None, help="text file of SH/SZ codes (one per line), e.g. csi500 constituents")
    ap.add_argument("--pit", action="store_true",
                    help="proper point-in-time fundamentals: merge on the REAL announce date (公告日期) via "
                         "ak.stock_yjbb_em instead of the statutory-deadline proxy")
    ap.add_argument("--out-dir", default=str(_OUT_DIR))
    args = ap.parse_args(argv)

    info = run_pipeline(
        start=args.start, end=args.end, train_end=args.train_end, n_names=args.n_names,
        gru_epochs=args.gru_epochs, gru_lookback=args.gru_lookback,
        out_dir=Path(args.out_dir), skip_fundamentals=args.skip_fundamentals, prices_source=args.prices,
        names=([l.strip() for l in open(args.names_file) if l.strip()] if args.names_file else None),
        pit=args.pit,
    )
    import json

    print("=" * 78)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
