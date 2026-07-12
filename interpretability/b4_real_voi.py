"""B4-REAL — the first REAL intraday execution-economics VoI test (BTCUSDT futures, 2023-03: the SVB month —
calm first week, vol storm mid-month = natural regime contrast).

Question (the program's main bet): does regime-tracking pay in REAL execution economics — i.e., does the
maker's provide-edge FLIP SIGN (or materially move) between benign and toxic regimes, giving belief-VoI > 0
where daily data provably gave 0.0 (WH2, B4-proxy)?

Data (Binance Vision, free): futures um daily aggTrades (exact maker/taker direction via is_buyer_maker —
no Lee-Ready needed) + monthly 1m klines (mid proxy for markouts + the regime clock). bookDepth kept for
later depth work.

Method (leak-safe):
  1. regime: 1m log-returns -> rolling 60m realized vol -> shift(1) -> hysteresis(0.80/0.55) episodes.
  2. per-minute, per-maker-side VWAP from aggTrades (chunked): maker_sell = aggressor-buy trades
     (is_buyer_maker=False), maker_buy = aggressor-sell.
  3. maker edge at horizon D: sell-side (p_vwap − m_{t+D})/m_t, buy-side (m_{t+D} − p_vwap)/m_t, in bp,
     volume-weighted across sides; m = 1m close.
  4. per-regime volume-weighted mean edge with MOVING-BLOCK bootstrap CI (120-min blocks); fees: gross and
     net of 2bp maker fee (Binance futures default tier; VIP tiers lower/rebates — sensitivity note).
  5. VoI (static, per-minute): regime-gated providing vs the best regime-blind policy.

Run: python interpretability/b4_real_voi.py
"""
from __future__ import annotations
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.series_g.generators import _hysteresis  # noqa: E402

DATA = ROOT / "data/intraday_crypto/btcusdt_2023-03"
OUT = HERE / "b4_real_voi_report.json"
MAKER_FEE_BP = 2.0
HORIZONS = (1, 5)                      # minutes
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qvol", "count",
           "taker_buy_vol", "taker_buy_qvol", "ignore"]


def load_klines():
    z = zipfile.ZipFile(DATA / "BTCUSDT-1m-2023-03.zip")
    raw = z.read(z.namelist()[0])
    first = raw[:64].decode("utf-8", "ignore")
    k = pd.read_csv(io.BytesIO(raw), header=0 if first.lower().startswith("open_time") else None,
                    names=KL_COLS, usecols=["open_time", "close"])
    k["minute"] = (k["open_time"] // 60_000).astype(np.int64)
    return k.set_index("minute")["close"].astype(float)


def minute_maker_vwap():
    """Aggregate every trade of the month into per-(minute, maker_side) VWAP + volume."""
    parts = []
    files = sorted(DATA.glob("BTCUSDT-aggTrades-2023-03-*.zip"))
    for f in files:
        z = zipfile.ZipFile(f)
        raw = z.open(z.namelist()[0])
        head = raw.read(64).decode("utf-8", "ignore"); raw.seek(0) if hasattr(raw, "seek") else None
        z2 = zipfile.ZipFile(f)  # reopen (stream seek unsupported)
        src = z2.open(z2.namelist()[0])
        names = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker"]
        rdr = pd.read_csv(src, header=0 if head.lower().startswith("agg") else None, names=names,
                          usecols=["price", "qty", "ts", "is_buyer_maker"], chunksize=2_000_000)
        for ch in rdr:
            ch["minute"] = (ch["ts"] // 60_000).astype(np.int64)
            ch["pq"] = ch["price"] * ch["qty"]
            g = ch.groupby(["minute", "is_buyer_maker"], sort=False)[["qty", "pq"]].sum()
            parts.append(g)
        print(f"  [b4-real] {f.name} done", flush=True)
    allg = pd.concat(parts).groupby(level=[0, 1]).sum()
    allg["vwap"] = allg["pq"] / allg["qty"]
    return allg


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    close = load_klines()
    print(f"[b4-real] klines: {len(close)} minutes")
    # regime: rolling 60m realized vol of 1m log-returns, PAST-only
    lr = np.log(close).diff()
    vol = lr.rolling(60).std().shift(1)
    m = vol.notna()
    minutes = close.index[m.values]
    tox = pd.Series(_hysteresis(vol[m].to_numpy(), 0.80, 0.55), index=minutes)
    print(f"[b4-real] regimes: toxic_rate={tox.mean():.3f}, episodes sticky "
          f"(p_stay_tox={1 - ((tox.values[:-1]==1)&(tox.values[1:]==0)).sum()/max(1,(tox.values[:-1]==1).sum()):.4f})")

    g = minute_maker_vwap()
    # per-minute maker edges at horizons
    rows = []
    close_d = close.to_dict()
    for (minute, ibm), r in g.iterrows():
        if minute not in close_d:
            continue
        m0 = close_d[minute]
        for D in HORIZONS:
            mD = close_d.get(minute + D)
            if mD is None:
                continue
            # is_buyer_maker=True -> maker BOUGHT at vwap; False -> maker SOLD
            edge = ((mD - r["vwap"]) if ibm else (r["vwap"] - mD)) / m0 * 1e4
            rows.append({"minute": minute, "D": D, "edge_bp": edge, "qty": r["qty"]})
    e = pd.DataFrame(rows)
    e["toxic"] = e["minute"].map(tox).fillna(0).astype(int)
    print(f"[b4-real] edge rows: {len(e)} (minute x side x horizon)")

    rng = np.random.default_rng(0)
    def wmean(df):
        return float(np.average(df["edge_bp"], weights=df["qty"]))
    def block_ci(df, n=1500, bl=120):
        mins = np.sort(df["minute"].unique()); T = len(mins)
        if T < bl * 3:
            return [float("nan")] * 2
        bymin = {mn: gg for mn, gg in df.groupby("minute")}
        vals = []
        nb = T // bl
        for _ in range(n):
            starts = rng.integers(0, T - bl, nb)
            sel = np.concatenate([mins[s:s + bl] for s in starts])
            sub = pd.concat([bymin[mn] for mn in sel if mn in bymin])
            vals.append(wmean(sub))
        return [round(float(np.quantile(vals, q)), 3) for q in (0.025, 0.975)]

    res = {}
    for D in HORIZONS:
        eD = e[e["D"] == D]
        for reg, name in ((0, "benign"), (1, "toxic")):
            sub = eD[eD["toxic"] == reg]
            res[f"D{D}_{name}"] = {"edge_bp_gross": round(wmean(sub), 3), "ci": block_ci(sub),
                                    "n_minutes": int(sub["minute"].nunique())}
    # VoI (static per-minute, at D=5, net of fee): gated vs best blind
    p_tox = float(tox.mean())
    eb = res["D5_benign"]["edge_bp_gross"] - MAKER_FEE_BP
    et = res["D5_toxic"]["edge_bp_gross"] - MAKER_FEE_BP
    blind = max(0.0, (1 - p_tox) * eb + p_tox * et)
    gated = (1 - p_tox) * max(0.0, eb) + p_tox * max(0.0, et)
    voi = gated - blind
    sign_flip = bool((eb > 0) != (et > 0))
    report = {
        "data": "BTCUSDT um futures 2023-03 (SVB month), Binance Vision", "maker_fee_bp": MAKER_FEE_BP,
        "regimes": {"toxic_rate": round(p_tox, 3)},
        "edges": res,
        "net_edges_D5_bp": {"benign": round(eb, 3), "toxic": round(et, 3)},
        "sign_flip_net": sign_flip,
        "VoI_bp_per_provided_minute": round(voi, 3),
        "verdict": ("CORNER OPENS on real intraday execution economics: the provide-edge flips sign across "
                    "regimes and belief-VoI > 0 — the Series-G structure exists in reality; proceed to the "
                    "real execution env (Series-G-real)." if sign_flip and voi > 0.1 else
                    "corner does NOT open at this granularity/market/costs — extend (symbols/periods/fees) "
                    "or the config-D law covers real execution economics too."),
        "honest_notes": ["1m-close mid proxy (no BBO): markouts include half-spread noise, symmetric across regimes",
                          "maker fee 2bp = default tier; VIP/rebate tiers shift net edges uniformly",
                          "single symbol, single month (chosen FOR its regime contrast); replication queue: more months/symbols",
                          "static per-minute VoI (no inventory dynamics) — the full MDP comes with the env"],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
