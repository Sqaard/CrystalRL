"""B4-REAL v2 — wider-spread instruments + TRUE BBO (the (a)-vs-(b) discriminator).

v1 (BTCUSDT, 1m-close mid proxy) found regime-flat negative maker edges — but BTCUSDT's ultra-tight spread
(~0.5-1bp) means the half-spread can't cover adverse selection ANYWHERE, so v1 cannot distinguish
(a) "config-D extends to real execution economics" from (b) "wrong instrument". v2 tests three wider-spread
alts (SOLUSDT / AVAXUSDT / GALAUSDT, um futures, 2023-08 — calm first half + the Aug-17 flash-crash cascade)
with the TRUE best-bid/offer stream (bookTicker):

  per second: mid + quoted spread from BBO (last update per second)
  per trade:  effective half-spread = |p − mid_s|/mid_s ; maker realized edge at Δ = side·(p − mid_{s+Δ})/mid_s
              (exact maker side from is_buyer_maker)
  regimes:    1m-kline vol(60m, shift 1) hysteresis(0.80/0.55) — leak-safe
  aggregation: per-minute volume-weighted; month stats per regime with 120-min moving-block bootstrap CI
  economics:  net of 2bp maker fee; VoI = regime-gated providing vs best regime-blind policy

Per-symbol verdict: does the provide-edge flip sign across regimes (corner opens) — or stay flat (config-D)?
Run: python interpretability/b4_real_voi_v2.py
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

SYMS = ["SOLUSDT", "AVAXUSDT", "GALAUSDT"]
MONTH = "2023-08"
DATA = ROOT / "data/intraday_crypto"
OUT = HERE / "b4_real_voi_v2_report.json"
MAKER_FEE_BP = 2.0
HORIZONS_S = (60, 300)
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qvol", "count",
           "taker_buy_vol", "taker_buy_qvol", "ignore"]


def read_zip_csv(path, names, usecols, chunksize=None):
    z = zipfile.ZipFile(path)
    inner = z.namelist()[0]
    head = z.open(inner).read(96).decode("utf-8", "ignore")
    hdr = 0 if any(head.lower().startswith(p) for p in (names[0].lower(), "open_time", "update_id", "agg")) else None
    return pd.read_csv(z.open(inner), header=hdr, names=names, usecols=usecols, chunksize=chunksize)


def day_second_mid(sym, day):
    """Per-second last (mid, spread_bp) from bookTicker for one day, as VECTORIZED arrays indexed by
    second-offset from day start (+600s tail for horizon lookups). Returns (day0_sec, mid_arr, spr_arr)."""
    f = DATA / f"{sym}_{MONTH}" / f"{sym}-bookTicker-{MONTH}-{day:02d}.zip"
    if not f.exists() or f.stat().st_size < 1000:
        return None
    names = ["update_id", "bid_p", "bid_q", "ask_p", "ask_q", "ts", "event_time"]
    day0 = None
    N = 86_400 + 600
    mid_arr = np.full(N, np.nan); spr_arr = np.full(N, np.nan)
    for ch in read_zip_csv(f, names, ["bid_p", "ask_p", "ts"], chunksize=3_000_000):
        sec = (ch["ts"].to_numpy() // 1000).astype(np.int64)
        if day0 is None:
            day0 = int(sec[0] // 86_400 * 86_400)
        off = sec - day0
        ok = (off >= 0) & (off < N)
        off = off[ok]
        bid = ch["bid_p"].to_numpy()[ok]; ask = ch["ask_p"].to_numpy()[ok]
        mid = (bid + ask) / 2.0
        spr = (ask - bid) / mid * 1e4
        mid_arr[off] = mid            # later rows overwrite -> last-per-second (file is chronological)
        spr_arr[off] = spr
    # forward-fill gaps (seconds with no BBO update carry the previous quote)
    idx = np.where(~np.isnan(mid_arr), np.arange(N), 0)
    np.maximum.accumulate(idx, out=idx)
    mid_arr = mid_arr[idx]; spr_arr = spr_arr[idx]
    return day0, mid_arr, spr_arr


def process_symbol(sym):
    base = DATA / f"{sym}_{MONTH}"
    # regimes from 1m klines
    kz = base / f"{sym}-1m-{MONTH}.zip"
    k = pd.concat([c for c in [read_zip_csv(kz, KL_COLS, ["open_time", "close"])]])
    k["minute"] = (k["open_time"] // 60_000).astype(np.int64)
    close = k.set_index("minute")["close"].astype(float).sort_index()
    lr = np.log(close).diff()
    vol = lr.rolling(60).std().shift(1)
    mmask = vol.notna()
    tox = pd.Series(_hysteresis(vol[mmask].to_numpy(), 0.80, 0.55), index=close.index[mmask.values])
    p_tox = float(tox.mean())

    minute_rows = []
    qspread_rows = []
    for day in range(1, 32):
        got = day_second_mid(sym, day)
        if got is None:
            continue
        day0, mid_arr, spr_arr = got
        # quoted spread per minute (mean over the day's seconds)
        sec_off = np.arange(86_400)
        qs = pd.DataFrame({"minute": (day0 + sec_off) // 60, "spr": spr_arr[:86_400]}).dropna()
        qspread_rows.append(qs.groupby("minute")["spr"].mean())
        # trades (fully vectorized second->mid via the day arrays)
        tf = base / f"{sym}-aggTrades-{MONTH}-{day:02d}.zip"
        if not tf.exists():
            continue
        names = ["agg_id", "price", "qty", "first_id", "last_id", "ts", "is_buyer_maker"]
        N = len(mid_arr)
        for ch in read_zip_csv(tf, names, ["price", "qty", "ts", "is_buyer_maker"], chunksize=2_000_000):
            sec = (ch["ts"].to_numpy() // 1000).astype(np.int64)
            off = sec - day0
            ok = (off >= 1) & (off + max(HORIZONS_S) < N)
            off = off[ok]
            price = ch["price"].to_numpy()[ok]; qty = ch["qty"].to_numpy()[ok]
            side = np.where(ch["is_buyer_maker"].to_numpy()[ok].astype(bool), 1.0, -1.0)  # maker bought:+1/sold:-1
            mid_now = mid_arr[off - 1]                       # mid BEFORE the trade second (stale-safe)
            rows = {"minute": (sec[ok] // 60), "qty": qty,
                    "eff_bp": np.abs(price - mid_now) / mid_now * 1e4}
            for D in HORIZONS_S:
                mid_fut = mid_arr[off + D]
                rows[f"edge{D}_bp"] = side * (mid_fut - price) / mid_now * 1e4  # sold: -(mf-p)=p-mf ✓
            df = pd.DataFrame(rows).dropna()
            df["wq"] = df["qty"]
            for c in ["eff_bp"] + [f"edge{D}_bp" for D in HORIZONS_S]:
                df[c] = df[c] * df["wq"]
            g = df.groupby("minute")[["wq", "eff_bp"] + [f"edge{D}_bp" for D in HORIZONS_S]].sum()
            minute_rows.append(g)
        print(f"  [{sym}] day {day:02d} done", flush=True)
    mm = pd.concat(minute_rows).groupby(level=0).sum()
    for c in mm.columns:
        if c != "wq":
            mm[c] = mm[c] / mm["wq"]
    mm["toxic"] = mm.index.map(tox).fillna(0).astype(int)
    qsp = pd.concat(qspread_rows).groupby(level=0).mean()
    mm["qspread_bp"] = mm.index.map(qsp)

    rng = np.random.default_rng(0)
    def stats(col, reg):
        sub = mm[(mm["toxic"] == reg)].dropna(subset=[col])
        w = sub["wq"].to_numpy(); v = sub[col].to_numpy()
        mean = float(np.average(v, weights=w))
        mins = sub.index.to_numpy(); T = len(mins); bl = 120
        if T < 3 * bl:
            return mean, [float("nan")] * 2
        nb = T // bl; vals = []
        for _ in range(1200):
            st = rng.integers(0, T - bl, nb)
            idx = np.concatenate([np.arange(s, s + bl) for s in st])
            vals.append(float(np.average(v[idx], weights=w[idx])))
        return mean, [round(float(np.quantile(vals, q)), 3) for q in (0.025, 0.975)]

    res = {"toxic_rate": round(p_tox, 3), "n_minutes": int(len(mm))}
    for reg, nm in ((0, "benign"), (1, "toxic")):
        qmean, qci = stats("qspread_bp", reg)
        emean, eci = stats("eff_bp", reg)
        res[nm] = {"quoted_spread_bp": round(qmean, 3), "quoted_ci": qci,
                   "effective_half_spread_bp": round(emean, 3)}
        for D in HORIZONS_S:
            m, ci = stats(f"edge{D}_bp", reg)
            res[nm][f"maker_edge_D{D//60}m_bp"] = round(m, 3)
            res[nm][f"maker_edge_D{D//60}m_ci"] = ci
    eb = res["benign"]["maker_edge_D5m_bp"] - MAKER_FEE_BP
    et = res["toxic"]["maker_edge_D5m_bp"] - MAKER_FEE_BP
    blind = max(0.0, (1 - p_tox) * eb + p_tox * et)
    gated = (1 - p_tox) * max(0.0, eb) + p_tox * max(0.0, et)
    res["net_D5m"] = {"benign": round(eb, 3), "toxic": round(et, 3)}
    res["sign_flip_net"] = bool((eb > 0) != (et > 0))
    res["VoI_bp_per_minute"] = round(gated - blind, 3)
    return res


def main():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    out = {"month": MONTH, "maker_fee_bp": MAKER_FEE_BP, "symbols": {}}
    for s in SYMS:
        print(f"[b4v2] processing {s}...", flush=True)
        out["symbols"][s] = process_symbol(s)
    opens = [s for s, r in out["symbols"].items() if r.get("sign_flip_net") and r.get("VoI_bp_per_minute", 0) > 0.1]
    out["verdict"] = (f"CORNER OPENS on {opens} — regime-conditional provide-edge sign-flip with VoI>0 on real "
                      "wider-spread instruments (hypothesis (b) was right: instrument, not law). Proceed to "
                      "Series-G-real env on these instruments." if opens else
                      "corner does NOT open on any of the 3 wider-spread instruments with TRUE BBO — hypothesis (a) "
                      "strengthens: the config-D law extends to real execution economics at retail-accessible "
                      "granularity (queue-position economics remain the final untested refuge).")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
