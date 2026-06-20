"""
Spartan Fuel — Petroleum Marketing Analytics
=============================================
Reads the MASTER sheet of a Google Sheet (link in a .env variable GOOGLE) and
builds a marketing-analytics workbench for a fuel-retail network.
PMS = petrol (red), AGO = diesel (green), Both = combined throughput.
Light & dark theme, phone-friendly.

Setup
-----
  .env  ->  GOOGLE=https://docs.google.com/spreadsheets/d/<id>/edit?usp=sharing
            (Share -> Anyone with the link – Viewer)
  pip install -r requirements.txt
  streamlit run fuel_analytics_app.py

Target  = 2 × median of the baseline MONTHLY totals, measured against the actual
          total sold in the current period (gauge = % of target obtained).
Only the MASTER tab is used. The target has its own date window. The values
below are fixed in code, not shown in the UI.
"""

import io
import os
import re
import math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

# ───────────────────────── fixed configuration (not shown in UI) ────────────
SHEET_NAME         = "MASTER"
RUNWAY_WINDOW      = 7
PRICE_EVENT_WINDOW = 14
RANK_W_ATTAIN      = 0.40
RANK_W_VOLUME      = 0.35
RANK_W_DISCIPLINE  = 0.25
EXCLUDE_ZERO       = True
STANDARD = {"PMS": 10.0, "AGO": 10.0}            # allowable dip variance, LITRES PER DAY per station
DELIVERY_CAP = 1000.0                            # a single-day Dv above this = unbooked delivery, excluded

PCOL  = {"PMS": "#E23744", "AGO": "#1F9D57", "BOTH": "#3A6EA5"}
PSTEP = {"PMS": "rgba(226,55,68,.22)", "AGO": "rgba(31,157,87,.22)", "BOTH": "rgba(58,110,165,.22)"}
PLABEL = {"PMS": "PMS · Petrol", "AGO": "AGO · Diesel", "BOTH": "PMS + AGO (combined)"}
SCALE = {"PMS": [[0, "rgba(226,55,68,.06)"], [1, "#E23744"]],
         "AGO": [[0, "rgba(31,157,87,.06)"], [1, "#1F9D57"]],
         "BOTH": [[0, "rgba(58,110,165,.06)"], [1, "#3A6EA5"]]}
GRID = "rgba(140,140,140,.16)"
AXIS = "rgba(140,140,140,.30)"
INK  = "#8b9096"


# ───────────────────────────────── parsing ─────────────────────────────────
def parse_num(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) if np.isfinite(v) else np.nan
    s = str(v).strip()
    if s == "" or s in {"-", "–", "—"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace(",", "")
    s = "".join(ch for ch in s if ch.isdigit() or ch in ".-")
    if s in {"", "-", ".", "-."}:
        return np.nan
    try:
        n = float(s)
    except ValueError:
        return np.nan
    return -n if neg else n


def parse_date(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return pd.NaT
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(pd.Timestamp(v).date())
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if 20000 < v < 60000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(v))
        return pd.NaT
    s = str(v).strip()
    if not s:
        return pd.NaT
    if "," in s and s.split(",")[0].isalpha():
        s = s.split(",", 1)[1].strip()
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return pd.Timestamp(dt.date()) if not pd.isna(dt) else pd.NaT


def _find(header, *cands, fallback=-1):
    up = [("" if c is None else str(c)).strip().upper() for c in header]
    cset = {c.upper() for c in cands}
    for i, h in enumerate(up):
        if h in cset:
            return i
    return fallback


def build_records(rows):
    hr = -1
    for i, r in enumerate(rows[:12]):
        up = {("" if c is None else str(c)).strip().upper() for c in r}
        if "DATE" in up and "STATION" in up:
            hr = i
            break
    if hr < 0:
        hr = 1 if len(rows) > 1 else 0
    header = rows[hr]
    c = {
        "date": _find(header, "DATE", fallback=0),
        "station": _find(header, "STATION", fallback=1),
        "PMS_price": _find(header, "PMS P", "PMS PRICE", fallback=2),
        "AGO_price": _find(header, "AGO P", "AGO PRICE", fallback=3),
        "PMS_disch": _find(header, "PMS D", "PMS DISCHARGE", fallback=6),
        "AGO_disch": _find(header, "AGO D", "AGO DISCHARGE", fallback=7),
        "PMS_vol": _find(header, "PMS S", "PMS SALES", fallback=8),
        "AGO_vol": _find(header, "AGO S", "AGO SALES", fallback=9),
        "PMS_short": _find(header, "PMS SHORT", fallback=11),
        "AGO_short": _find(header, "AGO SHORT", fallback=12),
        "PMS_close": _find(header, "PMS C", "PMS CLOSING", fallback=13),
        "AGO_close": _find(header, "AGO C", "AGO CLOSING", fallback=14),
        "PMS_dip": _find(header, "PMS DIP", fallback=15),
        "AGO_dip": _find(header, "AGO DIP", fallback=16),
        "PMS_dvar": _find(header, "PMS DV", fallback=17),
        "AGO_dvar": _find(header, "AGO DV", fallback=18),
        # ---- BANKING (station-level; new columns, located by header name only) ----
        "value": _find(header, "VALUE", fallback=10),
        "banked": _find(header, "BANKED", fallback=-1),
        "bankname": _find(header, "BANK", fallback=-1),
        "deposited": _find(header, "AMOUNT DEPOSITED", "DEPOSITED", "DEPOSIT", fallback=-1),
        "balance": _find(header, "BALANCE LEFT", "BALANCE", fallback=-1),
    }

    def g(r, key):
        idx = c[key]
        return r[idx] if 0 <= idx < len(r) else None

    recs = []
    for r in rows[hr + 1:]:
        if not r:
            continue
        d = parse_date(g(r, "date"))
        st = g(r, "station")
        st = "" if st is None else str(st).strip()
        if pd.isna(d) or not st or st.upper() in {"DATE", "STATION"}:
            continue
        def _clean(x):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return None
            s = str(x).strip()
            return None if s == "" or s.lower() == "nan" else s
        banking = {
            "sales_value": parse_num(g(r, "value")),
            "deposited": parse_num(g(r, "deposited")),
            "balance_left": parse_num(g(r, "balance")),
            "bank": _clean(g(r, "bankname")),
            "banked_flag": _clean(g(r, "banked")),
        }
        for p in ("PMS", "AGO"):
            recs.append({
                "date": d, "station": st, "product": p,
                "price": parse_num(g(r, f"{p}_price")),
                "volume": parse_num(g(r, f"{p}_vol")),
                "closing": parse_num(g(r, f"{p}_close")),
                "dip": parse_num(g(r, f"{p}_dip")),
                "dip_var": parse_num(g(r, f"{p}_dvar")),
                "shortage": parse_num(g(r, f"{p}_short")),
                "discharge": parse_num(g(r, f"{p}_disch")),
                **banking,
            })
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values(["station", "product", "date"]).reset_index(drop=True)
    return df


def with_combined(df):
    """Append a synthetic 'BOTH' product = PMS + AGO summed per station/date."""
    if df.empty:
        return df
    sm = lambda s: s.sum(min_count=1)
    g = (df.groupby(["station", "date"], as_index=False)
           .agg(volume=("volume", sm), closing=("closing", sm), dip=("dip", sm),
                dip_var=("dip_var", sm), shortage=("shortage", sm),
                discharge=("discharge", sm)))
    g["product"] = "BOTH"
    g["price"] = np.nan
    for col in ("sales_value", "deposited", "balance_left", "bank", "banked_flag"):
        if col in df.columns:
            g[col] = None if col in ("bank", "banked_flag") else np.nan
    g = g[df.columns]
    out = pd.concat([df, g], ignore_index=True)
    return out.sort_values(["station", "product", "date"]).reset_index(drop=True)


# ──────────────────────────── google sheet loader ──────────────────────────
def gsheet_export_url(link):
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", link) or re.search(r"[?&]id=([a-zA-Z0-9-_]+)", link)
    if not m:
        raise ValueError("Couldn't find a spreadsheet ID in the GOOGLE link.")
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=xlsx"


def load_master(link):
    import requests
    r = requests.get(gsheet_export_url(link), timeout=40)
    r.raise_for_status()
    content = r.content
    if content[:2] != b"PK":
        raise PermissionError("The sheet isn't publicly readable. Set Share → General "
                              "access → 'Anyone with the link – Viewer'.")
    xls = pd.ExcelFile(io.BytesIO(content))
    used = next((s for s in xls.sheet_names if s.strip().upper() == SHEET_NAME),
                xls.sheet_names[0])
    raw = pd.read_excel(xls, sheet_name=used, header=None, dtype=object)
    return build_records(raw.values.tolist()), used


# ───────────────────────────── core analytics ──────────────────────────────
def _slice(df, product, start, end):
    m = (df["product"] == product) & (df["date"] >= start) & (df["date"] <= end)
    return df.loc[m]


def monthly_totals(frame):
    v = frame.dropna(subset=["volume"])
    if v.empty:
        return np.array([])
    return v.groupby(v["date"].dt.to_period("M"))["volume"].sum().values.astype(float)


def compute_targets(df, product, base_start, base_end, cur_start, cur_end,
                    exclude_zero=EXCLUDE_ZERO):
    base = _slice(df, product, base_start, base_end)
    cur = _slice(df, product, cur_start, cur_end)
    out = []
    for st in sorted(df["station"].unique()):
        months = monthly_totals(base[base["station"] == st])
        median_month = float(np.median(months)) if len(months) else np.nan
        target = median_month * 2 if not np.isnan(median_month) else np.nan
        cv = cur[cur["station"] == st]["volume"].dropna()
        cur_days = int(cv.shape[0])
        actual_total = float(cv.sum()) if cur_days else 0.0
        attainment = (actual_total / target * 100
                      if target and not np.isnan(target) and target > 0 else np.nan)
        gap = actual_total - target if not np.isnan(target) else np.nan
        out.append({"station": st, "base_months": int(len(months)),
                    "median_month": median_month, "monthly_target": target,
                    "cur_days": cur_days, "actual_total": actual_total,
                    "attainment_pct": attainment, "gap_litres": gap})
    res = pd.DataFrame(out)
    if not res.empty:
        res = res.sort_values("actual_total", ascending=False).reset_index(drop=True)
    return res


def status_label(att):
    if att is None or (isinstance(att, float) and np.isnan(att)):
        return "no data"
    if att >= 100:
        return "on / above target"
    if att >= 75:
        return "approaching"
    return "below target"


def elasticity(df, station, product, start, end, exclude_zero=EXCLUDE_ZERO):
    s = _slice(df, product, start, end)
    s = s[s["station"] == station].dropna(subset=["price", "volume"])
    pts = s[(s["price"] > 0) & (s["volume"] > 0)]
    prices = sorted(pts["price"].round(2).unique())
    res = {"n": len(pts), "n_prices": len(prices), "elasticity": np.nan,
           "r2": np.nan, "lin_slope": np.nan, "per_10pesewa": np.nan, "prices": prices}
    if len(pts) >= 4 and len(prices) >= 2:
        lx, ly = np.log(pts["price"].values), np.log(pts["volume"].values)
        b, a = np.polyfit(lx, ly, 1)
        yhat = a + b * lx
        ss_res = float(np.sum((ly - yhat) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        res["elasticity"] = float(b)
        res["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        m, _ = np.polyfit(pts["price"].values, pts["volume"].values, 1)
        res["lin_slope"] = float(m)
        res["per_10pesewa"] = float(m * 0.10)
    return res


def elast_label(E):
    if E is None or np.isnan(E):
        return "—"
    a = abs(E)
    return "Elastic" if a >= 1.1 else ("Inelastic" if a <= 0.9 else "Unit-elastic")


def elast_short(E):
    if E is None or np.isnan(E):
        return "no price movement"
    a = abs(E)
    if E < 0:
        return ("buy much less if price ↑" if a >= 1.1 else
                "barely react to price" if a <= 0.9 else "buy ~1-for-1 less if price ↑")
    return "rose with price (other factors)"


def elast_brief(E):
    if E is None or np.isnan(E):
        return ("There isn't enough price movement in the data yet to tell how customers "
                "react to price at this station.")
    a = abs(E)
    if E < 0:
        if a >= 1.1:
            return (f"Demand is **elastic** ({E:.2f}). If you raise the price, customers buy "
                    f"meaningfully less — roughly {a:.1f}% less volume for every 1% price rise — "
                    "so a price increase tends to *reduce* total revenue here, while a price cut "
                    "can win a lot of extra volume.")
        if a <= 0.9:
            return (f"Demand is **inelastic** ({E:.2f}). Customers barely change how much they buy "
                    f"when price moves (about {a:.1f}% volume change per 1% price change), so you "
                    "have room to raise price without losing much volume — revenue tends to rise.")
        return (f"Demand is roughly **unit-elastic** ({E:.2f}): volume falls about 1% for every 1% "
                "price increase, so total revenue stays broadly flat as price changes.")
    return (f"Volume moved in the **same** direction as price here ({E:+.2f}), which is unusual for "
            "fuel. It usually means other factors — network growth, supply, or seasonality — drove "
            "sales more than price did over this window.")


def all_elasticities(df, product):
    rows = []
    dmin, dmax = df["date"].min(), df["date"].max()
    for st in sorted(df["station"].unique()):
        el = elasticity(df, st, product, dmin, dmax)
        rows.append({"station": st, "elasticity": el["elasticity"], "r2": el["r2"],
                     "type": elast_label(el["elasticity"]),
                     "per_10pesewa": el["per_10pesewa"],
                     "reaction": elast_short(el["elasticity"])})
    return pd.DataFrame(rows)


def price_levels(df, station, product, start, end, exclude_zero=EXCLUDE_ZERO):
    s = _slice(df, product, start, end)
    s = s[s["station"] == station].dropna(subset=["price", "volume"])
    s = s[(s["price"] > 0) & (s["volume"] > 0)]
    if s.empty:
        return pd.DataFrame(columns=["price", "days", "avg_daily"])
    return (s.assign(price=s["price"].round(2))
              .groupby("price")["volume"].agg(days="count", avg_daily="mean")
              .reset_index().sort_values("price"))


def price_events(df, station, product, start, end,
                 window=PRICE_EVENT_WINDOW, exclude_zero=EXCLUDE_ZERO):
    s = _slice(df, product, start, end).copy()
    s = s[s["station"] == station].sort_values("date")
    priced = s.dropna(subset=["price"])
    priced = priced[priced["price"] > 0]
    rows, prev = [], None
    for _, row in priced.iterrows():
        if prev is not None and abs(row["price"] - prev) > 1e-9:
            d = row["date"]

            def avgvol(lo, hi):
                w = s[(s["date"] >= lo) & (s["date"] <= hi)]["volume"].dropna()
                w = w[w > 0] if exclude_zero else w
                return float(w.mean()) if len(w) else np.nan

            qb = avgvol(d - pd.Timedelta(days=window), d - pd.Timedelta(days=1))
            qa = avgvol(d, d + pd.Timedelta(days=window - 1))
            dP = (row["price"] - prev) / ((row["price"] + prev) / 2)
            arc = np.nan
            if not np.isnan(qb) and not np.isnan(qa) and (qa + qb) > 0:
                arc = ((qa - qb) / ((qa + qb) / 2)) / dP if dP != 0 else np.nan
            rows.append({"date": d, "old_price": prev, "new_price": row["price"],
                         "price_chg_pct": (row["price"] - prev) / prev * 100,
                         "avg_before": qb, "avg_after": qa,
                         "vol_chg_pct": (qa - qb) / qb * 100 if (not np.isnan(qb) and qb) else np.nan,
                         "arc_elasticity": arc})
        prev = row["price"]
    return pd.DataFrame(rows)


def compute_runway(df, product, as_of, window=RUNWAY_WINDOW, exclude_zero=EXCLUDE_ZERO):
    s = df[(df["product"] == product) & (df["date"] <= as_of)]
    out = []
    for st in sorted(df["station"].unique()):
        ss = s[s["station"] == st].sort_values("date")
        if ss.empty:
            continue
        last = ss.iloc[-1]
        stock, src = last["dip"], "physical dip"
        if np.isnan(stock):
            stock, src = last["closing"], "book closing"
        win = ss[ss["date"] > last["date"] - pd.Timedelta(days=window)]
        v = win["volume"].dropna()
        v = v[v > 0] if exclude_zero else v
        avg = float(v.mean()) if len(v) else np.nan
        runway = stock / avg if (avg and avg > 0 and not np.isnan(stock)) else np.nan
        risk = ("no estimate" if np.isnan(runway) else "critical" if runway < 1.5 else
                "low" if runway < 3 else "watch" if runway < 6 else "healthy")
        out.append({"station": st, "as_of": last["date"], "stock_litres": stock,
                    "stock_source": src, "avg_daily_sales": avg,
                    "days_to_run_out": runway, "risk": risk})
    res = pd.DataFrame(out)
    if not res.empty:
        res = res.sort_values("days_to_run_out", na_position="last").reset_index(drop=True)
    return res


def compute_efficiency(df, product, exclude_zero=EXCLUDE_ZERO):
    """How fast each station sells through stock: average days to stock out
    (typical stock ÷ average daily sales), the empirical refill cycle (days
    between deliveries), turnover, deliveries and stock-out days."""
    out = []
    for st in sorted(df["station"].unique()):
        ss = df[(df["product"] == product) & (df["station"] == st)].sort_values("date")
        if ss.empty:
            continue
        vols = ss["volume"].dropna()
        sell = vols[vols > 0] if exclude_zero else vols
        avg_daily = float(sell.mean()) if len(sell) else np.nan
        stock_series = ss["dip"].where(ss["dip"].notna(), ss["closing"]).dropna()
        avg_stock = float(stock_series.mean()) if len(stock_series) else np.nan
        days_to_stockout = (avg_stock / avg_daily
                            if (avg_daily and avg_daily > 0 and not np.isnan(avg_stock))
                            else np.nan)
        dq = ss[ss["discharge"].fillna(0) > 0]
        deliveries = int(len(dq))
        if deliveries >= 2:
            gaps = dq["date"].sort_values().diff().dropna().dt.days
            gaps = gaps[gaps > 0]
            refill_cycle = float(gaps.mean()) if len(gaps) else np.nan
        else:
            refill_cycle = np.nan
        span = (ss["date"].max() - ss["date"].min()).days + 1
        refills_per_month = deliveries / max(span / 30.0, 1e-9) if deliveries else 0.0
        stockout_days = int((vols == 0).sum())
        turnover = (avg_daily / avg_stock if (avg_stock and avg_stock > 0
                    and not np.isnan(avg_daily)) else np.nan)  # fraction of tank sold per day
        out.append({"station": st, "avg_daily_sales": avg_daily, "avg_stock": avg_stock,
                    "days_to_stockout": days_to_stockout, "refill_cycle_days": refill_cycle,
                    "turnover_per_day": turnover, "deliveries": deliveries,
                    "refills_per_month": refills_per_month, "stockout_days": stockout_days})
    res = pd.DataFrame(out)
    if not res.empty:
        res = res.sort_values("days_to_stockout", na_position="last").reset_index(drop=True)
    return res


def compute_variance(df, product, targets_df, cur_start, cur_end, std_lpd=10.0, cap=DELIVERY_CAP):
    """Dip variance from the sheet's PMS Dv / AGO Dv columns, summed over the period
    for the total and judged per day against std_lpd litres/day. Single-day Dv values
    larger than `cap` litres are unbooked deliveries (a delivery raised the dip but
    wasn't entered as a discharge), not stock variance, so they're excluded and
    counted separately. Percentage columns are supplementary."""
    cur = _slice(df, product, cur_start, cur_end)
    out = []
    for st in sorted(df["station"].unique()):
        cs = cur[cur["station"] == st]
        dvall = cs["dip_var"].dropna()
        anomaly_days = int((dvall.abs() > cap).sum())
        keep = cs[cs["dip_var"].isna() | (cs["dip_var"].abs() <= cap)]
        dv = keep["dip_var"].dropna()
        days = int(dv.shape[0])
        total_var = float(dv.sum()) if days else np.nan
        avg_daily = total_var / days if days else np.nan
        throughput = float(keep["volume"].dropna().sum())
        avg_thru = (throughput / days) if days else np.nan
        within = (abs(avg_daily) <= std_lpd) if (days and not np.isnan(avg_daily)) else None
        over_by = (abs(avg_daily) - std_lpd) if (days and not np.isnan(avg_daily)
                  and abs(avg_daily) > std_lpd) else 0.0
        days_over = int((dv.abs() > std_lpd).sum()) if days else 0
        var_pct = (avg_daily / avg_thru * 100) if (avg_thru and avg_thru > 0
                  and not np.isnan(avg_daily)) else np.nan
        cum_pct = (total_var / throughput * 100) if (throughput and not np.isnan(total_var)) else np.nan
        std_pct = (std_lpd / avg_thru * 100) if (avg_thru and avg_thru > 0) else np.nan
        shortage = float(cs["shortage"].dropna().sum())
        out.append({"station": st, "throughput": throughput, "days": days,
                    "dip_variance": total_var, "avg_daily_var": avg_daily,
                    "days_over": days_over, "anomaly_days": anomaly_days,
                    "allowable": std_lpd * days if days else np.nan,
                    "over_by": over_by, "var_pct": var_pct, "stock_loss_pct": cum_pct,
                    "std_pct": std_pct, "within_standard": within, "delivery_shortage": shortage})
    return pd.DataFrame(out)


def _minmax(series):
    s = series.astype(float)
    lo, hi = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo) * 100


def compute_rankings(targets_df, variance_df,
                     w_attain=RANK_W_ATTAIN, w_volume=RANK_W_VOLUME, w_disc=RANK_W_DISCIPLINE):
    if targets_df.empty:
        return pd.DataFrame()
    t = targets_df.set_index("station")
    v = variance_df.set_index("station") if not variance_df.empty else None
    df = pd.DataFrame(index=t.index)
    df["total_volume"] = t["actual_total"]
    df["attainment_pct"] = t["attainment_pct"]
    df["stock_loss_pct"] = v.reindex(df.index)["stock_loss_pct"] if v is not None else np.nan
    df["s_volume"] = _minmax(df["total_volume"])
    df["s_attain"] = _minmax(df["attainment_pct"].clip(upper=200))
    df["s_disc"] = _minmax(-df["stock_loss_pct"].abs())
    tot = w_attain + w_volume + w_disc
    df["score"] = (w_attain * df["s_attain"] + w_volume * df["s_volume"]
                   + w_disc * df["s_disc"]) / tot
    df["rank_volume"] = df["total_volume"].rank(ascending=False, method="min")
    df["rank_attain"] = df["attainment_pct"].rank(ascending=False, method="min")
    df = df.sort_values("score", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index()


# ───────────────────────────── banking analytics ───────────────────────────
def banking_frame(df, start=None, end=None):
    """Station-level banking rows (read from PMS rows where banking fields are
    attached). Value = daily cash to bank; deposited = cash deposited; balance_left
    = running unbanked balance; bank = bank name; banked_flag = Yes/No."""
    cols = ["date", "station", "sales_value", "deposited", "balance_left", "bank", "banked_flag"]
    if df.empty or "sales_value" not in df.columns:
        return pd.DataFrame(columns=cols)
    b = df[df["product"] == "PMS"][cols].copy()
    if start is not None and end is not None:
        b = b[(b["date"] >= start) & (b["date"] <= end)]
    return b.sort_values(["station", "date"]).reset_index(drop=True)


def has_value(bk):
    return (not bk.empty) and bool(bk["sales_value"].notna().any())


def has_deposits(bk):
    if bk.empty:
        return False
    return bool(bk[["deposited", "balance_left"]].notna().any().any())


def compute_banking(bk):
    """Per station: cash generated (Value), deposited, banking rate, net unbanked,
    current outstanding (latest running balance), deposit count, last deposit."""
    out = []
    for st in sorted(bk["station"].unique()):
        ss = bk[bk["station"] == st].sort_values("date")
        cash = float(ss["sales_value"].dropna().sum())
        dep = float(ss["deposited"].dropna().sum())
        rate = (dep / cash * 100) if cash > 0 else np.nan
        net = cash - dep
        bl = ss["balance_left"].dropna()
        outstanding = float(bl.iloc[-1]) if len(bl) else (net if cash > 0 else np.nan)
        deposits = int((ss["deposited"].fillna(0) > 0).sum())
        ld = ss[ss["deposited"].fillna(0) > 0]["date"]
        last_deposit = ld.max() if len(ld) else pd.NaT
        out.append({"station": st, "cash_generated": cash, "deposited": dep,
                    "banking_rate": rate, "net_unbanked": net, "outstanding": outstanding,
                    "deposits": deposits, "last_deposit": last_deposit})
    res = pd.DataFrame(out)
    if not res.empty:
        res = res.sort_values("outstanding", ascending=False,
                              na_position="last").reset_index(drop=True)
    return res


def banking_by_bank(bk):
    b = bk.dropna(subset=["deposited"]).copy()
    b = b[b["bank"].notna() & (b["bank"].astype(str).str.strip() != "")]
    if b.empty:
        return pd.DataFrame(columns=["bank", "deposited", "deposits"])
    b["bank"] = b["bank"].astype(str).str.strip()
    return (b.groupby("bank")["deposited"].agg(deposited="sum", deposits="count")
              .reset_index().sort_values("deposited", ascending=False))


def banking_summary(bk):
    if not has_value(bk):
        return "No banking/Value figures found in this period yet."
    cb = compute_banking(bk)
    cash = cb["cash_generated"].sum()
    dep = cb["deposited"].sum()
    rate = dep / cash * 100 if cash else np.nan
    bits = [f"Network generated <b>GHS {cash:,.0f}</b> of bankable cash; "
            f"<b>GHS {dep:,.0f}</b> deposited"
            + (f" (<b>{rate:.0f}%</b> banked)." if not np.isnan(rate) else ".")]
    risk = cb.dropna(subset=["outstanding"])
    risk = risk[risk["outstanding"] > 0]
    if len(risk):
        top = risk.iloc[0]
        bits.append(f"⚠ Largest unbanked balance: <b>{top['station']}</b> "
                    f"(GHS {top['outstanding']:,.0f}).")
    if not has_deposits(bk):
        bits.append("Deposit columns aren't populated yet, so figures reflect cash generated only.")
    return " ".join(bits)


def analyst_summary(plabel, targets_df, runway_df):
    """Network read for the CURRENT period (target attainment basis)."""
    if targets_df.empty:
        return "No data in the selected windows."
    tot_a = targets_df["actual_total"].sum()
    tot_t = targets_df["monthly_target"].sum(skipna=True)
    overall = tot_a / tot_t * 100 if tot_t else np.nan
    bits = []
    if not np.isnan(overall):
        verdict = ("ahead of" if overall >= 100 else
                   "tracking toward" if overall >= 75 else "behind")
        bits.append(f"Network {plabel}: <b>{tot_a:,.0f} L</b> sold this period vs a monthly target of "
                    f"<b>{tot_t:,.0f} L</b> — <b>{overall:.0f}%</b> obtained, {verdict} plan.")
    valid = targets_df.dropna(subset=["attainment_pct"]).sort_values("attainment_pct", ascending=False)
    if len(valid):
        top, bot = valid.iloc[0], valid.iloc[-1]
        bits.append(f"This period, best target attainment: <b>{top['station']}</b> "
                    f"({top['attainment_pct']:.0f}%) · weakest: <b>{bot['station']}</b> "
                    f"({bot['attainment_pct']:.0f}%).")
    if runway_df is not None and not runway_df.empty:
        crit = runway_df[runway_df["risk"].isin(["critical", "low"])]
        if len(crit):
            bits.append(f"⚠ <b>{len(crit)}</b> tank(s) under ~3 days of cover "
                        f"({', '.join(crit['station'].head(4))}).")
        else:
            bits.append("All tanks hold healthy stock cover.")
    return " ".join(bits)


def default_windows(dmin, dmax):
    today = date.today()
    anchor = today if date(today.year, today.month, 1) <= dmax.date() else dmax.date()
    fom = date(anchor.year, anchor.month, 1)
    prev_end = fom - timedelta(days=1)
    ys = date(anchor.year, 1, 1)
    lo, hi = dmin.date(), dmax.date()
    cl = lambda d: min(max(d, lo), hi)
    bs, be = cl(ys), cl(prev_end)
    cs, ce = cl(fom), hi
    if bs > be:
        bs, be = lo, hi
    if cs > ce:
        cs = ce
    return (bs, be), (cs, ce)


# ─────────────────────── forecast / alerts / export engine ─────────────────
def forecast_month_end(df, product, targets_df, cur_e):
    """Project each station's full-month volume from its month-to-date run-rate,
    and whether it will hit its monthly target."""
    cur_e = pd.Timestamp(cur_e)
    mstart = pd.Timestamp(date(cur_e.year, cur_e.month, 1))
    dim = cur_e.days_in_month
    elapsed = (cur_e - mstart).days + 1
    mtd = _slice(df, product, mstart, cur_e)
    tmap = targets_df.set_index("station") if not targets_df.empty else None
    out = []
    for st in sorted(df["station"].unique()):
        v = mtd[mtd["station"] == st]["volume"].dropna()
        m = float(v.sum())
        rate = m / elapsed if elapsed > 0 else np.nan
        projected = rate * dim if not np.isnan(rate) else np.nan
        target = float(tmap.loc[st, "monthly_target"]) if (tmap is not None and st in tmap.index) else np.nan
        proj_attain = (projected / target * 100
                       if (target and target > 0 and not np.isnan(projected)) else np.nan)
        shortfall = (projected - target
                     if (not np.isnan(target) and not np.isnan(projected)) else np.nan)
        out.append({"station": st, "mtd": m, "daily_rate": rate, "projected": projected,
                    "monthly_target": target, "proj_attain": proj_attain,
                    "shortfall": shortfall,
                    "will_hit": (proj_attain >= 100) if not np.isnan(proj_attain) else None,
                    "elapsed": elapsed, "days_in_month": dim})
    res = pd.DataFrame(out)
    if not res.empty:
        res = res.sort_values("proj_attain", na_position="last").reset_index(drop=True)
    return res


def forecast_series(frame, horizon=30, lookback=120):
    """Trend + day-of-week seasonal forecast for a daily volume series.
    Returns (history_df, forecast_df) or None if too little data."""
    s = frame.dropna(subset=["volume"]).sort_values("date")
    s = s[s["volume"] > 0]
    if len(s) < 14:
        return None
    s = s.tail(lookback).copy()
    s["t"] = (s["date"] - s["date"].min()).dt.days
    b, a = np.polyfit(s["t"].values, s["volume"].values, 1)
    base = s["volume"].mean()
    dowf = (s.assign(dow=s["date"].dt.dayofweek).groupby("dow")["volume"].mean() / base).to_dict()
    fitted = (a + b * s["t"]) * s["date"].dt.dayofweek.map(dowf).fillna(1.0)
    resid = s["volume"].values - fitted.values
    sd = float(np.std(resid)) if len(resid) > 2 else 0.0
    last_t, last_d = s["t"].max(), s["date"].max()
    rows = []
    for h in range(1, horizon + 1):
        d = last_d + pd.Timedelta(days=h)
        f = max((a + b * (last_t + h)) * dowf.get(d.dayofweek, 1.0), 0.0)
        rows.append({"date": d, "yhat": f, "lo": max(f - 1.28 * sd, 0), "hi": f + 1.28 * sd})
    return s[["date", "volume"]], pd.DataFrame(rows)


def volume_anomalies(df, product, lookback=45, z=3.0):
    """Flag stations whose latest recorded day is a statistical outlier."""
    end = df["date"].max()
    start = end - pd.Timedelta(days=lookback)
    out = []
    for st in sorted(df["station"].unique()):
        s = df[(df["product"] == product) & (df["station"] == st)].dropna(subset=["volume"])
        s = s[s["volume"] > 0]
        recent = s[s["date"] >= start]["volume"]
        if len(recent) < 8:
            continue
        m, sd = recent.mean(), recent.std()
        if not sd or sd == 0:
            continue
        last = s.sort_values("date").iloc[-1]
        zz = (last["volume"] - m) / sd
        if abs(zz) >= z:
            out.append({"station": st, "date": last["date"], "volume": last["volume"],
                        "z": zz, "mean": m})
    return pd.DataFrame(out)


def build_excel(sheets):
    """sheets: dict name -> DataFrame. Returns xlsx bytes."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, d in sheets.items():
            dd = d if (isinstance(d, pd.DataFrame) and not d.empty) else pd.DataFrame({"info": ["no data"]})
            dd.to_excel(xw, sheet_name=str(name)[:31], index=False)
    return bio.getvalue()


# ─────────────────────────────────── theme ─────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --acc:#E23744;--muted:#8b9096;
  --line:rgba(140,140,140,.18);--card:rgba(140,140,140,.06);--card2:rgba(140,140,140,.11);
  --shadow:0 1px 2px rgba(0,0,0,.05),0 10px 26px rgba(0,0,0,.07);
  --shadow-lg:0 22px 60px rgba(0,0,0,.24);--r:16px;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
h1,h2,h3,h4{font-family:'Sora',sans-serif;letter-spacing:-.015em;}
.block-container{padding-top:1.0rem;padding-bottom:3.5rem;max-width:1300px;}

/* sidebar */
section[data-testid="stSidebar"]{border-right:1px solid var(--line);
  background:linear-gradient(180deg,rgba(140,140,140,.05),rgba(140,140,140,0));}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] h4{font-size:12px;
  text-transform:uppercase;letter-spacing:.14em;color:var(--muted);font-family:'IBM Plex Mono',monospace;}

/* hero — forecourt at dusk */
.hero{position:relative;overflow:hidden;border-radius:22px;padding:27px 30px;color:#fff;
  background:linear-gradient(125deg,#0C1014 0%,#1A232C 58%,#222F3C 100%);
  box-shadow:var(--shadow-lg);margin-bottom:18px;border:1px solid rgba(255,255,255,.06);}
.hero::before{content:"";position:absolute;inset:0 0 auto 0;height:1px;
  background:linear-gradient(90deg,transparent,var(--acc),transparent);opacity:.6;}
.hero::after{content:"";position:absolute;left:-80px;top:-120px;width:420px;height:300px;
  background:radial-gradient(circle,var(--acc),transparent 68%);opacity:.22;filter:blur(8px);}
.hero h1{color:#fff;font-size:25px;margin:0 0 6px;font-weight:800;line-height:1.12;position:relative;}
.hero .meta{color:#aab2bc;font-family:'IBM Plex Mono',monospace;font-size:12px;line-height:1.6;position:relative;}
.hero .badge{display:inline-block;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.20);
  color:#fff;border-radius:999px;padding:3px 11px;font-family:'IBM Plex Mono',monospace;
  font-size:10.5px;letter-spacing:.16em;margin-left:10px;vertical-align:middle;backdrop-filter:blur(6px);}

/* readout strip */
.summary{position:relative;background:linear-gradient(90deg,var(--card2),var(--card));
  border:1px solid var(--line);border-radius:16px;padding:15px 18px 15px 22px;line-height:1.62;
  font-size:14.5px;margin-bottom:18px;color:inherit;box-shadow:var(--shadow);}
.summary::before{content:"";position:absolute;left:0;top:11px;bottom:11px;width:4px;border-radius:4px;
  background:linear-gradient(180deg,var(--acc),transparent);}

/* KPI tiles — instrument panel */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:4px 0 8px;}
.kpi{position:relative;overflow:hidden;background:var(--card);border:1px solid var(--line);
  border-radius:var(--r);padding:16px 16px 18px;color:inherit;box-shadow:var(--shadow);
  transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;}
.kpi:hover{transform:translateY(-3px);box-shadow:0 16px 38px rgba(0,0,0,.13);
  border-color:rgba(140,140,140,.34);}
.kpi .l{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--muted);}
.kpi .v{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:29px;margin-top:10px;
  line-height:1;font-variant-numeric:tabular-nums;color:inherit;}
.kpi .v .u{font-size:12px;color:var(--muted);margin-left:5px;font-weight:500;}
.kpi .s{font-size:11.5px;color:var(--muted);margin-top:8px;min-height:14px;}
.kpi .tick{position:absolute;left:0;bottom:0;height:4px;width:100%;
  background:linear-gradient(90deg,var(--acc),transparent)!important;opacity:.95;}

/* eyebrow + section heads */
.eyebrow{display:flex;align-items:center;gap:9px;font-family:'IBM Plex Mono',monospace;font-size:11px;
  letter-spacing:.15em;text-transform:uppercase;color:var(--acc);font-weight:600;margin:16px 0 3px;}
.eyebrow::before{content:"";width:16px;height:2px;background:var(--acc);border-radius:2px;
  display:inline-block;box-shadow:0 0 8px var(--acc);}
.block-container h2,.block-container h3{font-weight:700;}
.prodhead{font-family:'Sora',sans-serif;font-weight:700;font-size:16px;margin:12px 0 4px;
  padding-left:11px;border-left:4px solid var(--acc2);}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);flex-wrap:wrap;padding-bottom:2px;}
.stTabs [data-baseweb="tab"]{padding:8px 15px;border-radius:11px;font-family:'Sora',sans-serif;
  font-weight:600;font-size:13px;color:var(--muted);transition:all .15s ease;}
.stTabs [data-baseweb="tab"]:hover{background:var(--card2);color:inherit;}
.stTabs [aria-selected="true"]{background:var(--acc);color:#fff!important;box-shadow:0 6px 16px rgba(0,0,0,.18);}
.stTabs [data-baseweb="tab-highlight"]{display:none;}
.stTabs [data-baseweb="tab-border"]{display:none;}

/* metrics as mini tiles */
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:13px 16px;box-shadow:var(--shadow);}
[data-testid="stMetricLabel"] p{font-family:'IBM Plex Mono',monospace;font-size:10px!important;
  letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace;font-weight:600;
  font-variant-numeric:tabular-nums;}

/* dataframe + charts */
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);}
[data-testid="stPlotlyChart"]{border-radius:14px;}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:11px;font-family:'Sora',sans-serif;
  font-weight:600;border:1px solid var(--line);transition:all .15s ease;}
.stButton>button:hover{border-color:var(--acc);color:var(--acc);transform:translateY(-1px);}
.stDownloadButton>button{background:var(--acc);color:#fff;border:none;}
.stDownloadButton>button:hover{filter:brightness(1.08);color:#fff;transform:translateY(-1px);}

.note{color:var(--muted);font-size:12px;font-family:'IBM Plex Mono',monospace;line-height:1.55;}
hr{border-color:var(--line);}

@media (max-width:820px){
  .kpi-row{grid-template-columns:repeat(2,1fr);}
  .hero{padding:18px 18px;border-radius:18px;} .hero h1{font-size:20px;}
  .kpi .v{font-size:23px;} .block-container{padding-left:.7rem;padding-right:.7rem;}
}
@media (max-width:430px){ .kpi-row{grid-template-columns:1fr;} }
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>
"""


def style_fig(fig, height=340, accent="#E23744"):
    fig.update_layout(
        height=height, margin=dict(l=12, r=16, t=50, b=12),
        font=dict(family="Inter, sans-serif", size=12, color=INK),
        title=dict(font=dict(family="Sora, sans-serif", size=15.5, color=INK), x=0,
                   xanchor="left", pad=dict(l=2, b=6)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.16, x=1, xanchor="right",
                    font=dict(size=11, color=INK), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="rgba(16,20,26,.94)", bordercolor="rgba(255,255,255,.10)",
                        font=dict(family="IBM Plex Mono, monospace", size=11.5, color="#fff")),
        colorway=[accent, "#9A9DA3", "#C5821C", "#3A6EA5", "#7A0010", "#1F9D57"])
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=INK),
                     title_font=dict(color=INK, size=11.5))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=INK),
                     title_font=dict(color=INK, size=11.5))
    return fig


# ─────────────────────────────────── UI ────────────────────────────────────
def main():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    st.set_page_config(page_title="Spartan Fuel Analytics", layout="wide", page_icon="⛽")
    st.markdown(CSS, unsafe_allow_html=True)

    link = os.getenv("GOOGLE")
    if not link:
        st.title("⛽ Spartan Fuel — Marketing Analytics")
        st.error("No data source. Create a **.env** with `GOOGLE=<Google Sheets link>` "
                 "and set sharing to *Anyone with the link – Viewer*.")
        st.stop()

    @st.cache_data(ttl=600, show_spinner="Reading the MASTER sheet…")
    def _load(lnk):
        return load_master(lnk)

    try:
        df, used_sheet = _load(link)
    except Exception as e:
        st.title("⛽ Spartan Fuel — Marketing Analytics")
        st.error(f"Couldn't load the sheet: {e}")
        st.stop()
    if df.empty:
        st.error("The MASTER sheet has no readable daily rows (need DATE and STATION columns).")
        st.stop()

    df_all = with_combined(df)
    stations = sorted(df["station"].unique())
    dmin, dmax = df["date"].min(), df["date"].max()
    (bs_def, be_def), (cs_def, ce_def) = default_windows(dmin, dmax)

    fmt = lambda d: pd.Timestamp(d).strftime("%d %b %Y")

    # ============================ BANKING MODULE ============================
    def render_banking():
        acc, step = "#2563EB", "rgba(37,99,235,.22)"
        bscale = [[0, "rgba(37,99,235,.06)"], [1, "#2563EB"]]
        st.markdown(f"<style>:root{{--acc:{acc};}}</style>", unsafe_allow_html=True)
        with st.sidebar:
            st.markdown("#### Banking period")
            st.markdown("<span class='note'>cash generated (Value) vs amount deposited; "
                        "outstanding = unbanked cash still held.</span>", unsafe_allow_html=True)
            bperiod = st.date_input("Period", (dmin.date(), dmax.date()),
                                    min_value=dmin.date(), max_value=dmax.date(), key="bperiod")
            bfocus = st.selectbox("Station focus", ["All stations"] + stations, key="bfocus")
            st.divider()
            if st.button("↻ Refresh data", use_container_width=True, key="brefresh"):
                st.cache_data.clear()
                st.rerun()
            st.caption(f"Source: Google · sheet **{used_sheet}**")
        if isinstance(bperiod, (tuple, list)) and len(bperiod) == 2:
            bs, be = pd.Timestamp(bperiod[0]), pd.Timestamp(bperiod[1])
        else:
            bs, be = dmin, dmax
        bk = banking_frame(df, bs, be)

        st.markdown(
            f"<div class='hero'><h1>🏦 Spartan Fuel — Banking"
            f"<span class='badge'>{used_sheet}</span></h1>"
            f"<div class='meta'>{len(stations)} stations · cash reconciliation · "
            f"period {fmt(bs)} → {fmt(be)}</div></div>", unsafe_allow_html=True)

        if not has_value(bk):
            st.info("No banking/Value figures in this range yet. The **Value** column feeds "
                    "cash generated, and **Amount Deposited / Balance Left / Bank** feed the "
                    "rest — populate them in the sheet and this section fills in automatically.")
            return

        st.markdown(f"<div class='summary'>🏦 {banking_summary(bk)}</div>", unsafe_allow_html=True)
        cb = compute_banking(bk)

        if bfocus == "All stations":
            cash = cb["cash_generated"].sum()
            dep = cb["deposited"].sum()
            outst = cb["outstanding"].sum(skipna=True)
            ctx = "All stations"
        else:
            row = cb[cb["station"] == bfocus]
            cash = float(row["cash_generated"].iloc[0]) if len(row) else 0.0
            dep = float(row["deposited"].iloc[0]) if len(row) else 0.0
            outst = float(row["outstanding"].iloc[0]) if len(row) else np.nan
            ctx = bfocus
        rate = dep / cash * 100 if cash else np.nan

        g0 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"

        def kcards(items):
            cards = "".join(
                f"<div class='kpi'><div class='l'>{l}</div>"
                f"<div class='v'>{v}<span class='u'>{u}</span></div>"
                f"<div class='s'>{s}</div><div class='tick' style='background:{acc}'></div></div>"
                for l, v, u, s in items)
            st.markdown(f"<div class='kpi-row'>{cards}</div>", unsafe_allow_html=True)

        deposits_seen = has_deposits(bk)
        btabs = st.tabs(["Overview", "Reconciliation", "By bank", "Trend"])

        # -------- Overview --------
        with btabs[0]:
            st.markdown(f"<div class='eyebrow'>{ctx} · cash banking</div>", unsafe_allow_html=True)
            kcards([
                ("Cash generated (Value)", g0(cash), "GHS", "to be banked this period"),
                ("Amount deposited", g0(dep), "GHS", "actually banked"),
                ("Banked", "—" if np.isnan(rate) else f"{rate:,.0f}", "%",
                 "of cash deposited"),
                ("Outstanding (unbanked)", g0(outst), "GHS", "cash still held / at risk"),
            ])
            if not deposits_seen:
                st.caption("Deposit columns aren't populated yet — banked % and deposits show 0 "
                           "until you fill Amount Deposited.")
            st.write("")
            c1, c2 = st.columns([1, 1.5], gap="large")
            with c1:
                gv = 0 if np.isnan(rate) else rate
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=gv,
                    number={"suffix": "%", "font": {"size": 40, "color": acc}},
                    title={"text": "of cash banked", "font": {"size": 13, "color": INK}},
                    gauge={"axis": {"range": [0, max(110, gv + 10)], "tickcolor": INK,
                                    "tickfont": {"color": INK}},
                           "bar": {"color": acc, "thickness": 0.3},
                           "bgcolor": "rgba(140,140,140,.10)", "borderwidth": 0,
                           "steps": [{"range": [0, 90], "color": "rgba(140,140,140,.10)"},
                                     {"range": [90, 100], "color": step}],
                           "threshold": {"line": {"color": INK, "width": 3},
                                         "thickness": 0.9, "value": 100}}))
                st.plotly_chart(style_fig(gauge, 290, acc), use_container_width=True)
            with c2:
                top = cb.sort_values("cash_generated", ascending=False).head(12).iloc[::-1]
                labels = ["—" if np.isnan(r) else f"{r:.0f}%" for r in top["banking_rate"]]
                fig = go.Figure()
                fig.add_bar(y=top["station"], x=top["cash_generated"], orientation="h",
                            name="Cash generated", marker_color="rgba(140,140,140,.30)")
                fig.add_bar(y=top["station"], x=top["deposited"], orientation="h", name="Deposited",
                            marker_color=acc, text=labels, textposition="outside",
                            textfont=dict(color=INK, size=11), cliponaxis=False)
                fig.update_layout(barmode="group",
                                  title="Cash generated vs deposited (label = % banked)")
                st.plotly_chart(style_fig(fig, 330, acc), use_container_width=True)

        # -------- Reconciliation --------
        with btabs[1]:
            st.markdown("<div class='eyebrow'>Per-station reconciliation</div>",
                        unsafe_allow_html=True)
            outdf = cb.dropna(subset=["outstanding"])
            outdf = outdf[outdf["outstanding"].abs() > 0].sort_values("outstanding")
            if not outdf.empty:
                fig = px.bar(outdf, x="outstanding", y="station", orientation="h",
                             title="Outstanding (unbanked) cash by station",
                             labels={"outstanding": "GHS unbanked", "station": ""})
                fig.update_traces(marker_color=["#B00020" if v > 0 else "#1F9D57"
                                                for v in outdf["outstanding"]],
                                  text=[f"{v:,.0f}" for v in outdf["outstanding"]],
                                  textposition="outside", textfont=dict(color=INK, size=11),
                                  cliponaxis=False)
                st.plotly_chart(style_fig(fig, max(280, 34 * len(outdf)), acc),
                                use_container_width=True)
            show = cb.copy()
            show["last_deposit"] = show["last_deposit"].apply(
                lambda d: "—" if pd.isna(d) else pd.Timestamp(d).strftime("%d %b %Y"))
            show = show.rename(columns={
                "station": "Station", "cash_generated": "Cash generated (GHS)",
                "deposited": "Deposited (GHS)", "banking_rate": "Banked %",
                "net_unbanked": "Net unbanked (GHS)", "outstanding": "Outstanding (GHS)",
                "deposits": "Deposits", "last_deposit": "Last deposit"})
            cols = ["Station", "Cash generated (GHS)", "Deposited (GHS)", "Banked %",
                    "Net unbanked (GHS)", "Outstanding (GHS)", "Deposits", "Last deposit"]
            st.dataframe(show[cols].style.format({
                "Cash generated (GHS)": "{:,.0f}", "Deposited (GHS)": "{:,.0f}",
                "Banked %": "{:,.0f}%", "Net unbanked (GHS)": "{:,.0f}",
                "Outstanding (GHS)": "{:,.0f}"}, na_rep="—"),
                use_container_width=True, hide_index=True,
                height=min(520, 80 + 36 * len(show)))
            st.download_button("⬇ Download banking (CSV)", show[cols].to_csv(index=False),
                               "banking.csv", "text/csv")

        # -------- By bank --------
        with btabs[2]:
            st.markdown("<div class='eyebrow'>Deposits by bank</div>", unsafe_allow_html=True)
            bb = banking_by_bank(bk)
            if bb.empty:
                st.info("No bank-tagged deposits in this period yet (the Bank and Amount "
                        "Deposited columns are still being filled in).")
            else:
                c1, c2 = st.columns([1, 1], gap="large")
                with c1:
                    donut = px.pie(bb, names="bank", values="deposited", hole=0.58)
                    donut.update_traces(textposition="outside", textinfo="label+percent",
                                        textfont=dict(color=INK),
                                        marker=dict(line=dict(color="rgba(0,0,0,0)", width=2)))
                    donut.update_layout(showlegend=False,
                                        colorway=["#2563EB", "#1F9D57", "#C5821C", "#E23744",
                                                  "#7A0010", "#3A6EA5"])
                    st.plotly_chart(style_fig(donut, 340, acc), use_container_width=True)
                with c2:
                    show = bb.rename(columns={"bank": "Bank", "deposited": "Deposited (GHS)",
                                              "deposits": "Deposits"})
                    st.dataframe(show.style.format({"Deposited (GHS)": "{:,.0f}"}),
                                 use_container_width=True, hide_index=True)

        # -------- Trend --------
        with btabs[3]:
            st.markdown("<div class='eyebrow'>Cash vs deposits over time</div>",
                        unsafe_allow_html=True)
            if bfocus == "All stations":
                s = (bk.groupby("date", as_index=False)
                     .agg(value=("sales_value", "sum"), dep=("deposited", "sum")))
                s["balance_left"] = np.nan
            else:
                s = bk[bk["station"] == bfocus][["date", "sales_value", "deposited",
                                                 "balance_left"]].rename(
                    columns={"sales_value": "value", "deposited": "dep"})
            s = s.sort_values("date")
            if s.empty:
                st.info("No data in this range.")
            else:
                s["cum_cash"] = s["value"].fillna(0).cumsum()
                s["cum_dep"] = s["dep"].fillna(0).cumsum()
                fig = go.Figure()
                fig.add_scatter(x=s["date"], y=s["cum_cash"], name="Cumulative cash generated",
                                mode="lines", line=dict(color=INK, width=1.6, dash="dot"))
                fig.add_scatter(x=s["date"], y=s["cum_dep"], name="Cumulative deposited",
                                mode="lines", line=dict(color=acc, width=3), fill="tozeroy",
                                fillcolor=step)
                fig.update_layout(title="Cumulative cash vs deposits (gap = unbanked)")
                fig.update_yaxes(title_text="GHS")
                st.plotly_chart(style_fig(fig, 340, acc), use_container_width=True)
                if bfocus != "All stations" and s["balance_left"].notna().any():
                    fig = go.Figure()
                    fig.add_scatter(x=s["date"], y=s["balance_left"], name="Balance left",
                                    mode="lines", line=dict(color="#B00020", width=2.5))
                    fig.update_layout(title="Running unbanked balance (Balance Left)")
                    fig.update_yaxes(title_text="GHS outstanding")
                    st.plotly_chart(style_fig(fig, 300, acc), use_container_width=True)
        st.caption("Banking figures in GHS · Value = daily cash to bank · Balance Left is the "
                   "running unbanked balance carried forward.")

    module = st.sidebar.radio("Module", ["STOCKS", "BANKING"], horizontal=True, key="module")
    st.sidebar.divider()
    if module == "BANKING":
        render_banking()
        return

    with st.sidebar:
        st.markdown("#### View")
        product = st.radio("Fuel grade", ["PMS", "AGO", "BOTH"],
                           format_func=lambda p: PLABEL[p])
        focus = st.selectbox("Station focus", ["All stations"] + stations)
        st.divider()
        st.markdown("#### Target window")
        st.markdown("<span class='note'>median of the baseline months × 2 = the monthly "
                    "target; actual total is measured over the current period.</span>",
                    unsafe_allow_html=True)
        base = st.date_input("Baseline period", (bs_def, be_def),
                             min_value=dmin.date(), max_value=dmax.date())
        cur = st.date_input("Current period", (cs_def, ce_def),
                            min_value=dmin.date(), max_value=dmax.date())
        st.divider()
        if st.button("↻ Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Source: Google · sheet **{used_sheet}**")

    def rng(x, fb):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return pd.Timestamp(x[0]), pd.Timestamp(x[1])
        return fb
    base_s, base_e = rng(base, (bs_def, be_def))
    cur_s, cur_e = rng(cur, (cs_def, ce_def))
    accent = PCOL[product]
    real_products = ["PMS", "AGO"] if product == "BOTH" else [product]
    st.markdown(f"<style>:root{{--acc:{accent};}}</style>", unsafe_allow_html=True)

    targets = compute_targets(df_all, product, base_s, base_e, cur_s, cur_e)
    variance_rank = compute_variance(df_all, product, targets, cur_s, cur_e)
    rankings = compute_rankings(targets, variance_rank)
    runway_all = pd.concat([compute_runway(df, rp, dmax).assign(product=rp)
                            for rp in real_products], ignore_index=True)

    fmt = lambda d: pd.Timestamp(d).strftime("%d %b %Y")
    st.markdown(
        f"<div class='hero'><h1>⛽ Spartan Fuel — Marketing Analytics"
        f"<span class='badge'>{used_sheet}</span></h1>"
        f"<div class='meta'>{len(stations)} stations · {fmt(dmin)} → {fmt(dmax)} · "
        f"viewing {PLABEL[product]} · baseline {fmt(base_s)}→{fmt(base_e)} · "
        f"current {fmt(cur_s)}→{fmt(cur_e)}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='summary'>📋 {analyst_summary(PLABEL[product], targets, runway_all)}</div>",
                unsafe_allow_html=True)

    def kpi_row(items):
        cards = "".join(
            f"<div class='kpi'><div class='l'>{l}</div>"
            f"<div class='v'>{v}<span class='u'>{u}</span></div>"
            f"<div class='s'>{s}</div><div class='tick' style='background:{a}'></div></div>"
            for l, v, u, s, a in items)
        st.markdown(f"<div class='kpi-row'>{cards}</div>", unsafe_allow_html=True)

    def phead(rp):
        st.markdown(f"<div class='prodhead' style='--acc2:{PCOL[rp]}'>{PLABEL[rp]}</div>",
                    unsafe_allow_html=True)

    f0 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"
    tabs = st.tabs(["Overview", "Targets vs Actual", "Price Sensitivity", "Days to Run Out",
                    "Efficiency", "Variance", "Rankings", "🔮 Forecast", "🚨 Alerts", "Trends"])

    # ============================ OVERVIEW ============================
    with tabs[0]:
        if focus == "All stations":
            tot_a = targets["actual_total"].sum()
            tot_t = targets["monthly_target"].sum(skipna=True)
            med = float(np.nansum(targets["median_month"].values))
            ctx = "All stations"
        else:
            row = targets[targets["station"] == focus]
            tot_a = float(row["actual_total"].iloc[0]) if len(row) else 0
            tot_t = float(row["monthly_target"].iloc[0]) if len(row) else np.nan
            med = float(row["median_month"].iloc[0]) if len(row) else np.nan
            ctx = focus
        att = tot_a / tot_t * 100 if tot_t and not np.isnan(tot_t) and tot_t > 0 else np.nan

        st.markdown(f"<div class='eyebrow'>{ctx} · {PLABEL[product]} · current period</div>",
                    unsafe_allow_html=True)
        kpi_row([
            ("Actual sold (current)", f0(tot_a), "L", "this period so far", accent),
            ("Monthly target", f0(tot_t), "L", "2× median baseline month", accent),
            ("Target obtained", "—" if np.isnan(att) else f"{att:,.0f}", "%",
             status_label(att), accent),
            ("Median month", f0(med), "L", "typical baseline month", accent),
        ])
        st.write("")
        g1, g2 = st.columns([1, 1.5], gap="large")
        with g1:
            gv = 0 if np.isnan(att) else att
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=gv,
                number={"suffix": "%", "font": {"size": 40, "color": accent}},
                title={"text": "of monthly target obtained", "font": {"size": 13, "color": INK}},
                gauge={"axis": {"range": [0, max(120, gv + 15)], "tickcolor": INK,
                                "tickfont": {"color": INK}},
                       "bar": {"color": accent, "thickness": 0.3},
                       "bgcolor": "rgba(140,140,140,.10)", "borderwidth": 0,
                       "steps": [{"range": [0, 75], "color": "rgba(140,140,140,.10)"},
                                 {"range": [75, 100], "color": PSTEP[product]}],
                       "threshold": {"line": {"color": INK, "width": 3},
                                     "thickness": 0.9, "value": 100}}))
            st.plotly_chart(style_fig(gauge, 290, accent), use_container_width=True)
        with g2:
            top = targets.dropna(subset=["actual_total"]).head(12).iloc[::-1]
            labels = ["—" if np.isnan(a) else f"{a:.0f}%" for a in top["attainment_pct"]]
            fig = go.Figure()
            fig.add_bar(y=top["station"], x=top["monthly_target"], orientation="h",
                        name="Target", marker_color="rgba(140,140,140,.30)")
            fig.add_bar(y=top["station"], x=top["actual_total"], orientation="h", name="Actual",
                        marker_color=accent, text=labels, textposition="outside",
                        textfont=dict(color=INK, size=11), cliponaxis=False)
            fig.update_layout(barmode="group",
                              title="Actual vs target by station (label = % obtained)")
            st.plotly_chart(style_fig(fig, 330, accent), use_container_width=True)

        st.markdown("<div class='eyebrow'>Month-to-date pace toward target</div>",
                    unsafe_allow_html=True)
        if focus == "All stations":
            cs = (_slice(df_all, product, cur_s, cur_e).groupby("date", as_index=False)
                  .agg(volume=("volume", "sum")))
        else:
            cs = _slice(df_all, product, cur_s, cur_e)
            cs = cs[cs["station"] == focus][["date", "volume"]]
        cs = cs.sort_values("date")
        if not cs.empty and not np.isnan(tot_t):
            cs["cum"] = cs["volume"].fillna(0).cumsum()
            m0 = pd.Timestamp(date(cur_s.year, cur_s.month, 1))
            m_end = (m0 + pd.offsets.MonthEnd(1)).normalize()
            fig = go.Figure()
            fig.add_scatter(x=[m0, m_end], y=[0, tot_t], name="Ideal pace",
                            line=dict(color=INK, width=1.4, dash="dot"))
            fig.add_scatter(x=cs["date"], y=cs["cum"], name="Cumulative actual", mode="lines",
                            line=dict(color=accent, width=3), fill="tozeroy",
                            fillcolor=PSTEP[product])
            fig.add_hline(y=tot_t, line_dash="dash", line_color=accent,
                          annotation_text="monthly target", annotation_font_color=INK)
            fig.update_layout(title="Cumulative sales vs target this month")
            fig.update_yaxes(title_text="Cumulative litres")
            st.plotly_chart(style_fig(fig, 320, accent), use_container_width=True)

    # ============================ TARGETS ============================
    with tabs[1]:
        st.markdown("<div class='eyebrow'>Target = twice the median baseline month</div>",
                    unsafe_allow_html=True)
        st.subheader("Actual total sold vs monthly target")
        view = targets.copy()
        view["status"] = view["attainment_pct"].apply(status_label)
        show = view.rename(columns={
            "station": "Station", "base_months": "Baseline months",
            "median_month": "Median month (L)", "monthly_target": "Monthly target (L)",
            "cur_days": "Operating days", "actual_total": "Actual total (L)",
            "attainment_pct": "Attainment %", "gap_litres": "Gap (L)", "status": "Status"})
        cols = ["Station", "Baseline months", "Median month (L)", "Monthly target (L)",
                "Operating days", "Actual total (L)", "Attainment %", "Gap (L)", "Status"]
        st.dataframe(show[cols].style.format({
            "Median month (L)": "{:,.0f}", "Monthly target (L)": "{:,.0f}",
            "Actual total (L)": "{:,.0f}", "Attainment %": "{:,.0f}%", "Gap (L)": "{:,.0f}"},
            na_rep="—"), use_container_width=True, hide_index=True,
            height=min(560, 80 + 36 * len(show)))
        att = targets.dropna(subset=["attainment_pct"]).sort_values("attainment_pct")
        if not att.empty:
            fig = px.bar(att, x="attainment_pct", y="station", orientation="h",
                         labels={"attainment_pct": "Attainment %", "station": ""},
                         title="Attainment % by station")
            fig.update_traces(
                marker_color=[accent if a >= 100 else (PSTEP[product] if a >= 75 else
                              "rgba(140,140,140,.45)") for a in att["attainment_pct"]],
                text=[f"{a:.0f}%" for a in att["attainment_pct"]], textposition="outside",
                textfont=dict(color=INK, size=11), cliponaxis=False)
            fig.add_vline(x=100, line_dash="dash", line_color=INK, annotation_text="target")
            st.plotly_chart(style_fig(fig, max(280, 34 * len(att)), accent), use_container_width=True)
        st.download_button("⬇ Download targets (CSV)", show[cols].to_csv(index=False),
                           f"targets_{product}.csv", "text/csv")

    # ============================ PRICE SENSITIVITY ============================
    with tabs[2]:
        st.markdown("<div class='eyebrow'>Across all available history</div>",
                    unsafe_allow_html=True)

        def render_price(rp):
            acc = PCOL[rp]
            if focus == "All stations":
                ae = all_elasticities(df, rp)
                plotable = ae.dropna(subset=["elasticity"])
                if not plotable.empty:
                    pe = plotable.sort_values("elasticity")
                    fig = px.bar(pe, x="elasticity", y="station", orientation="h",
                                 title="Price elasticity by station "
                                 "(more negative = more price-sensitive)",
                                 labels={"elasticity": "Elasticity", "station": ""})
                    fig.update_traces(marker_color=acc, text=[f"{e:.2f}" for e in pe["elasticity"]],
                                      textposition="outside", textfont=dict(color=INK, size=11),
                                      cliponaxis=False)
                    fig.add_vline(x=-1, line_dash="dot", line_color=INK, annotation_text="unit-elastic")
                    st.plotly_chart(style_fig(fig, max(280, 36 * len(pe)), acc),
                                    use_container_width=True)
                tbl = ae.rename(columns={"station": "Station", "elasticity": "Elasticity",
                                         "r2": "Fit R²", "type": "Type",
                                         "per_10pesewa": "Litres per +GHS0.10",
                                         "reaction": "Customer reaction"})
                st.dataframe(tbl.style.format({"Elasticity": "{:.2f}", "Fit R²": "{:.2f}",
                                               "Litres per +GHS0.10": "{:,.0f}"}, na_rep="—"),
                             use_container_width=True, hide_index=True,
                             height=min(480, 80 + 36 * len(tbl)))
                st.caption("Pick a single station in the sidebar for its demand curve and event study.")
            else:
                el = elasticity(df, focus, rp, dmin, dmax)
                if not np.isnan(el["elasticity"]):
                    kpi_row([
                        ("Price elasticity", f"{el['elasticity']:.2f}", "",
                         elast_label(el["elasticity"]), acc),
                        ("Litres per +GHS0.10",
                         "—" if np.isnan(el["per_10pesewa"]) else f"{el['per_10pesewa']:,.0f}",
                         "L", "linear sensitivity", acc),
                        ("Model fit", "—" if np.isnan(el["r2"]) else f"{el['r2']*100:.0f}", "%",
                         "log-log R²", acc),
                        ("Price levels", f"{el['n_prices']}", "", "distinct prices", acc)])
                st.info("💡 " + elast_brief(el["elasticity"]))
                c1, c2 = st.columns([1.25, 1], gap="large")
                with c1:
                    s = df[(df["product"] == rp) & (df["station"] == focus)].dropna(
                        subset=["price", "volume"])
                    s = s[(s["price"] > 0) & (s["volume"] > 0)]
                    if not s.empty:
                        fig = px.scatter(s, x="price", y="volume",
                                         labels={"price": "Price (GHS/L)", "volume": "Volume (L/day)"},
                                         title="Daily volume vs price — demand curve")
                        fig.update_traces(marker=dict(color=acc, size=8, opacity=0.6))
                        m, b = np.polyfit(s["price"], s["volume"], 1)
                        xs = np.array([s["price"].min(), s["price"].max()])
                        fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                        line=dict(color=INK, dash="dash", width=2))
                        st.plotly_chart(style_fig(fig, 340, acc), use_container_width=True)
                with c2:
                    pl = price_levels(df, focus, rp, dmin, dmax)
                    if not pl.empty:
                        fig = go.Figure()
                        for _, r in pl.iterrows():
                            fig.add_shape(type="line", x0=0, x1=r["avg_daily"],
                                          y0=str(r["price"]), y1=str(r["price"]),
                                          line=dict(color="rgba(140,140,140,.4)", width=2))
                        fig.add_trace(go.Scatter(x=pl["avg_daily"], y=pl["price"].astype(str),
                                                 mode="markers", marker=dict(color=acc, size=13)))
                        fig.update_layout(title="Avg daily volume by price level",
                                          xaxis_title="Avg daily (L)", yaxis_title="Price (GHS/L)")
                        st.plotly_chart(style_fig(fig, 340, acc), use_container_width=True)
                ev = price_events(df, focus, rp, dmin, dmax)
                if not ev.empty:
                    st.markdown("<div class='note'>Price-change event study</div>",
                                unsafe_allow_html=True)
                    evs = ev.rename(columns={"date": "Date", "old_price": "Old", "new_price": "New",
                                             "price_chg_pct": "Price Δ%", "avg_before": "Avg before (L)",
                                             "avg_after": "Avg after (L)", "vol_chg_pct": "Volume Δ%",
                                             "arc_elasticity": "Arc elasticity"})
                    evs["Date"] = pd.to_datetime(evs["Date"]).dt.strftime("%d %b %Y")
                    st.dataframe(evs.style.format({
                        "Old": "{:,.2f}", "New": "{:,.2f}", "Price Δ%": "{:+.1f}%",
                        "Avg before (L)": "{:,.0f}", "Avg after (L)": "{:,.0f}",
                        "Volume Δ%": "{:+.1f}%", "Arc elasticity": "{:.2f}"}, na_rep="—"),
                        use_container_width=True, hide_index=True)

                # ---- what-if price → revenue simulator ----
                sd = df[(df["product"] == rp) & (df["station"] == focus)].dropna(subset=["price", "volume"])
                sd = sd[(sd["price"] > 0) & (sd["volume"] > 0)]
                if len(sd) >= 4 and sd["price"].nunique() >= 2:
                    st.markdown("<div class='eyebrow'>What-if price → revenue</div>",
                                unsafe_allow_html=True)
                    b, a = np.polyfit(sd["price"].values, sd["volume"].values, 1)   # Q = a + bP
                    p_cur = float(sd.sort_values("date")["price"].iloc[-1])
                    q_cur = max(a + b * p_cur, 0.0)
                    prop = st.number_input(f"Proposed price (GHS/L) — {rp}", min_value=0.0,
                                           value=round(p_cur, 2), step=0.10, key=f"sim_{rp}")
                    q_prop = max(a + b * prop, 0.0)
                    rev_cur, rev_prop = p_cur * q_cur, prop * q_prop
                    drev = (rev_prop - rev_cur) / rev_cur * 100 if rev_cur else np.nan
                    cc = st.columns(3)
                    cc[0].metric("Predicted volume", f"{q_prop:,.0f} L/day",
                                 f"{(q_prop-q_cur)/q_cur*100:+.0f}%" if q_cur else None)
                    cc[1].metric("Predicted revenue", f"GHS {rev_prop:,.0f}/day",
                                 None if np.isnan(drev) else f"{drev:+.0f}%")
                    if b < 0:
                        p_opt = -a / (2 * b)
                        q_opt = max(a + b * p_opt, 0.0)
                        cc[2].metric("Revenue-max price", f"GHS {p_opt:,.2f}",
                                     f"GHS {p_opt*q_opt:,.0f}/day")
                        xs = np.linspace(max(sd["price"].min() * 0.9, 0.1), sd["price"].max() * 1.1, 60)
                        rev = xs * np.clip(a + b * xs, 0, None)
                        fig = go.Figure()
                        fig.add_scatter(x=xs, y=rev, mode="lines", name="Revenue",
                                        line=dict(color=acc, width=2.5))
                        fig.add_vline(x=p_opt, line_dash="dash", line_color="#1F9D57",
                                      annotation_text="rev-max")
                        fig.add_vline(x=p_cur, line_dash="dot", line_color=INK,
                                      annotation_text="current")
                        fig.add_scatter(x=[prop], y=[rev_prop], mode="markers", name="Proposed",
                                        marker=dict(color="#E23744", size=13))
                        fig.update_layout(title="Daily revenue vs price")
                        fig.update_xaxes(title_text="Price (GHS/L)")
                        fig.update_yaxes(title_text="Revenue (GHS/day)")
                        st.plotly_chart(style_fig(fig, 300, acc), use_container_width=True)
                    else:
                        cc[2].metric("Revenue-max price", "—", "no interior optimum")
                        st.caption("Volume rose with price in this data (no downward demand curve), "
                                   "so there's no interior revenue-maximising price — directional only.")
                    st.caption("Modelled from the fitted demand curve; fuel prices may be regulated, "
                               "so treat as a planning what-if.")

        for rp in real_products:
            if len(real_products) > 1:
                phead(rp)
            render_price(rp)

    # ============================ RUNWAY ============================
    with tabs[3]:
        st.markdown("<div class='eyebrow'>Stock cover · rolling-average method</div>",
                    unsafe_allow_html=True)
        st.caption(f"Latest tank stock ÷ {RUNWAY_WINDOW}-day rolling-average daily sales. "
                   "Physical dip used where available, else book closing stock.")
        colormap = {"critical": "#7A0010", "low": "#E23744", "watch": "#C5821C",
                    "healthy": "#1F9D57", "no estimate": "rgba(140,140,140,.55)"}

        def render_runway(rp):
            rw = runway_all[runway_all["product"] == rp].drop(columns=["product"])
            if rw.empty:
                st.info("No stock readings for this product.")
                return
            plotr = rw.dropna(subset=["days_to_run_out"])
            if not plotr.empty:
                order = list(plotr.sort_values("days_to_run_out", ascending=False)["station"])
                fig = px.bar(plotr, x="days_to_run_out", y="station", orientation="h",
                             color="risk", color_discrete_map=colormap,
                             category_orders={"station": order},
                             labels={"days_to_run_out": "Days of cover", "station": "", "risk": ""},
                             title="Stock cover by station (label = days)")
                fig.update_traces(texttemplate="%{x:.1f}", textposition="outside",
                                  textfont=dict(color=INK, size=11), cliponaxis=False)
                fig.add_vline(x=3, line_dash="dot", line_color=INK, annotation_text="3-day floor")
                st.plotly_chart(style_fig(fig, max(280, 36 * len(plotr)), PCOL[rp]),
                                use_container_width=True)
            rv = rw.copy()
            rv["as_of"] = pd.to_datetime(rv["as_of"]).dt.strftime("%d %b %Y")
            show = rv.rename(columns={"station": "Station", "as_of": "As of",
                                      "stock_litres": "Stock (L)", "stock_source": "Stock source",
                                      "avg_daily_sales": "Avg daily sales (L)",
                                      "days_to_run_out": "Days to run out", "risk": "Risk"})
            st.dataframe(show.style.format({
                "Stock (L)": "{:,.0f}", "Avg daily sales (L)": "{:,.0f}",
                "Days to run out": "{:,.1f}"}, na_rep="—"),
                use_container_width=True, hide_index=True)

        for rp in real_products:
            st.subheader(f"Days to run out — {PLABEL[rp]}")
            render_runway(rp)

    # ============================ EFFICIENCY ============================
    with tabs[4]:
        st.markdown("<div class='eyebrow'>Sell-through speed</div>", unsafe_allow_html=True)
        st.caption("Average days to stock out = typical tank stock ÷ average daily sales. "
                   "Refill cycle = average days between deliveries. Shorter = faster turnover.")

        def render_eff(rp):
            eff = compute_efficiency(df, rp)
            if eff.empty:
                st.info("No data for this product.")
                return
            valid = eff.dropna(subset=["days_to_stockout"])
            if not valid.empty:
                fast = valid.iloc[0]
                slow = valid.iloc[-1]
                st.markdown(f"<div class='note'>Fastest mover: <b>{fast['station']}</b> "
                            f"(~{fast['days_to_stockout']:.1f} days to stock out) · slowest: "
                            f"<b>{slow['station']}</b> (~{slow['days_to_stockout']:.1f} days).</div>",
                            unsafe_allow_html=True)
                order = list(valid.sort_values("days_to_stockout", ascending=False)["station"])
                fig = px.bar(valid, x="days_to_stockout", y="station", orientation="h",
                             category_orders={"station": order},
                             labels={"days_to_stockout": "Avg days to stock out", "station": ""},
                             title="Average days to stock out (shorter = sells through faster)")
                fig.update_traces(marker_color=PCOL[rp],
                                  text=[f"{d:.1f}" for d in valid["days_to_stockout"]],
                                  textposition="outside", textfont=dict(color=INK, size=11),
                                  cliponaxis=False)
                st.plotly_chart(style_fig(fig, max(280, 36 * len(valid)), PCOL[rp]),
                                use_container_width=True)
            show = eff.rename(columns={
                "station": "Station", "avg_daily_sales": "Avg daily sales (L)",
                "avg_stock": "Avg stock (L)", "days_to_stockout": "Avg days to stock out",
                "refill_cycle_days": "Refill cycle (days)", "turnover_per_day": "Turnover/day",
                "deliveries": "Deliveries", "refills_per_month": "Refills / month",
                "stockout_days": "Stock-out days"})
            st.dataframe(show.style.format({
                "Avg daily sales (L)": "{:,.0f}", "Avg stock (L)": "{:,.0f}",
                "Avg days to stock out": "{:,.1f}", "Refill cycle (days)": "{:,.1f}",
                "Turnover/day": "{:.2%}", "Refills / month": "{:,.1f}"}, na_rep="—"),
                use_container_width=True, hide_index=True)

        for rp in real_products:
            st.subheader(f"Efficiency — {PLABEL[rp]}")
            render_eff(rp)

    # ============================ VARIANCE ============================
    with tabs[5]:
        st.markdown("<div class='eyebrow'>Dip variance · vs 10 L/day standard</div>",
                    unsafe_allow_html=True)
        st.caption("Dip variance is taken from the sheet's PMS Dv / AGO Dv columns (total = sum "
                   "over the period, matching your Sheet2 VAR). It's judged against a ±10 L/day "
                   "band: a loss worse than 10 L/day flags as a loss, and a gain bigger than "
                   "10 L/day flags too (usually an unbooked delivery). Percentage columns are "
                   "supplementary.")

        def render_var(rp):
            vv = compute_variance(df, rp, pd.DataFrame(), cur_s, cur_e, STANDARD[rp])
            if vv.empty:
                st.info("No variance data.")
                return
            std = STANDARD[rp]
            within = vv["within_standard"]
            n_ok = int((within == True).sum())
            n_bad = int((within == False).sum())
            n_anom = int(vv["anomaly_days"].sum()) if "anomaly_days" in vv else 0
            anom_txt = (f" · <b>{n_anom}</b> delivery-sized day(s) (>1,000 L) excluded as unbooked "
                        f"deliveries" if n_anom else "")
            st.markdown(f"<div class='note'>Total variance = sum of the sheet's {rp} Dv "
                        f"(matches your Sheet2 VAR). Standard: <b>±{std:.0f} L/day</b> "
                        f"· within: <b>{n_ok}</b> · exceeding: <b>{n_bad}</b>{anom_txt}. A large "
                        f"swing either way is flagged — a big <b>gain</b> usually means a delivery "
                        f"that bumped the dip wasn't booked.</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                vt = vv.dropna(subset=["dip_variance"]).sort_values("dip_variance")
                if not vt.empty:
                    fig = px.bar(vt, x="dip_variance", y="station", orientation="h",
                                 title=f"Total {rp} variance for the period (L)",
                                 labels={"dip_variance": "Total variance (L)", "station": ""})
                    fig.update_traces(
                        marker_color=["#7A0010" if abs(a) / max(d, 1) > std else "#1F9D57"
                                      for a, d in zip(vt["dip_variance"], vt["days"])],
                        text=[f"{a:,.1f}" for a in vt["dip_variance"]], textposition="outside",
                        textfont=dict(color=INK, size=11), cliponaxis=False)
                    fig.add_vline(x=0, line_color=INK)
                    st.plotly_chart(style_fig(fig, max(280, 32 * len(vt)), PCOL[rp]),
                                    use_container_width=True)
            with c2:
                vp = vv.dropna(subset=["avg_daily_var"]).sort_values("avg_daily_var")
                if not vp.empty:
                    fig = px.bar(vp, x="avg_daily_var", y="station", orientation="h",
                                 title=f"Avg per day (L) vs ±{std:.0f} standard",
                                 labels={"avg_daily_var": "Litres per day", "station": ""})
                    fig.update_traces(
                        marker_color=["#7A0010" if abs(a) > std else "#1F9D57"
                                      for a in vp["avg_daily_var"]],
                        text=[f"{a:.1f}" for a in vp["avg_daily_var"]], textposition="outside",
                        textfont=dict(color=INK, size=11), cliponaxis=False)
                    fig.add_vline(x=std, line_dash="dash", line_color=INK, annotation_text=f"+{std:.0f}")
                    fig.add_vline(x=-std, line_dash="dash", line_color=INK, annotation_text=f"-{std:.0f}")
                    st.plotly_chart(style_fig(fig, max(280, 32 * len(vp)), PCOL[rp]),
                                    use_container_width=True)

            def _status(w, a):
                if w is None or pd.isna(a):
                    return "—"
                if w:
                    return "✓ within"
                return "✗ loss" if a < 0 else "⚠ gain (check)"
            show = vv.copy()
            show["Status"] = [_status(w, a) for w, a in
                              zip(show["within_standard"], show["avg_daily_var"])]
            show = show.rename(columns={
                "station": "Station", "days": "Days", "dip_variance": "Total variance (L)",
                "avg_daily_var": "Avg/day (L)", "days_over": "Days outside ±std",
                "var_pct": "Variance %/day", "std_pct": "Standard %/day",
                "delivery_shortage": "Delivery shortage (L)"})
            cols = ["Station", "Days", "Total variance (L)", "Avg/day (L)", "Days outside ±std",
                    "Variance %/day", "Standard %/day", "Status", "Delivery shortage (L)"]
            st.dataframe(show[cols].style.format({
                "Total variance (L)": "{:+,.1f}", "Avg/day (L)": "{:+,.1f}",
                "Variance %/day": "{:+.2f}%", "Standard %/day": "{:.2f}%",
                "Delivery shortage (L)": "{:,.0f}"}, na_rep="—"),
                use_container_width=True, hide_index=True)

        for rp in real_products:
            st.subheader(f"Variance — {PLABEL[rp]}")
            render_var(rp)

    # ============================ RANKINGS ============================
    with tabs[6]:
        st.markdown("<div class='eyebrow'>Composite performance index · current period</div>",
                    unsafe_allow_html=True)
        st.subheader(f"Station rankings — {PLABEL[product]}")
        st.caption(f"Blends attainment {RANK_W_ATTAIN:.0%}, throughput {RANK_W_VOLUME:.0%}, "
                   f"stock discipline {RANK_W_DISCIPLINE:.0%}.")
        if rankings.empty:
            st.info("No data to rank.")
        else:
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            rk = rankings.copy()
            rk["rank"] = rk["rank"].apply(lambda r: f"{medals.get(r, '')} {int(r)}".strip())
            plo = rankings.head(15).iloc[::-1]
            fig = px.bar(plo, x="score", y="station", orientation="h", title="Performance index",
                         labels={"score": "Index (0–100)", "station": ""})
            fig.update_traces(marker_color=accent, text=[f"{s:.0f}" for s in plo["score"]],
                              textposition="outside", textfont=dict(color=INK, size=11),
                              cliponaxis=False)
            st.plotly_chart(style_fig(fig, max(280, 36 * min(15, len(rankings))), accent),
                            use_container_width=True)
            show = rk.rename(columns={"rank": "Rank", "station": "Station", "score": "Index",
                                      "total_volume": "Volume (L)", "attainment_pct": "Attainment %",
                                      "stock_loss_pct": "Stock loss %", "rank_volume": "Vol rank",
                                      "rank_attain": "Attain rank"})
            cols = ["Rank", "Station", "Index", "Volume (L)", "Attainment %",
                    "Stock loss %", "Vol rank", "Attain rank"]
            st.dataframe(show[cols].style.format({
                "Index": "{:,.0f}", "Volume (L)": "{:,.0f}", "Attainment %": "{:,.0f}%",
                "Stock loss %": "{:+.2f}%", "Vol rank": "{:.0f}", "Attain rank": "{:.0f}"},
                na_rep="—"), use_container_width=True, hide_index=True)

    # ============================ FORECAST ============================
    with tabs[7]:
        st.markdown("<div class='eyebrow'>Run-rate projection · current month</div>",
                    unsafe_allow_html=True)
        st.subheader(f"Will we hit target? — {PLABEL[product]}")
        fc = forecast_month_end(df_all, product, targets, cur_e)
        net_mtd = fc["mtd"].sum()
        net_proj = fc["projected"].sum(skipna=True)
        net_tgt = targets["monthly_target"].sum(skipna=True)
        net_attain = net_proj / net_tgt * 100 if net_tgt else np.nan
        elapsed = int(fc["elapsed"].iloc[0]) if len(fc) else 0
        dim = int(fc["days_in_month"].iloc[0]) if len(fc) else 0
        kpi_row([
            ("Month-to-date", f0(net_mtd), "L", f"{elapsed} of {dim} days elapsed", accent),
            ("Projected month-end", f0(net_proj), "L", "at current run-rate", accent),
            ("Monthly target", f0(net_tgt), "L", "2× median month", accent),
            ("Projected attainment", "—" if np.isnan(net_attain) else f"{net_attain:,.0f}", "%",
             "on track" if (not np.isnan(net_attain) and net_attain >= 100) else "short of target",
             accent),
        ])
        st.write("")
        fp = fc.dropna(subset=["projected"]).sort_values("proj_attain")
        if not fp.empty:
            fig = go.Figure()
            fig.add_bar(y=fp["station"], x=fp["monthly_target"], orientation="h", name="Target",
                        marker_color="rgba(140,140,140,.30)")
            fig.add_bar(y=fp["station"], x=fp["projected"], orientation="h", name="Projected",
                        marker_color=["#1F9D57" if w else "#E23744"
                                      for w in fp["will_hit"].fillna(False)],
                        text=["—" if np.isnan(a) else f"{a:.0f}%" for a in fp["proj_attain"]],
                        textposition="outside", textfont=dict(color=INK, size=11), cliponaxis=False)
            fig.update_layout(barmode="group",
                              title="Projected month-end vs target (green = on track to hit)")
            st.plotly_chart(style_fig(fig, max(300, 34 * len(fp)), accent), use_container_width=True)
        show = fc.rename(columns={
            "station": "Station", "mtd": "MTD (L)", "daily_rate": "Daily rate (L)",
            "projected": "Projected month-end (L)", "monthly_target": "Target (L)",
            "proj_attain": "Proj. attainment %", "shortfall": "Proj. gap (L)"})
        cols = ["Station", "MTD (L)", "Daily rate (L)", "Projected month-end (L)",
                "Target (L)", "Proj. attainment %", "Proj. gap (L)"]
        st.dataframe(show[cols].style.format({
            "MTD (L)": "{:,.0f}", "Daily rate (L)": "{:,.0f}", "Projected month-end (L)": "{:,.0f}",
            "Target (L)": "{:,.0f}", "Proj. attainment %": "{:,.0f}%", "Proj. gap (L)": "{:,.0f}"},
            na_rep="—"), use_container_width=True, hide_index=True)

        st.markdown("<div class='eyebrow'>30-day demand outlook</div>", unsafe_allow_html=True)
        st.caption("Trend + day-of-week seasonality with an 80% confidence band. "
                   + ("Network total." if focus == "All stations" else f"{focus}."))
        if focus == "All stations":
            ser = (df_all[df_all["product"] == product].groupby("date", as_index=False)
                   .agg(volume=("volume", "sum")))
        else:
            ser = df_all[(df_all["product"] == product) & (df_all["station"] == focus)][
                ["date", "volume"]]
        res = forecast_series(ser)
        if res is None:
            st.caption("Not enough history for a forecast here.")
        else:
            hist, fcast = res
            fig = go.Figure()
            fig.add_scatter(x=fcast["date"], y=fcast["hi"], line=dict(width=0),
                            showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=fcast["date"], y=fcast["lo"], fill="tonexty", fillcolor=PSTEP[product],
                            line=dict(width=0), name="80% band")
            fig.add_scatter(x=hist["date"], y=hist["volume"], mode="lines", name="History",
                            line=dict(color="rgba(140,140,140,.75)", width=1.5))
            fig.add_scatter(x=fcast["date"], y=fcast["yhat"], mode="lines", name="Forecast",
                            line=dict(color=accent, width=2.6))
            fig.update_layout(title="Daily volume — history & 30-day forecast")
            fig.update_yaxes(title_text="L/day")
            st.plotly_chart(style_fig(fig, 340, accent), use_container_width=True)

    # ============================ ALERTS ============================
    with tabs[8]:
        st.markdown("<div class='eyebrow'>Exceptions across the network</div>",
                    unsafe_allow_html=True)
        st.subheader("Alerts")
        SEV = {1: "🔴 High", 2: "🟠 Medium", 3: "🟡 Low"}
        alerts = []
        for rp in real_products:
            for _, r in runway_all[runway_all["product"] == rp].iterrows():
                if r["risk"] == "critical":
                    alerts.append((1, "Stock-out", rp, r["station"],
                                   f"~{r['days_to_run_out']:.1f} days of cover — refill now"))
                elif r["risk"] == "low":
                    alerts.append((2, "Stock-out", rp, r["station"],
                                   f"~{r['days_to_run_out']:.1f} days of cover"))
            tg = compute_targets(df_all, rp, base_s, base_e, cur_s, cur_e)
            for _, r in forecast_month_end(df_all, rp, tg, cur_e).iterrows():
                pa = r["proj_attain"]
                if not np.isnan(pa) and pa < 70:
                    alerts.append((2, "Off target", rp, r["station"],
                                   f"projected {pa:.0f}% of monthly target"))
                elif not np.isnan(pa) and pa < 90:
                    alerts.append((3, "Off target", rp, r["station"],
                                   f"projected {pa:.0f}% of monthly target"))
            for _, r in compute_variance(df, rp, pd.DataFrame(), cur_s, cur_e, STANDARD[rp]).iterrows():
                if r["within_standard"] == False:  # noqa: E712
                    alerts.append((2, "Dip variance", rp, r["station"],
                                   f"avg {r['avg_daily_var']:+.1f} L/day exceeds ±{STANDARD[rp]:.0f} L/day"))
            for _, r in volume_anomalies(df, rp).iterrows():
                d = "spike" if r["z"] > 0 else "drop"
                alerts.append((2, "Anomaly", rp, r["station"],
                               f"{pd.Timestamp(r['date']).strftime('%d %b')} volume {d} ({r['z']:+.1f}σ)"))
        cb = compute_banking(banking_frame(df))
        if not cb.empty:
            thr = max(cb["outstanding"].quantile(0.75), 1)
            for _, r in cb.iterrows():
                o = r["outstanding"]
                if not np.isnan(o) and o > 0 and o >= thr:
                    sev = 1 if (np.isnan(r["banking_rate"]) or r["banking_rate"] < 50) else 2
                    alerts.append((sev, "Unbanked cash", "—", r["station"],
                                   f"GHS {o:,.0f} unbanked"))
        if not alerts:
            st.success("No active alerts — everything is within thresholds. ✅")
        else:
            adf = pd.DataFrame(alerts, columns=["sev", "Category", "Product", "Station", "Detail"])
            adf = adf.sort_values("sev").reset_index(drop=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 High", int((adf.sev == 1).sum()))
            m2.metric("🟠 Medium", int((adf.sev == 2).sum()))
            m3.metric("🟡 Low", int((adf.sev == 3).sum()))
            adf["Severity"] = adf["sev"].map(SEV)
            st.dataframe(adf[["Severity", "Category", "Product", "Station", "Detail"]],
                         use_container_width=True, hide_index=True,
                         height=min(600, 80 + 34 * len(adf)))
        st.markdown("<div class='eyebrow'>Stakeholder report</div>", unsafe_allow_html=True)
        try:
            sheets = {"Targets": targets, "Forecast": fc,
                      "Runway": runway_all, "Rankings": rankings,
                      "Banking": cb}
            st.download_button("⬇ Download executive report (Excel)", build_excel(sheets),
                               "spartan_executive_report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.caption(f"Report export unavailable: {e}")

    # ============================ TRENDS ============================
    with tabs[9]:
        st.markdown("<div class='eyebrow'>Full history</div>", unsafe_allow_html=True)
        st.subheader(f"Trends — {PLABEL[product]} · {focus}")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        any_data = False
        if product == "BOTH":
            for rp in ["PMS", "AGO"]:
                if focus == "All stations":
                    s = (df[df["product"] == rp].groupby("date", as_index=False)
                         .agg(volume=("volume", "sum")))
                else:
                    s = df[(df["product"] == rp) & (df["station"] == focus)][["date", "volume"]]
                s = s.sort_values("date")
                if not s.empty:
                    any_data = True
                    fig.add_scatter(x=s["date"], y=s["volume"], name=rp,
                                    line=dict(color=PCOL[rp], width=2), secondary_y=False)
            fig.update_layout(title="Daily volume by product")
        else:
            if focus == "All stations":
                s = (df[df["product"] == product].groupby("date", as_index=False)
                     .agg(volume=("volume", "sum")))
                s["price"] = np.nan
            else:
                s = df[(df["product"] == product) & (df["station"] == focus)][
                    ["date", "volume", "price"]]
            s = s.sort_values("date")
            if not s.empty:
                any_data = True
                s["ma"] = s["volume"].rolling(7, min_periods=1).mean()
                fig.add_bar(x=s["date"], y=s["volume"], name="Daily volume",
                            marker_color=PSTEP[product], secondary_y=False)
                fig.add_scatter(x=s["date"], y=s["ma"], name="7-day average",
                                line=dict(color=accent, width=2.5), secondary_y=False)
                if focus != "All stations" and s["price"].notna().any():
                    fig.add_scatter(x=s["date"], y=s["price"], name="Price",
                                    line=dict(color=INK, width=1.5, shape="hv"), secondary_y=True)
                    fig.update_yaxes(title_text="Price (GHS/L)", secondary_y=True)
                fig.update_layout(title="Daily volume & price")
        if any_data:
            fig.add_vrect(x0=cur_s, x1=cur_e, fillcolor="rgba(140,140,140,.10)", line_width=0,
                          annotation_text="current", annotation_position="top left")
            fig.update_yaxes(title_text="Volume (L/day)", secondary_y=False)
            st.plotly_chart(style_fig(fig, 360, accent), use_container_width=True)
            st.markdown("<div class='eyebrow'>Weekly volume heatmap</div>", unsafe_allow_html=True)
            hm = df_all[df_all["product"] == product].copy()
            hm["week"] = hm["date"].dt.to_period("W").dt.start_time
            piv = hm.pivot_table(index="station", columns="week", values="volume", aggfunc="sum")
            if not piv.empty:
                fig = go.Figure(go.Heatmap(
                    z=piv.values, x=[d.strftime("%d %b") for d in piv.columns],
                    y=list(piv.index), colorscale=SCALE[product], colorbar=dict(title="L/wk")))
                st.plotly_chart(style_fig(fig, max(260, 30 * len(piv)), accent),
                                use_container_width=True)
        else:
            st.info("No data for this selection.")

    st.caption("All figures from the MASTER sheet · prices in GHS · target = 2× median "
               "baseline month, measured against the current period.")


if __name__ == "__main__":
    main()