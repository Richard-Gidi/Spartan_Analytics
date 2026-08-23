"""
Spartan OMC — Downstream Petroleum Analytics
============================================
An operating system for a Ghanaian Oil Marketing Company, built on the daily
station returns already kept in the MASTER tab of a Google Sheet.

Modules
-------
  Stocks & Sales   throughput, targets, price sensitivity, runway, efficiency,
                   dip variance, rankings, forecast, alerts, trends, cost of
                   losses, and the monthly stakeholder PDF
  Margins          revenue at pump price less ex-depot cost, dealer margin,
                   per-litre opex and the cost of missing litres; break-even
                   volume and working capital tied up in wet stock
  Pricing          NPA pricing windows (1-15, 16-EOM), price-floor compliance,
                   network price spread, price/volume response, and the gain or
                   loss on stock held when the depot price moves
  Supply           ullage, reorder points, a compartment-level load plan, and
                   transit shortage between depot and tank
  Control          wetstock control charts against a throughput tolerance,
                   daily-return discipline, integrity flags, statutory volume
                   returns
  Site card        one site, one page, for a dealer review
  Banking          cash generated vs deposited (unchanged)
  Settings         the commercial model - ex-depot cost, dealer margin, opex,
                   UPPF, price floors, tank capacities, haulage

PMS = petrol (red), AGO = diesel (green), Both = combined throughput.
Light and dark theme, phone-friendly.

Setup
-----
  .env  ->  GOOGLE=https://docs.google.com/spreadsheets/d/<id>/edit?usp=sharing
            (Share -> Anyone with the link - Viewer)
            OMC_CONFIG=omc_config.json          (optional, this is the default)
  pip install -r requirements.txt
  streamlit run fuel_analytics_app.py

Every money figure depends on the commercial model in Settings. Nothing is
assumed: until the ex-depot cost table is filled in, margin columns stay blank
rather than showing a number that was invented.

Target  = 2 x median of the baseline MONTHLY totals, measured against the actual
          total sold in the current period (gauge = % of target obtained).
Only the MASTER tab is used.
"""

import io
import os
import re
from pathlib import Path
import math
import base64
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


# ─────────────────── reporting engine (PDF + WhatsApp) ─────────────────────
COLORS = {"PMS": "#E23744", "AGO": "#1F9D57"}
GLABEL = {"PMS": "PMS \u00b7 Petrol", "AGO": "AGO \u00b7 Diesel"}
DOT = {"PMS": "\U0001F534", "AGO": "\U0001F7E2"}
SPARTAN_LOGO_B64 = "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"


def logo_bytes():
    try:
        return base64.b64decode(SPARTAN_LOGO_B64)
    except Exception:
        return None


def _logo_rgba():
    try:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(logo_bytes())).convert("RGBA"))
    except Exception:
        return None


def _resolve_font():
    """Prefer Candara (the user's brand font); fall back cleanly if it isn't installed."""
    try:
        import os
        from matplotlib import font_manager as fm
        names = {f.name for f in fm.fontManager.ttflist}
        if "Candara" not in names:
            for p in (r"C:\Windows\Fonts\Candara.ttf", r"C:\Windows\Fonts\candara.ttf",
                      r"C:\Windows\Fonts\Candarab.ttf", r"C:\Windows\Fonts\candarab.ttf"):
                if os.path.exists(p):
                    try:
                        fm.fontManager.addfont(p)
                    except Exception:
                        pass
            names = {f.name for f in fm.fontManager.ttflist}
        for fam in ("Candara", "Calibri", "Carlito", "Corbel", "Gill Sans", "DejaVu Sans"):
            if fam in names:
                return fam
    except Exception:
        pass
    return "DejaVu Sans"


# ─────────────────────────── pillar ranking (per grade) ────────────────────
def build_pillars(targets, variance_df, efficiency_df):
    tgt = targets[["station", "actual_total", "monthly_target", "attainment_pct",
                   "gap_litres"]].copy()
    tgt = tgt.sort_values("attainment_pct", ascending=False,
                          na_position="last").reset_index(drop=True)
    tgt.insert(0, "rank", range(1, len(tgt) + 1))

    vr = variance_df[["station", "avg_daily_var", "dip_variance", "within_standard",
                      "days"]].copy()
    vr["abs_daily"] = vr["avg_daily_var"].abs()
    vr = vr.sort_values("abs_daily", ascending=True,
                        na_position="last").reset_index(drop=True)
    vr.insert(0, "rank", range(1, len(vr) + 1))

    er = efficiency_df[["station", "days_to_stockout", "turnover_per_day",
                        "avg_daily_sales"]].copy()
    er = er.sort_values("days_to_stockout", ascending=True,
                        na_position="last").reset_index(drop=True)
    er.insert(0, "rank", range(1, len(er) + 1))
    return tgt, vr, er


def _verdict(att):
    if att is None or np.isnan(att):
        return "no target basis"
    return "ahead of plan" if att >= 100 else ("on track" if att >= 75 else "behind plan")


def report_kpis(tgt_rank, var_rank, eff_rank, std_lpd):
    tot_a = float(np.nansum(tgt_rank["actual_total"].values)) if len(tgt_rank) else 0.0
    tot_t = float(np.nansum(tgt_rank["monthly_target"].values)) if len(tgt_rank) else 0.0
    att = (tot_a / tot_t * 100) if tot_t > 0 else np.nan
    net_var = float(np.nansum(var_rank["dip_variance"].values)) if len(var_rank) else 0.0
    lost = float(-var_rank["dip_variance"].clip(upper=0).sum()) if len(var_rank) else 0.0
    flagged = int((var_rank["within_standard"] == False).sum()) if len(var_rank) else 0  # noqa:E712

    def _lead(frame, col, ascending):
        f = frame.dropna(subset=[col])
        if f.empty:
            return None, np.nan
        r = f.sort_values(col, ascending=ascending).iloc[0]
        return r["station"], float(r[col])

    tgt_top, tgt_top_v = _lead(tgt_rank, "attainment_pct", False)
    tgt_bot, tgt_bot_v = _lead(tgt_rank, "attainment_pct", True)
    eff_top, eff_top_v = _lead(eff_rank, "days_to_stockout", True)
    if var_rank["avg_daily_var"].notna().any():
        vr0 = var_rank.dropna(subset=["avg_daily_var"]).iloc[0]
        var_top, var_top_v = vr0["station"], float(vr0["avg_daily_var"])
    else:
        var_top, var_top_v = None, np.nan

    return {"actual": tot_a, "target": tot_t, "attainment": att, "verdict": _verdict(att),
            "net_var": net_var, "litres_lost": lost, "flagged": flagged, "std_lpd": std_lpd,
            "tgt_top": tgt_top, "tgt_top_v": tgt_top_v, "tgt_bot": tgt_bot, "tgt_bot_v": tgt_bot_v,
            "eff_top": eff_top, "eff_top_v": eff_top_v, "var_top": var_top, "var_top_v": var_top_v,
            "stations": int(len(tgt_rank))}


def _f0(x):
    return "\u2014" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"


def _pct(x):
    return "\u2014" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0f}%"


def _medal(i):
    return {0: "\U0001F947", 1: "\U0001F948", 2: "\U0001F949"}.get(i, f"{i + 1}.")


# ────────────────────────────── WhatsApp text ──────────────────────────────
def build_whatsapp_text(meta, per_product, top_n=3):
    """A plain-language brief anyone can read on WhatsApp: a headline, a short key
    that explains each figure, then a concise section per grade."""
    kP, kA = per_product["PMS"]["kpis"], per_product["AGO"]["kpis"]
    tot_a = kP["actual"] + kA["actual"]
    tot_t = kP["target"] + kA["target"]
    std = int(per_product["PMS"]["std"])

    # friendly target phrase (strip the leading "Target = " from the note)
    tn = meta.get("target_note", "twice the median of the earlier months this year")
    for pref in ("Target = ", "Target= ", "Target "):
        if tn.startswith(pref):
            tn = tn[len(pref):]
            break

    L = ["*SPARTAN FUEL \u2014 MONTHLY REPORT*",
         f"_{meta['period']} \u00b7 whole network (PMS + AGO)_", ""]

    # ---- headline ----
    L.append("\U0001F4CA *THE HEADLINE*")
    if tot_t > 0:
        L.append(f"This month we sold *{_f0(tot_a)} L* in total \u2014 that is *{tot_a / tot_t * 100:.0f}%* "
                 f"of our {_f0(tot_t)} L target.")
        L.append(f"{DOT['PMS']} Petrol (PMS): {_pct(kP['attainment'])} of target    "
                 f"{DOT['AGO']} Diesel (AGO): {_pct(kA['attainment'])} of target")
    else:
        L.append(f"This month we sold *{_f0(tot_a)} L* in total.")
        L.append("No target was set for this month \u2014 it is the first month of the year, so there "
                 "are no earlier months yet to base a target on.")
    L.append("")

    # ---- key: what each number means ----
    L.append("\u2139\uFE0F *WHAT THE NUMBERS MEAN*")
    L.append(f"\u2022 *Target* \u2014 {tn}")
    L.append("\u2022 *% of target* \u2014 how much a station sold compared with its target "
             "(100% means it hit the goal, above 100% means it beat it).")
    L.append(f"\u2022 *Stock control (variance)* \u2014 the daily gap between the fuel actually in the "
             f"tank and what the dip readings say should be there. We allow up to \u00b1{std} litres a "
             "day per station; smaller is better, and a shortage is fuel we can't account for.")
    L.append("\u2022 *Sell-through* \u2014 how many days of stock a station has left at its current "
             "selling pace. Fewer days means it is moving fuel faster.")
    L.append("")

    # ---- per grade ----
    names = {"PMS": "PETROL (PMS)", "AGO": "DIESEL (AGO)"}
    for g in ("PMS", "AGO"):
        b = per_product[g]
        k, tgt = b["kpis"], b["tgt"]
        L.append(f"{DOT[g]} *{names[g]}*")
        if not np.isnan(k["attainment"]):
            L.append(f"Sold {_f0(k['actual'])} L against a {_f0(k['target'])} L target "
                     f"\u2014 *{_pct(k['attainment'])}* of target.")
        else:
            L.append(f"Sold {_f0(k['actual'])} L (no target set this month).")
        top = tgt.dropna(subset=["attainment_pct"]).head(top_n)
        if not top.empty:
            best = ", ".join(f"{r['station']} ({r['attainment_pct']:.0f}%)" for _, r in top.iterrows())
            L.append(f"\U0001F3C6 Best performers: {best}.")
        if k["tgt_bot"] and k["tgt_bot"] != k["tgt_top"] and not (
                isinstance(k["tgt_bot_v"], float) and np.isnan(k["tgt_bot_v"])):
            L.append(f"\u26A0 Needs attention: {k['tgt_bot']} at {_pct(k['tgt_bot_v'])} of target.")
        if k["flagged"] > 0:
            L.append(f"Stock control: {k['flagged']} station(s) went outside the \u00b1{std} L/day "
                     f"limit \u2014 about {_f0(k['litres_lost'])} L unaccounted for. Tightest control: "
                     f"{k['var_top']}.")
        else:
            L.append(f"Stock control: every station stayed within the \u00b1{std} L/day limit \u2705.")
        if k["eff_top"]:
            ev = "" if np.isnan(k["eff_top_v"]) else f" (about {k['eff_top_v']:.1f} days of stock)"
            L.append(f"Selling fastest: {k['eff_top']}{ev}.")
        L.append("")

    L.append("\U0001F4CE The attached PDF has the full ranked charts and every station's figures.")
    L.append(f"_Generated {meta['generated']}_")
    return "\n".join(L)


# ───────────────────────────────── PDF ─────────────────────────────────────
def build_report_pdf(meta, per_product):
    import textwrap as _tw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
    from matplotlib.backends.backend_pdf import PdfPages

    FONT = _resolve_font()
    INK, SUB, LINE, DARK = "#1d2330", "#6b7280", "#e3e6ea", "#0C1014"
    SOFT = "#fbfbfc"
    A4 = (8.27, 11.69)
    plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "axes.edgecolor": LINE,
                         "text.color": INK, "axes.labelcolor": INK})
    LOGO = _logo_rgba()
    laspect = (LOGO.shape[0] / LOGO.shape[1]) if LOGO is not None else 1.15

    def new_page():
        fig = plt.figure(figsize=A4, dpi=150)
        bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
        return fig, bg

    def watermark(bg):
        if LOGO is None:
            return
        bw = 0.62
        bh = bw * (A4[0] / A4[1]) * laspect
        cx, cy = 0.5, 0.47
        bg.imshow(LOGO, extent=[cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2],
                  origin="upper", alpha=0.06, zorder=0, aspect="auto", interpolation="bilinear")

    def header_logo(fig):
        if LOGO is None:
            return
        lw = 0.085
        lh = lw * (A4[0] / A4[1]) * laspect
        ax = fig.add_axes([0.885, 0.905 + (0.095 - lh) / 2, lw, lh]); ax.axis("off")
        ax.imshow(LOGO, origin="upper", interpolation="bilinear")

    def band(fig, bg, title, sub):
        bg.add_patch(Rectangle((0, 0.9), 1, 0.1, color=DARK, zorder=1))
        bg.add_patch(Rectangle((0, 0.897), 1, 0.006, color=COLORS["PMS"], zorder=2))
        bg.text(0.06, 0.957, title, color="white", fontsize=15, fontweight="bold",
                va="center", zorder=3)
        bg.text(0.06, 0.922, sub, color="#aab2bc", fontsize=8.4, va="center", zorder=3)
        header_logo(fig)

    def footer(bg, page_txt):
        bg.add_patch(Rectangle((0, 0), 1, 0.032, color="#f4f5f7", zorder=5))
        bg.text(0.06, 0.016, meta["footer"], color=SUB, fontsize=6.8, va="center", zorder=6)
        bg.text(0.94, 0.016, page_txt, color=SUB, fontsize=6.8, va="center", ha="right", zorder=6)

    def legend(bg, y, x=0.06):
        bg.add_patch(Circle((x + 0.005, y), 0.006, color=COLORS["PMS"], zorder=6))
        bg.text(x + 0.02, y, "PMS \u00b7 Petrol", fontsize=8.4, color=INK, va="center", zorder=6)
        bg.add_patch(Circle((x + 0.175, y), 0.006, color=COLORS["AGO"], zorder=6))
        bg.text(x + 0.19, y, "AGO \u00b7 Diesel", fontsize=8.4, color=INK, va="center", zorder=6)

    def heading(bg, y, text):
        bg.text(0.06, y, text, fontsize=8.2, color=COLORS["PMS"], fontweight="bold", zorder=6)
        bg.add_patch(Rectangle((0.06, y - 0.007), 0.03, 0.003, color=COLORS["PMS"], zorder=6))

    def panel(fig, bg, rect, subtitle, d, valcol, color, mode, std=10.0):
        bg.text(rect[0] - 0.26, rect[1] + rect[3] + 0.012, subtitle, fontsize=9.6,
                color=INK, fontweight="bold", va="center", zorder=6)
        ax = fig.add_axes(rect); ax.set_facecolor("none")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
        ax.tick_params(colors=SUB, labelsize=8, length=0)
        ax.grid(axis="x", color="#eef0f2", zorder=1); ax.set_axisbelow(True)
        dd = d.dropna(subset=[valcol]); n = len(dd)
        if n == 0:
            ax.text(0.5, 0.5, "No data for this grade / month.", ha="center", va="center",
                    transform=ax.transAxes, color=SUB, fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([]); return
        vals = dd[valcol].values[::-1]; stns = list(dd["station"])
        labels = [f"{i + 1}. {s}" for i, s in enumerate(stns)][::-1]
        ax.barh(range(n), vals, color=color, height=0.62, zorder=3)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontweight="bold", color=INK)   # bold station names
        if mode == "target":
            ax.axvline(100, color=INK, lw=1.0, ls="--", zorder=4)
            top = max(vals.max(), 100)
            for i, v in enumerate(vals):
                ax.text(v + top * 0.012, i, f"{v:.0f}%", va="center", fontsize=7.4,
                        color=INK, fontweight="bold")
            ax.set_xlim(0, top * 1.18); ax.set_xlabel("Attainment %  (dashed = 100% target)",
                                                      fontsize=8)
        elif mode == "variance":
            ax.axvline(0, color=INK, lw=1.0, zorder=4)
            for s in (std, -std):
                ax.axvline(s, color=SUB, lw=1.0, ls="--", zorder=4)
            span = max(abs(vals.min()), abs(vals.max()), std) * 1.32
            for i, v in enumerate(vals):
                ax.text(v + (span * 0.02 if v >= 0 else -span * 0.02), i, f"{v:+.1f}",
                        va="center", ha="left" if v >= 0 else "right", fontsize=7.2,
                        color=INK, fontweight="bold")
            ax.set_xlim(-span, span)
            ax.set_xlabel(f"Avg dip variance L/day  (dashed = \u00b1{std:.0f})", fontsize=8)
        else:
            top = max(vals.max(), 1)
            for i, v in enumerate(vals):
                ax.text(v + top * 0.012, i, f"{v:.1f} d", va="center", fontsize=7.4,
                        color=INK, fontweight="bold")
            ax.set_xlim(0, top * 1.18); ax.set_xlabel("Days to stock out  (shorter = faster)",
                                                      fontsize=8)

    pages = PdfPages(meta["_buffer"])
    kP, kA = per_product["PMS"]["kpis"], per_product["AGO"]["kpis"]

    # ---------------- PAGE 1 — cover ----------------
    fig, bg = new_page()
    watermark(bg)
    band(fig, bg, "SPARTAN FUEL \u2014 PERFORMANCE REPORT", meta["cover_sub"])

    tot_a, tot_t = kP["actual"] + kA["actual"], kP["target"] + kA["target"]
    att = (tot_a / tot_t * 100) if tot_t > 0 else np.nan
    net_v, flg = kP["net_var"] + kA["net_var"], kP["flagged"] + kA["flagged"]
    cards = [("TOTAL SOLD", f"{_f0(tot_a)} L", f"of {_f0(tot_t)} L target"),
             ("ATTAINMENT", _pct(att), _verdict(att)),
             ("NET VARIANCE", f"{net_v:+,.0f} L", f"{flg} grade-station(s) flagged"),
             ("STATIONS", f"{max(kP['stations'], kA['stations'])}", "PMS + AGO, both shown")]
    x0, w, gap, y, h = 0.06, 0.205, 0.0133, 0.775, 0.09
    for i, (lab, val, sub) in enumerate(cards):
        x = x0 + i * (w + gap)
        bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
                     linewidth=1, edgecolor=LINE, facecolor=SOFT, zorder=2))
        bg.add_patch(Rectangle((x, y), w, 0.004, color=COLORS["PMS"], zorder=3))
        bg.text(x + 0.014, y + h - 0.020, lab, fontsize=6.3, color=SUB, va="center", zorder=3)
        s = str(val); fs = 15 if len(s) <= 9 else (11.5 if len(s) <= 12 else 9.5)
        bg.text(x + 0.014, y + 0.046, s, fontsize=fs, color=INK, fontweight="bold",
                va="center", zorder=3)
        bg.text(x + 0.014, y + 0.016, sub, fontsize=6.2, color=SUB, va="center", zorder=3)

    legend(bg, 0.742)

    heading(bg, 0.706, "SUMMARY")
    bg.text(0.06, 0.688, meta["headline"], fontsize=10.5, color=INK, fontweight="bold",
            va="top", zorder=6)
    bg.text(0.06, 0.66, _tw.fill(meta["summary"], width=108), fontsize=8.8, color="#33383f",
            va="top", linespacing=1.5, zorder=6)

    # how targets are set (boxed note)
    ty = 0.556
    bg.add_patch(FancyBboxPatch((0.06, ty - 0.036), 0.88, 0.05,
                 boxstyle="round,pad=0.004,rounding_size=0.008",
                 linewidth=1, edgecolor="#f0d9dc", facecolor="#fdf3f4", zorder=2))
    bg.add_patch(Rectangle((0.06, ty - 0.036), 0.006, 0.05, color=COLORS["PMS"], zorder=3))
    bg.text(0.085, ty + 0.001, "HOW TARGETS ARE SET", fontsize=7.6, color=COLORS["PMS"],
            fontweight="bold", va="center", zorder=4)
    bg.text(0.085, ty - 0.022, meta["target_note"], fontsize=8.4, color="#33383f",
            va="center", zorder=4)

    heading(bg, 0.48, "BY GRADE")
    yy = 0.446
    for g in ("PMS", "AGO"):
        k = per_product[g]["kpis"]; c = COLORS[g]
        bg.add_patch(FancyBboxPatch((0.06, yy - 0.092), 0.88, 0.10,
                     boxstyle="round,pad=0.004,rounding_size=0.008",
                     linewidth=1, edgecolor=LINE, facecolor=SOFT, zorder=2))
        bg.add_patch(Rectangle((0.06, yy - 0.092), 0.006, 0.10, color=c, zorder=3))
        bg.add_patch(Circle((0.092, yy - 0.004), 0.006, color=c, zorder=4))
        bg.text(0.11, yy - 0.004, GLABEL[g], fontsize=10.5, color=INK, fontweight="bold",
                va="center", zorder=4)
        bg.text(0.5, yy - 0.004, f"Sold {_f0(k['actual'])} L  /  target {_f0(k['target'])} L  "
                f"\u00b7  {_pct(k['attainment'])} ({k['verdict']})", fontsize=8.6, color=SUB,
                va="center", zorder=4)
        bg.text(0.11, yy - 0.04, f"Top: {k['tgt_top']} ({_pct(k['tgt_top_v'])})   \u00b7   "
                f"Watch: {k['tgt_bot']} ({_pct(k['tgt_bot_v'])})", fontsize=8.2, color="#33383f",
                va="center", zorder=4)
        vtxt = (f"all within \u00b1{k['std_lpd']:.0f} L/day" if k["flagged"] == 0
                else f"{k['flagged']} outside \u00b1{k['std_lpd']:.0f} L/day (~{_f0(k['litres_lost'])} L lost)")
        etxt = (k["eff_top"] or "\u2014") + ("" if np.isnan(k["eff_top_v"])
                                             else f" ({k['eff_top_v']:.1f} d)")
        bg.text(0.11, yy - 0.066, f"Variance: {vtxt}   \u00b7   Fastest sell-through: {etxt}",
                fontsize=8.2, color="#33383f", va="center", zorder=4)
        yy -= 0.116

    footer(bg, "Page 1  \u00b7  Summary")
    pages.savefig(fig); plt.close(fig)

    # ---------------- PAGES 2-4 — pillars ----------------
    pillars = [
        ("1  \u00b7  TARGET vs ACTUAL REALIZED",
         "How much each station sold this month against its target. Longer bar = higher "
         "attainment; the dashed line is 100% (target met).",
         "attainment_pct", "target", "tgt", "Page 2  \u00b7  Target vs Actual"),
        ("2  \u00b7  STOCK VARIANCE",
         "Tightness of stock control \u2014 average dip variance per day. Bars nearer zero are "
         "better; dashed lines mark the \u00b1 allowed standard. Bars past them are flagged.",
         "avg_daily_var", "variance", "var", "Page 3  \u00b7  Variance"),
        ("3  \u00b7  EFFICIENCY",
         "How fast each station sells through its stock \u2014 average days to stock out. "
         "Shorter bar = faster turnover.",
         "days_to_stockout", "efficiency", "eff", "Page 4  \u00b7  Efficiency"),
    ]
    for title, blurb, valcol, mode, key, pagetxt in pillars:
        fig, bg = new_page(); watermark(bg)
        band(fig, bg, title, meta["cover_sub"])
        bg.text(0.06, 0.867, _tw.fill(blurb, width=112), fontsize=8.3, color=SUB, va="top",
                linespacing=1.4, zorder=6)
        legend(bg, 0.822)
        panel(fig, bg, [0.31, 0.475, 0.62, 0.295], GLABEL["PMS"],
              per_product["PMS"][key], valcol, COLORS["PMS"], mode, per_product["PMS"]["std"])
        panel(fig, bg, [0.31, 0.085, 0.62, 0.295], GLABEL["AGO"],
              per_product["AGO"][key], valcol, COLORS["AGO"], mode, per_product["AGO"]["std"])
        footer(bg, pagetxt)
        pages.savefig(fig); plt.close(fig)

    # ---------------- PAGE 5 — comprehensive data table ----------------
    fig, bg = new_page(); watermark(bg)
    band(fig, bg, "4  \u00b7  STATION DATA \u2014 ALL FIGURES", meta["cover_sub"])
    bg.text(0.06, 0.867, "Every station, both grades. Sold vs target with attainment, "
            "average daily stock variance, and days to stock out.", fontsize=8.3, color=SUB,
            va="top", zorder=6)

    def grade_table(rect, g):
        b = per_product[g]
        m = b["tgt"][["station", "actual_total", "monthly_target", "attainment_pct"]].copy()
        m = m.merge(b["var"][["station", "avg_daily_var"]], on="station", how="left")
        m = m.merge(b["eff"][["station", "days_to_stockout"]], on="station", how="left")
        m = m.sort_values("actual_total", ascending=False)
        rows = []
        for _, r in m.iterrows():
            rows.append([str(r["station"]), _f0(r["actual_total"]), _f0(r["monthly_target"]),
                         _pct(r["attainment_pct"]),
                         "\u2014" if pd.isna(r["avg_daily_var"]) else f"{r['avg_daily_var']:+.1f}",
                         "\u2014" if pd.isna(r["days_to_stockout"]) else f"{r['days_to_stockout']:.1f}"])
        cols = ["Station", "Sold (L)", "Target (L)", "Attain", "Var L/day", "Days to SO"]
        ax = fig.add_axes(rect); ax.axis("off")
        ax.text(0, 1.02, GLABEL[g], transform=ax.transAxes, fontsize=10.5, color=INK,
                fontweight="bold", va="bottom")
        if not rows:
            ax.text(0.5, 0.5, "No data for this grade / month.", ha="center", color=SUB); return
        tbl = ax.table(cellText=rows, colLabels=cols, loc="upper center",
                       cellLoc="right", colLoc="center",
                       colWidths=[0.28, 0.15, 0.15, 0.12, 0.15, 0.15])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.2); tbl.scale(1, 1.5)
        ncol = len(cols)
        for (rr, cc), cell in tbl.get_celld().items():
            cell.set_edgecolor("#eceef1")
            if rr == 0:
                cell.set_facecolor(COLORS[g]); cell.set_text_props(color="white",
                                                                   fontweight="bold")
                cell.set_height(cell.get_height() * 1.05)
            else:
                cell.set_facecolor("#ffffff" if rr % 2 else "#f7f8fa")
                if cc == 0:
                    cell.set_text_props(fontweight="bold", ha="left")
                    cell.PAD = 0.04
        return

    grade_table([0.08, 0.60, 0.86, 0.20], "PMS")
    grade_table([0.08, 0.30, 0.86, 0.20], "AGO")
    bg.text(0.06, 0.20, meta["target_note"], fontsize=8.2, color="#33383f", style="italic",
            va="center", zorder=6)
    bg.text(0.06, 0.175, "Var L/day = average daily dip variance (\u2212 = shortage, + = gain). "
            "Days to SO = days of stock left at the current sales pace.", fontsize=7.8,
            color=SUB, va="center", zorder=6)
    footer(bg, "Page 5  \u00b7  Station data")
    pages.savefig(fig); plt.close(fig)

    pages.close()
    return meta["_buffer"].getvalue()

# ═══════════════════════════════════════════════════════════════════════════
#                        OMC LAYER — configuration store
# ═══════════════════════════════════════════════════════════════════════════
# Everything commercial (what we pay the BDC, what we pay the dealer, NPA price
# floors, tank capacities, haulage) lives in one JSON file so the numbers can be
# updated every pricing window without touching code.
import json
import copy

# Where the config lives. A bare relative name resolves against the process
# working directory, which is NOT reliably the folder holding this script —
# on Streamlit Community Cloud it is the repo root, and if the app sits in a
# subfolder the file is silently missed. So we search a list of candidates and
# remember exactly what we tried, because "it didn't pick it up" with no
# explanation is the worst possible failure mode.
CONFIG_ENV = os.getenv("OMC_CONFIG")
CONFIG_PATH = CONFIG_ENV or "omc_config.json"
CONFIG_LOAD = {"path": None, "error": None, "tried": [], "source": "defaults"}


def config_candidates():
    here = Path(__file__).resolve().parent
    names = []
    if CONFIG_ENV:
        names.append(Path(CONFIG_ENV))
    for base in (here, here.parent, Path.cwd()):
        names.append(base / "omc_config.json")
        names.append(base / "config" / "omc_config.json")
    seen, out = set(), []
    for n in names:
        r = str(n)
        if r not in seen:
            seen.add(r)
            out.append(n)
    return out
TOLERANCE_PCT = 0.5          # wetstock control limit, % of cumulative throughput
WINDOW_DAY = 15              # Ghana pricing windows: 1–15 and 16–month end

DEFAULT_CONFIG = {
    "company": {
        "name": "Spartan Fuel",
        "npa_licence": "",
        "currency": "GHS",
        "supplier_bdc": "",
    },
    # Per-litre commercial model. exdepot is a dated table because it moves every
    # window; the margins are the fixed pesewas-per-litre legs of the build-up.
    "economics": {
        "PMS": {
            "exdepot": {},            # optional override, "YYYY-MM-DD" -> GHS/L all-in
            "exdepot_base": {},       # BDC selling price before duties, GHS/L
            "tax": {},                # duties + levies per litre, GHS/L
            "dealer_margin": 0.0,     # GHS/L paid out to the dealer / operator
            "opex_per_litre": 0.0,    # GHS/L haulage, marking, site overhead, shrink allowance
            "uppf_recovery": 0.0,     # GHS/L recovered on approved routes, 0 if not claimed
        },
        "AGO": {
            "exdepot": {},
            "exdepot_base": {},
            "tax": {},
            "dealer_margin": 0.0,
            "opex_per_litre": 0.0,
            "uppf_recovery": 0.0,
        },
    },
    # NPA minimum pump price by window-start date.
    "floors": {"PMS": {}, "AGO": {}},
    # Per-site register.
    "stations": {},
    "logistics": {
        "brv_sizes": [45000, 34000, 25000, 16000],
        "min_drop": 9000,
        "ullage_reserve_pct": 5.0,     # never plan to fill the last 5% of a tank
    },
    "control": {"tolerance_pct": TOLERANCE_PCT},
    # Optional: national monthly consumption for market-share, "YYYY-MM" -> litres.
    "market": {"national_monthly_litres": {}},
}

STATION_DEFAULTS = {
    "PMS_capacity": 0.0,
    "AGO_capacity": 0.0,
    "opex_month": 0.0,          # GHS fixed monthly site cost, for break-even
    "lead_time_days": 2.0,      # order to discharge
    "safety_days": 1.5,
    "region": "",
    "dealer": "",
}


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None):
    """Find and read the config. Never raises, but always records what happened
    in CONFIG_LOAD so the Settings screen can tell you why nothing appeared."""
    CONFIG_LOAD["tried"] = []
    CONFIG_LOAD["error"] = None
    CONFIG_LOAD["path"] = None
    CONFIG_LOAD["source"] = "defaults"

    candidates = [Path(path)] if path else config_candidates()
    for cand in candidates:
        CONFIG_LOAD["tried"].append(str(cand))
        try:
            if not cand.is_file():
                continue
            with open(cand, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as e:
            # A malformed file is a real error and must not be swallowed —
            # silently falling back to defaults is how a bad edit goes unnoticed.
            CONFIG_LOAD["error"] = f"{cand} is not valid JSON: {e}"
            continue
        except Exception as e:
            CONFIG_LOAD["error"] = f"couldn't read {cand}: {e}"
            continue
        CONFIG_LOAD["path"] = str(cand)
        CONFIG_LOAD["source"] = "file"
        return _deep_merge(DEFAULT_CONFIG, data)

    # Streamlit Community Cloud has no writable, persistent disk. Pasting the
    # whole config into Secrets as omc_config = '''{...}''' works there.
    try:
        import streamlit as _st
        blob = _st.secrets.get("omc_config") if hasattr(_st, "secrets") else None
        if blob:
            data = json.loads(blob) if isinstance(blob, str) else dict(blob)
            CONFIG_LOAD["path"] = "st.secrets['omc_config']"
            CONFIG_LOAD["source"] = "secrets"
            return _deep_merge(DEFAULT_CONFIG, data)
    except Exception as e:
        if CONFIG_LOAD["error"] is None:
            CONFIG_LOAD["error"] = f"secrets lookup failed: {e}"

    if CONFIG_LOAD["error"] is None:
        CONFIG_LOAD["error"] = "no omc_config.json found in any of the paths tried"
    return copy.deepcopy(DEFAULT_CONFIG)


def config_is_ephemeral():
    """True when we're somewhere the filesystem resets — Streamlit Community
    Cloud rebuilds the container on every reboot, so Save writes a file that
    will not survive. Better to say so than to let a save quietly evaporate."""
    return any(os.getenv(v) for v in ("STREAMLIT_SHARING_MODE", "STREAMLIT_SERVER_PORT")) \
        and not os.access(str(Path(__file__).resolve().parent), os.W_OK)


def save_config(cfg, path=None):
    target = Path(path) if path else Path(CONFIG_LOAD.get("path") or "")
    if not path and (not target or CONFIG_LOAD.get("source") != "file"):
        target = Path(__file__).resolve().parent / "omc_config.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    return str(target)


def config_bytes(cfg):
    return json.dumps(cfg, indent=2, sort_keys=True).encode("utf-8")


def station_cfg(cfg, station):
    return _deep_merge(STATION_DEFAULTS, (cfg.get("stations") or {}).get(station, {}))


def dated_rate(table, d, default=np.nan):
    """Step lookup: the value in force on date `d` is the latest entry on or
    before it. Returns `default` when nothing has been entered yet."""
    if not table:
        return default
    d = pd.Timestamp(d)
    best_k, best_v = None, default
    for k, v in table.items():
        try:
            kd = pd.Timestamp(k)
        except Exception:
            continue
        if kd <= d and (best_k is None or kd > best_k):
            best_k, best_v = kd, v
    try:
        return float(best_v)
    except (TypeError, ValueError):
        return default


def rate_table_frame(table):
    rows = []
    for k, v in sorted((table or {}).items()):
        try:
            rows.append({"effective_from": pd.Timestamp(k), "value": float(v)})
        except Exception:
            continue
    return pd.DataFrame(rows)


# ─────────────────────────── Ghana pricing windows ─────────────────────────
def window_bounds(d):
    """The NPA pricing window containing `d`: 1–15, or 16–month end."""
    d = pd.Timestamp(d)
    if d.day <= WINDOW_DAY:
        s = d.replace(day=1)
        e = d.replace(day=WINDOW_DAY)
    else:
        s = d.replace(day=WINDOW_DAY + 1)
        e = d + pd.offsets.MonthEnd(0)
    return pd.Timestamp(s.date()), pd.Timestamp(e.date())


def window_label(d):
    s, e = window_bounds(d)
    return f"{s:%d}–{e:%d %b %Y}"


def add_windows(frame):
    f = frame.copy()
    if f.empty:
        f["window_start"] = pd.NaT
        f["window"] = ""
        return f
    f["window_start"] = f["date"].map(lambda d: window_bounds(d)[0])
    f["window"] = f["date"].map(window_label)
    return f


def window_index(dmin, dmax):
    """All window starts spanned by the data, oldest first."""
    outs, cur = [], window_bounds(dmin)[0]
    while cur <= pd.Timestamp(dmax):
        outs.append(cur)
        cur = window_bounds(window_bounds(cur)[1] + pd.Timedelta(days=1))[0]
    return outs


# ───────────────────────────── margin engine ───────────────────────────────
def resolved_exdepot(cfg, product):
    """Ex-depot cost per litre = what the BDC charges, plus the duties and levies
    that ride on it. Keeping the two apart matters: the base price moves with
    every lifting, the tax moves when policy moves, and mixing them into one
    number means a levy change forces you to re-enter hundreds of prices.

    An explicit `exdepot` entry always wins, so a known invoice figure can
    override the computed one for any date.
    """
    ec = (cfg.get("economics") or {}).get(product, {}) or {}
    legacy = ec.get("exdepot") or {}
    base = ec.get("exdepot_base") or {}
    tax = ec.get("tax") or {}
    if not base and not tax:
        return legacy
    out = dict(legacy)
    for k in sorted(set(base) | set(tax)):
        if k in out:
            continue
        b = dated_rate(base, k)
        if np.isnan(b):
            continue
        t = dated_rate(tax, k, default=0.0)
        out[k] = round(float(b) + (0.0 if np.isnan(t) else float(t)), 4)
    return out


def margin_legs(cfg, product):
    ec = (cfg.get("economics") or {}).get(product, {}) or {}
    f = lambda k: float(ec.get(k) or 0.0)
    return (f("dealer_margin"), f("opex_per_litre"), f("uppf_recovery"),
            resolved_exdepot(cfg, product))


def priced_frame(df, product, cfg, start, end):
    """Daily rows for one grade with pump price carried forward, ex-depot cost
    attached from the dated table, and every per-litre leg applied."""
    s = _slice(df, product, start, end).dropna(subset=["volume"]).copy()
    if s.empty:
        return s
    dealer, opexl, uppf, exd_tbl = margin_legs(cfg, product)
    s = s.sort_values(["station", "date"])
    s["price"] = s.groupby("station")["price"].ffill().bfill()
    s["exdepot"] = s["date"].map(lambda d: dated_rate(exd_tbl, d))
    s["revenue"] = s["volume"] * s["price"]
    s["cogs"] = s["volume"] * s["exdepot"]
    s["gross"] = s["revenue"] - s["cogs"]
    s["dealer_cost"] = s["volume"] * dealer
    s["site_opex"] = s["volume"] * opexl
    s["uppf_income"] = s["volume"] * uppf
    s["net"] = s["gross"] - s["dealer_cost"] - s["site_opex"] + s["uppf_income"]
    return s


def compute_margins(df, product, cfg, start, end, cap=DELIVERY_CAP):
    """Per-station commercial P&L for one grade, with the cost of stock losses
    charged at ex-depot (the price we actually paid for the litres that vanished)."""
    s = priced_frame(df, product, cfg, start, end)
    if s.empty:
        return pd.DataFrame()
    dealer, opexl, uppf, exd_tbl = margin_legs(cfg, product)
    var = _slice(df, product, start, end)
    rows = []
    for st, g in s.groupby("station"):
        vol = float(g["volume"].sum())
        rev = float(g["revenue"].sum(min_count=1))
        cogs = float(g["cogs"].sum(min_count=1))
        gross = float(g["gross"].sum(min_count=1))
        dc = float(g["dealer_cost"].sum())
        ox = float(g["site_opex"].sum())
        up = float(g["uppf_income"].sum())
        net = float(g["net"].sum(min_count=1))
        vs = var[var["station"] == st]["dip_var"].dropna()
        vs = vs[vs.abs() <= cap]
        lost_l = max(-float(vs.sum()), 0.0) if len(vs) else 0.0
        unit_cost = (cogs / vol) if vol else np.nan
        loss_cost = lost_l * unit_cost if (vol and not np.isnan(unit_cost)) else 0.0
        cont = net - loss_cost
        scfg = station_cfg(cfg, st)
        days = max(int(g["date"].nunique()), 1)
        fixed = float(scfg["opex_month"] or 0) * days / 30.0
        npl = (cont / vol) if vol else np.nan
        rows.append({
            "station": st, "days": days, "volume": vol,
            "avg_price": (rev / vol) if vol else np.nan,
            "avg_cost": unit_cost, "revenue": rev, "cogs": cogs,
            "gross_margin": gross, "gross_per_litre": (gross / vol) if vol else np.nan,
            "dealer_cost": dc, "site_opex": ox, "uppf_income": up,
            "net_margin": net, "loss_litres": lost_l, "loss_cost": loss_cost,
            "contribution": cont, "net_per_litre": npl,
            "fixed_cost": fixed, "profit": cont - fixed,
            "breakeven_litres": (float(scfg["opex_month"]) / npl
                                 if (npl and npl > 0 and scfg["opex_month"]) else np.nan),
        })
    res = pd.DataFrame(rows)
    return res.sort_values("contribution", ascending=False).reset_index(drop=True) if len(res) else res


def margin_by_month(df, product, cfg):
    s = priced_frame(df, product, cfg, df["date"].min(), df["date"].max())
    if s.empty:
        return pd.DataFrame()
    s["month"] = s["date"].dt.to_period("M").dt.to_timestamp()
    g = s.groupby("month").agg(volume=("volume", "sum"), revenue=("revenue", "sum"),
                               cogs=("cogs", "sum"), gross=("gross", "sum"),
                               net=("net", "sum")).reset_index()
    g["gross_per_litre"] = g["gross"] / g["volume"].replace(0, np.nan)
    g["net_per_litre"] = g["net"] / g["volume"].replace(0, np.nan)
    return g


def margin_health(cfg, product):
    """Is the commercial model configured well enough to trust the money numbers?"""
    dealer, opexl, uppf, exd = margin_legs(cfg, product)
    return {"has_cost": bool(exd), "cost_points": len(exd or {}),
            "dealer": dealer, "opex": opexl, "uppf": uppf}


# ────────────────────────── pricing & floor control ────────────────────────
def compute_window_pricing(df, product, cfg, start, end):
    """One row per pricing window: what we charged, what the floor was, what we
    sold, and how volume responded to the price move."""
    s = _slice(df, product, start, end).dropna(subset=["volume"]).copy()
    if s.empty:
        return pd.DataFrame()
    s = add_windows(s)
    floors = (cfg.get("floors") or {}).get(product, {})
    dealer, opexl, uppf, exd_tbl = margin_legs(cfg, product)
    rows = []
    for ws, g in s.groupby("window_start"):
        px = g["price"].dropna()
        px = px[px > 0]
        rows.append({
            "window_start": ws, "window": window_label(ws),
            "days": int(g["date"].nunique()),
            "avg_price": float(px.mean()) if len(px) else np.nan,
            "min_price": float(px.min()) if len(px) else np.nan,
            "max_price": float(px.max()) if len(px) else np.nan,
            "floor": float(np.nanmean([dated_rate(floors, d)
                                       for d in g["date"].unique()]))
                     if floors else np.nan,
            "exdepot": dated_rate(exd_tbl, ws),
            "volume": float(g["volume"].sum()),
            "stations": int(g["station"].nunique()),
        })
    w = pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)
    w["daily_volume"] = w["volume"] / w["days"].replace(0, np.nan)
    w["price_chg"] = w["avg_price"].diff()
    w["price_chg_pct"] = w["avg_price"].pct_change() * 100
    w["vol_chg_pct"] = w["daily_volume"].pct_change() * 100
    w["headroom"] = w["avg_price"] - w["floor"]
    w["unit_margin"] = w["avg_price"] - w["exdepot"] - (dealer + opexl - uppf)
    w["window_margin"] = w["unit_margin"] * w["volume"]
    # only meaningful when the price actually moved; a 0.001% move divides into noise
    moved = w["price_chg_pct"].abs() >= 0.25
    with np.errstate(divide="ignore", invalid="ignore"):
        w["arc_elasticity"] = np.where(moved, w["vol_chg_pct"] / w["price_chg_pct"], np.nan)
    return w


def compute_floor_breaches(df, product, cfg, start, end, tol=0.005):
    """Days where a site sold below the NPA minimum. Selling under the floor is a
    licence matter, not just a margin one."""
    floors = (cfg.get("floors") or {}).get(product, {})
    if not floors:
        return pd.DataFrame()
    s = _slice(df, product, start, end).dropna(subset=["price"]).copy()
    s = s[s["price"] > 0]
    if s.empty:
        return pd.DataFrame()
    s["floor"] = s["date"].map(lambda d: dated_rate(floors, d))
    s = s.dropna(subset=["floor"])
    b = s[s["price"] < s["floor"] - tol].copy()
    if b.empty:
        return pd.DataFrame()
    b["shortfall"] = b["floor"] - b["price"]
    b["exposure"] = b["shortfall"] * b["volume"].fillna(0)
    return (b[["date", "station", "price", "floor", "shortfall", "volume", "exposure"]]
            .sort_values(["date", "station"]).reset_index(drop=True))


# ───────────────────── NPA price-floor fetcher (best effort) ───────────────
# The NPA publishes the minimum ex-pump price floor for each pricing window at
# npa.gov.gh/price-floor. There is no API, no published schema and no guarantee
# the page keeps its current shape, so this is written to fail loudly and hand
# over to manual entry rather than to guess. NOTHING it finds is written to the
# config automatically — it is staged for a human to confirm, because a wrong
# floor silently turns the compliance report into fiction.
NPA_FLOOR_URL = os.getenv("NPA_FLOOR_URL", "https://npa.gov.gh/price-floor/")
NPA_UA = ("Mozilla/5.0 (compatible; OMC-Analytics/1.0; internal pricing compliance)")

# Order matters: "Marine Gas Oil" must be claimed before the "gas oil" in AGO.
NPA_PRODUCTS = [
    ("MGO",  r"\bm\.?g\.?o\b|marine\s+gas"),
    ("LPG",  r"\blpg\b|liquefied"),
    ("KERO", r"\bkerosene\b|\bdpk\b|\bkero\b"),
    ("PMS",  r"\bpms\b|petrol|gasoline|\bsuper\b|premium\s+motor"),
    ("AGO",  r"\bago\b|diesel|gas\s*oil|automotive"),
]
NPA_NUM = r"\d{1,3}(?:,\d{3})*\.\d{1,4}"


def _npa_product(label):
    t = str(label).lower()
    for code, pat in NPA_PRODUCTS:
        if re.search(pat, t):
            return code
    return None


def _strip_html(payload):
    t = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", payload)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&")
          .replace("&#8211;", "-").replace("&ndash;", "-").replace("&#162;", "\u00a2"))
    return re.sub(r"[ \t\r\f\v]+", " ", t)


def npa_window_dates(payload):
    """Pull the effective date range out of the notice. Windows are nominally
    1-15 and 16-month end, but the published start slips (August 2026 ran from
    the 4th), so the printed dates are worth more than the convention."""
    t = _strip_html(payload)
    mons = ("january february march april may june july august september "
            "october november december").split()
    pat_mon = "|".join(mons)
    m = re.search(rf"(\d{{1,2}})\s*(?:-|\u2013|to)\s*(\d{{1,2}})\s+({pat_mon})\w*\s+(\d{{4}})",
                  t, re.I)
    if m:
        d1, d2, mon, yr = m.groups()
        i = mons.index(mon.lower()[:3] and
                       next(x for x in mons if x.startswith(mon.lower()[:3]))) + 1
        return (pd.Timestamp(int(yr), i, int(d1)), pd.Timestamp(int(yr), i, int(d2)))
    m = re.search(rf"({pat_mon})\w*\s+(\d{{1,2}})\s*(?:-|\u2013|to)\s*(?:(?:{pat_mon})\w*\s+)?"
                  rf"(\d{{1,2}}),?\s*(\d{{4}})?", t, re.I)
    if m:
        mon, d1, d2, yr = m.groups()
        i = mons.index(next(x for x in mons if x.startswith(mon.lower()[:3]))) + 1
        yr = int(yr) if yr else date.today().year
        return (pd.Timestamp(yr, i, int(d1)), pd.Timestamp(yr, i, int(d2)))
    return (None, None)


def parse_npa_floors(payload):
    """Read floors out of HTML or out of text pasted straight from the notice.
    Tries real tables first, then falls back to reading the sentences."""
    rows = []
    try:
        tables = pd.read_html(io.StringIO(payload))
    except Exception:
        tables = []
    for t in tables:
        try:
            t = t.astype(str)
        except Exception:
            continue
        for _, r in t.iterrows():
            cells = [str(x) for x in r.values]
            code = next((c for c in (_npa_product(x) for x in cells) if c), None)
            if not code:
                continue
            nums = []
            for x in cells:
                nums += re.findall(NPA_NUM, x)
            nums = [float(n.replace(",", "")) for n in nums]
            nums = [n for n in nums if 0.5 < n < 500]
            if nums:
                rows.append({"product": code, "price": nums[0],
                             "unit": "GHS/kg" if code == "LPG" else "GHS/L",
                             "source": "table", "raw": " | ".join(cells)[:160]})
    if not rows:
        text = _strip_html(payload)
        for code, pat in NPA_PRODUCTS:
            m = re.search(rf"(?:{pat})[^.;\n]{{0,160}}?(?:GH[\u00a2\u20b5cC]|GHS|GHC)?\s*({NPA_NUM})",
                          text, re.I)
            if not m:
                continue
            val = float(m.group(1).replace(",", ""))
            if not (0.5 < val < 500):
                continue
            rows.append({"product": code, "price": val,
                         "unit": "GHS/kg" if code == "LPG" else "GHS/L",
                         "source": "text", "raw": m.group(0).strip()[:160]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (out.drop_duplicates(subset=["product"], keep="first")
               .sort_values("product").reset_index(drop=True))


def fetch_npa_floors(url=None, timeout=30):
    """Returns (frame, (start, end), note). Never raises — the caller shows the
    note and falls through to pasting or typing the figures."""
    url = url or NPA_FLOOR_URL
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": NPA_UA,
                                                        "Accept": "text/html"})
        r.raise_for_status()
    except Exception as e:
        return pd.DataFrame(), (None, None), f"Couldn't reach {url} — {e}"
    body = r.text
    frame = parse_npa_floors(body)
    lo, hi = npa_window_dates(body)
    if frame.empty:
        note = ("Reached the page but found no floor figures in it. That usually means the "
                "table is rendered by JavaScript after load, which a plain fetch can't see. "
                "Copy the notice text from the page and paste it below — the same parser "
                "reads pasted text.")
    else:
        note = f"Read {len(frame)} product(s) from the page."
    return frame, (lo, hi), note


def price_dispersion(df, product, as_of=None):
    """Are all sites on the same price today? Spread across the network on the
    latest priced day, which is where uncontrolled discounting shows up."""
    s = df[(df["product"] == product)].dropna(subset=["price"])
    s = s[s["price"] > 0]
    if s.empty:
        return pd.DataFrame(), np.nan
    day = pd.Timestamp(as_of) if as_of is not None else s["date"].max()
    latest = (s[s["date"] <= day].sort_values("date").groupby("station")
                .tail(1)[["station", "date", "price", "volume"]]
                .sort_values("price", ascending=False).reset_index(drop=True))
    spread = float(latest["price"].max() - latest["price"].min()) if len(latest) else np.nan
    return latest, spread


def stock_revaluation(df, product, cfg, as_of=None):
    """Every ex-depot move re-prices the litres already in our tanks. Positive =
    we hold stock bought below the new cost (a windfall); negative = we are
    holding expensive stock into a falling market."""
    dealer, opexl, uppf, exd_tbl = margin_legs(cfg, product)
    tbl = rate_table_frame(exd_tbl)
    if tbl.empty or len(tbl) < 2:
        return pd.DataFrame()
    end = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    tbl = tbl[tbl["effective_from"] <= end].sort_values("effective_from")
    tbl["delta"] = tbl["value"].diff()
    rows = []
    for _, r in tbl.dropna(subset=["delta"]).iterrows():
        d = r["effective_from"]
        prev = d - pd.Timedelta(days=1)
        s = df[(df["product"] == product) & (df["date"] <= prev)]
        held = 0.0
        for st, g in s.groupby("station"):
            g = g.sort_values("date")
            stock = g["dip"].dropna()
            stock = stock.iloc[-1] if len(stock) else (g["closing"].dropna().iloc[-1]
                                                       if g["closing"].notna().any() else np.nan)
            if not np.isnan(stock):
                held += float(stock)
        if held <= 0:
            continue
        rows.append({"effective_from": d, "old_cost": r["value"] - r["delta"],
                     "new_cost": r["value"], "delta": r["delta"],
                     "stock_held": held, "revaluation": held * r["delta"]})
    return pd.DataFrame(rows)


# ───────────────────────── supply & replenishment ──────────────────────────
def plan_trucks(qty, sizes, min_drop):
    """Break a required volume into the compartment sizes actually available.
    The last leg may be a part-load, which is how drops really work; anything
    below the minimum economic drop is left for the next trip."""
    sizes = sorted([s for s in (sizes or []) if s and s > 0], reverse=True)
    if qty <= 0 or not sizes:
        return [], 0.0
    floor_qty = float(min_drop) if min_drop else sizes[-1] * 0.5
    load, left = [], float(qty)
    while left >= floor_qty and len(load) < 12:
        pick = next((s for s in sizes if s <= left), None)
        if pick is None:
            pick = round(left / 500.0) * 500.0
            if pick < floor_qty:
                break
        load.append(float(pick))
        left -= pick
    return load, float(sum(load))


def compute_replenishment(df, product, cfg, as_of, window=RUNWAY_WINDOW):
    """The load plan: who is closest to dry, how much ullage they have, and what
    to put on the road."""
    run = compute_runway(df, product, as_of, window)
    if run.empty:
        return pd.DataFrame()
    log = cfg.get("logistics") or {}
    sizes = log.get("brv_sizes") or [45000]
    min_drop = float(log.get("min_drop") or 0)
    reserve = float(log.get("ullage_reserve_pct") or 0) / 100.0
    dealer, opexl, uppf, exd_tbl = margin_legs(cfg, product)
    cost = dated_rate(exd_tbl, as_of)
    rows = []
    for _, r in run.iterrows():
        sc = station_cfg(cfg, r["station"])
        cap = float(sc.get(f"{product}_capacity") or 0)
        stock = float(r["stock_litres"]) if not np.isnan(r["stock_litres"]) else np.nan
        avg = float(r["avg_daily_sales"]) if not np.isnan(r["avg_daily_sales"]) else np.nan
        usable = cap * (1 - reserve) if cap else np.nan
        ullage = max(usable - stock, 0.0) if (cap and not np.isnan(stock)) else np.nan
        lead = float(sc["lead_time_days"]) + float(sc["safety_days"])
        rop = lead * avg if not np.isnan(avg) else np.nan
        due = (stock <= rop) if (not np.isnan(rop) and not np.isnan(stock)) else None
        load, planned = plan_trucks(ullage if not np.isnan(ullage) else 0.0, sizes, min_drop)
        rows.append({
            "station": r["station"], "stock_litres": stock, "capacity": cap,
            "ullage": ullage, "fill_pct": (stock / cap * 100) if cap else np.nan,
            "avg_daily_sales": avg, "days_cover": r["days_to_run_out"],
            "reorder_point": rop, "order_now": due, "risk": r["risk"],
            "suggested_order": planned, "truck_plan": " + ".join(f"{x:,.0f}" for x in load) or "—",
            "order_value": planned * cost if not np.isnan(cost) else np.nan,
            "as_of": r["as_of"],
        })
    res = pd.DataFrame(rows)
    return res.sort_values("days_cover", na_position="last").reset_index(drop=True)


def compute_delivery_perf(df, product, start, end):
    """Discharge vs shortage on arrival — transit loss is money that never
    reached the tank."""
    s = _slice(df, product, start, end)
    s = s[s["discharge"].fillna(0) > 0]
    if s.empty:
        return pd.DataFrame()
    rows = []
    for st, g in s.groupby("station"):
        disch = float(g["discharge"].sum())
        short = float(g["shortage"].dropna().sum())
        drops = int(len(g))
        gaps = g["date"].sort_values().diff().dropna().dt.days
        gaps = gaps[gaps > 0]
        rows.append({"station": st, "drops": drops, "discharged": disch,
                     "avg_drop": disch / drops if drops else np.nan,
                     "shortage": short,
                     "loss_pct": (short / disch * 100) if disch else np.nan,
                     "short_drops": int((g["shortage"].fillna(0) > 0).sum()),
                     "avg_gap_days": float(gaps.mean()) if len(gaps) else np.nan})
    return pd.DataFrame(rows).sort_values("loss_pct", ascending=False,
                                          na_position="last").reset_index(drop=True)


def delivery_log(df, product, start, end):
    s = _slice(df, product, start, end)
    s = s[s["discharge"].fillna(0) > 0]
    cols = ["date", "station", "discharge", "shortage", "dip", "closing", "price"]
    return s[cols].sort_values("date", ascending=False).reset_index(drop=True)


# ───────────────────────── control & compliance ────────────────────────────
def compute_submission(df, start, end):
    """Daily returns discipline. A site that stops reporting is either closed or
    hiding something; either way it needs a call."""
    rows = []
    for st, g in df[df["product"] == "PMS"].groupby("station"):
        first = max(g["date"].min(), pd.Timestamp(start))
        last = pd.Timestamp(end)
        if first > last:
            continue
        span = pd.date_range(first, last, freq="D")
        got = set(g[(g["date"] >= first) & (g["date"] <= last)]
                  .dropna(subset=["volume"])["date"])
        missing = [d for d in span if d not in got]
        rows.append({"station": st, "expected": len(span), "submitted": len(span) - len(missing),
                     "rate": (len(span) - len(missing)) / len(span) * 100 if len(span) else np.nan,
                     "missing_days": len(missing),
                     "last_return": max(got) if got else pd.NaT,
                     "days_silent": (last - max(got)).days if got else np.nan,
                     "missing_list": ", ".join(d.strftime("%d %b") for d in missing[-12:])})
    res = pd.DataFrame(rows)
    return res.sort_values("rate").reset_index(drop=True) if len(res) else res


def wetstock_control(df, station, product, start, end, tol_pct=TOLERANCE_PCT,
                     cap=DELIVERY_CAP):
    """Cumulative variance against a tolerance band that widens with throughput.
    A random error wanders around zero; a leak or a theft trends."""
    s = _slice(df, product, start, end)
    s = s[s["station"] == station].sort_values("date").copy()
    s = s[s["dip_var"].notna() & (s["dip_var"].abs() <= cap)]
    if s.empty:
        return pd.DataFrame(), {}
    s["cum_var"] = s["dip_var"].cumsum()
    s["cum_thru"] = s["volume"].fillna(0).cumsum()
    s["band"] = s["cum_thru"] * tol_pct / 100.0
    last = s.iloc[-1]
    breached = abs(last["cum_var"]) > last["band"] and last["band"] > 0
    tail = s["dip_var"].tail(14)
    drift = float(tail.mean()) if len(tail) else np.nan
    persistent = bool(len(tail) >= 7 and (tail < 0).mean() >= 0.7)
    verdict = ("leak or theft suspected" if breached and last["cum_var"] < 0 else
               "unexplained gain — check delivery bookings" if breached else
               "drifting negative" if persistent else "within control")
    return s, {"cum_var": float(last["cum_var"]), "band": float(last["band"]),
               "throughput": float(last["cum_thru"]), "breached": bool(breached),
               "drift_lpd": drift, "persistent_negative": persistent, "verdict": verdict}


def statutory_returns(df, cfg, year, month):
    """Volume return in the shape NPA asks for: litres received and litres sold
    by product, per site, for the month."""
    s = pd.Timestamp(date(year, month, 1))
    e = s + pd.offsets.MonthEnd(0)
    rows = []
    for rp in ("PMS", "AGO"):
        g = _slice(df, rp, s, e)
        for st, gg in g.groupby("station"):
            rows.append({"station": st, "product": rp,
                         "opening_litres": (gg.sort_values("date")["dip"].dropna().iloc[0]
                                            if gg["dip"].notna().any() else np.nan),
                         "received_litres": float(gg["discharge"].dropna().sum()),
                         "sold_litres": float(gg["volume"].dropna().sum()),
                         "closing_litres": (gg.sort_values("date")["dip"].dropna().iloc[-1]
                                            if gg["dip"].notna().any() else np.nan),
                         "days_reported": int(gg["volume"].notna().sum())})
    r = pd.DataFrame(rows)
    return r.sort_values(["station", "product"]).reset_index(drop=True) if len(r) else r


def integrity_flags(df, product, start, end, cap=DELIVERY_CAP):
    """Rows that should never happen in a clean book. Each one is a question for
    the site, not a conviction."""
    s = _slice(df, product, start, end).copy()
    out = []
    a = s[(s["volume"].fillna(0) > 0) & (s["dip"].isna()) & (s["closing"].isna())]
    for _, r in a.iterrows():
        out.append({"date": r["date"], "station": r["station"], "flag": "sales with no stock reading",
                    "detail": f"{r['volume']:,.0f} L sold, no dip or closing recorded"})
    b = s[s["dip_var"].abs() > cap]
    for _, r in b.iterrows():
        out.append({"date": r["date"], "station": r["station"], "flag": "unbooked delivery",
                    "detail": f"dip variance {r['dip_var']:,.0f} L — delivery likely not entered"})
    c = s[(s["discharge"].fillna(0) > 0) & (s["volume"].fillna(0) == 0)]
    for _, r in c.iterrows():
        out.append({"date": r["date"], "station": r["station"], "flag": "delivery, no sales",
                    "detail": f"{r['discharge']:,.0f} L received but zero sales that day"})
    d = s[(s["dip"].notna()) & (s["closing"].notna())]
    d = d[(d["dip"] - d["closing"]).abs() > 1500]
    for _, r in d.iterrows():
        out.append({"date": r["date"], "station": r["station"], "flag": "dip vs book gap",
                    "detail": f"dip {r['dip']:,.0f} L vs book {r['closing']:,.0f} L"})
    e = s.sort_values(["station", "date"]).copy()
    e["pchg"] = e.groupby("station")["price"].diff()
    e = e[e["pchg"].abs() > 1.5]
    for _, r in e.iterrows():
        out.append({"date": r["date"], "station": r["station"], "flag": "large price step",
                    "detail": f"price moved {r['pchg']:+,.2f} GHS/L in one day"})
    res = pd.DataFrame(out)
    return res.sort_values("date", ascending=False).reset_index(drop=True) if len(res) else res


# ─────────────────── working capital & market position ─────────────────────
def compute_stock_value(df, cfg, as_of, products=("PMS", "AGO")):
    rows = []
    for rp in products:
        dealer, opexl, uppf, exd_tbl = margin_legs(cfg, rp)
        cost = dated_rate(exd_tbl, as_of)
        s = df[(df["product"] == rp) & (df["date"] <= pd.Timestamp(as_of))]
        for st, g in s.groupby("station"):
            g = g.sort_values("date")
            stock = g["dip"].dropna()
            stock = float(stock.iloc[-1]) if len(stock) else (
                float(g["closing"].dropna().iloc[-1]) if g["closing"].notna().any() else np.nan)
            v = g["volume"].dropna().tail(RUNWAY_WINDOW)
            v = v[v > 0]
            avg = float(v.mean()) if len(v) else np.nan
            rows.append({"station": st, "product": rp, "stock_litres": stock,
                         "unit_cost": cost,
                         "stock_value": stock * cost if not (np.isnan(stock) or np.isnan(cost)) else np.nan,
                         "avg_daily_sales": avg,
                         "days_of_inventory": stock / avg if (avg and avg > 0 and not np.isnan(stock)) else np.nan})
    return pd.DataFrame(rows)


def compute_market_share(df, cfg):
    nat = ((cfg.get("market") or {}).get("national_monthly_litres") or {})
    if not nat:
        return pd.DataFrame()
    s = df[df["product"].isin(["PMS", "AGO"])].dropna(subset=["volume"]).copy()
    s["month"] = s["date"].dt.to_period("M").astype(str)
    g = s.groupby("month")["volume"].sum().reset_index(name="our_litres")
    g["national_litres"] = g["month"].map(lambda m: float(nat.get(m, np.nan))
                                          if nat.get(m) is not None else np.nan)
    g["share_pct"] = g["our_litres"] / g["national_litres"] * 100
    return g.dropna(subset=["national_litres"])


def site_scorecard(df, cfg, station, start, end):
    """Everything about one site on one page — the thing a territory manager
    actually carries into a review meeting."""
    card = {"station": station, "grades": {}}
    sc = station_cfg(cfg, station)
    card["profile"] = sc
    for rp in ("PMS", "AGO"):
        m = compute_margins(df, rp, cfg, start, end)
        m = m[m["station"] == station]
        run = compute_runway(df, rp, pd.Timestamp(end))
        run = run[run["station"] == station]
        eff = compute_efficiency(df, rp)
        eff = eff[eff["station"] == station]
        _, ctl = wetstock_control(df, station, rp, start, end,
                                  float((cfg.get("control") or {}).get("tolerance_pct", TOLERANCE_PCT)))
        card["grades"][rp] = {
            "margin": m.iloc[0].to_dict() if len(m) else {},
            "runway": run.iloc[0].to_dict() if len(run) else {},
            "efficiency": eff.iloc[0].to_dict() if len(eff) else {},
            "control": ctl,
        }
    return card


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

    # ═══════════════════════════════════════════════════════════════════════
    #                            OMC MODULES
    # ═══════════════════════════════════════════════════════════════════════
    if "omc_cfg" not in st.session_state:
        st.session_state["omc_cfg"] = load_config()
    cfg = st.session_state["omc_cfg"]
    CO = (cfg.get("company") or {}).get("name") or "Spartan Fuel"
    TOL = float((cfg.get("control") or {}).get("tolerance_pct", TOLERANCE_PCT))

    def _acc(colour):
        st.markdown(f"<style>:root{{--acc:{colour};}}</style>", unsafe_allow_html=True)

    def _cards(items, colour):
        cards = "".join(
            f"<div class='kpi'><div class='l'>{l}</div>"
            f"<div class='v'>{v}<span class='u'>{u}</span></div>"
            f"<div class='s'>{s}</div><div class='tick' style='background:{colour}'></div></div>"
            for l, v, u, s in items)
        st.markdown(f"<div class='kpi-row'>{cards}</div>", unsafe_allow_html=True)

    def phead(rp):
        st.markdown(f"<div class='prodhead' style='--acc2:{COLORS.get(rp, PCOL.get(rp, INK))}'>"
                    f"{GLABEL.get(rp, rp)}</div>", unsafe_allow_html=True)

    def _hero(icon, title, meta, badge=None):
        b = f"<span class='badge'>{badge}</span>" if badge else ""
        st.markdown(f"<div class='hero'><h1>{icon} {CO} — {title}{b}</h1>"
                    f"<div class='meta'>{meta}</div></div>", unsafe_allow_html=True)

    def _period(key, label="Period", default=None):
        d0, d1 = default or (dmin.date(), dmax.date())
        v = st.sidebar.date_input(label, (d0, d1), min_value=dmin.date(),
                                  max_value=dmax.date(), key=key)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return pd.Timestamp(v[0]), pd.Timestamp(v[1])
        return dmin, dmax

    def _refresh(key):
        st.sidebar.divider()
        if st.sidebar.button("↻ Refresh data", use_container_width=True, key=key):
            st.cache_data.clear()
            st.rerun()
        st.sidebar.caption(f"Source: Google · sheet **{used_sheet}**")

    def _n0(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"

    def _n2(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.2f}"

    def _cfg_gate(products):
        """The money modules are only honest if the ex-depot cost table is filled."""
        missing = [p for p in products if not margin_health(cfg, p)["has_cost"]]
        if missing:
            st.warning(
                f"No ex-depot cost entered for **{', '.join(missing)}**, so revenue is the only "
                "figure that can be trusted here — margin, cost and profit will read blank. "
                "Open **⚙️ Settings → Commercial model** and enter what you pay the BDC per "
                "litre, dated from the window it took effect.")
        return not missing

    # ══════════════════════════ MODULE · MARGINS ═══════════════════════════
    def render_margins():
        acc = "#C5821C"
        _acc(acc)
        st.sidebar.markdown("#### Margin period")
        st.sidebar.markdown("<span class='note'>revenue at pump price, cost at ex-depot, "
                            "less dealer margin, per-litre opex and the cost of stock "
                            "losses.</span>", unsafe_allow_html=True)
        ms, me = _period("mperiod", "Period", (cs_def, ce_def))
        mfocus = st.sidebar.selectbox("Station focus", ["All stations"] + stations, key="mfocus")
        _refresh("mrefresh")

        _hero("💰", "Margin & Contribution",
              f"{len(stations)} sites · {fmt(ms)} → {fmt(me)} · pump-to-depot economics",
              badge="P&L")
        _cfg_gate(["PMS", "AGO"])

        mg = {rp: compute_margins(df, rp, cfg, ms, me) for rp in ("PMS", "AGO")}
        allm = pd.concat([v.assign(product=k) for k, v in mg.items() if not v.empty],
                         ignore_index=True) if any(not v.empty for v in mg.values()) else pd.DataFrame()
        if allm.empty:
            st.info("No sales rows in this period.")
            return
        sel = allm if mfocus == "All stations" else allm[allm["station"] == mfocus]

        vol = sel["volume"].sum()
        rev = sel["revenue"].sum(skipna=True)
        gross = sel["gross_margin"].sum(skipna=True)
        cont = sel["contribution"].sum(skipna=True)
        losc = sel["loss_cost"].sum(skipna=True)
        gpl = gross / vol if vol else np.nan
        cpl = cont / vol if vol else np.nan

        mtabs = st.tabs(["Overview", "By station", "Waterfall", "Monthly trend",
                         "Break-even", "Working capital"])

        with mtabs[0]:
            st.markdown(f"<div class='eyebrow'>{mfocus} · PMS + AGO · {fmt(ms)} → {fmt(me)}</div>",
                        unsafe_allow_html=True)
            _cards([("Volume sold", _n0(vol), "L", "PMS + AGO"),
                    ("Revenue", _n0(rev), "GHS", "at pump price"),
                    ("Gross margin", _n0(gross), "GHS",
                     f"{_n2(gpl)} GHS/L over ex-depot"),
                    ("Contribution", _n0(cont), "GHS",
                     f"{_n2(cpl)} GHS/L after dealer, opex &amp; losses")], acc)
            loss_share = (losc / gross * 100) if (gross and gross == gross and gross > 0) else 0.0
            verdict = ("healthy" if (cpl == cpl and cpl > 0.25) else
                       "thin" if (cpl == cpl and cpl > 0) else "negative")
            st.markdown(
                f"<div class='summary'>💰 Every litre sold left <b>{_n2(cpl)} GHS</b> in the "
                f"business after paying the depot, the dealer and per-litre running costs, and "
                f"after writing off stock that went missing — a <b>{verdict}</b> margin. "
                f"Stock losses alone cost <b>GHS {_n0(losc)}</b> in this period, which is "
                f"{loss_share:,.1f}% of the gross margin earned."
                "</div>", unsafe_allow_html=True)

            by = allm.groupby("product").agg(volume=("volume", "sum"),
                                             gross=("gross_margin", "sum"),
                                             cont=("contribution", "sum")).reset_index()
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = go.Figure()
                for _, r in by.iterrows():
                    fig.add_bar(x=[GLABEL.get(r["product"], r["product"])], y=[r["cont"]],
                                marker_color=COLORS.get(r["product"], acc),
                                name=r["product"],
                                text=[f"GHS {r['cont']:,.0f}"], textposition="outside")
                fig.update_layout(title="Contribution by grade", showlegend=False)
                fig.update_yaxes(title_text="GHS")
                st.plotly_chart(style_fig(fig, 320, acc), use_container_width=True)
            with c2:
                by["per_litre"] = by["cont"] / by["volume"].replace(0, np.nan)
                fig = go.Figure()
                for _, r in by.iterrows():
                    fig.add_bar(x=[GLABEL.get(r["product"], r["product"])], y=[r["per_litre"]],
                                marker_color=COLORS.get(r["product"], acc),
                                text=[f"{r['per_litre']:,.2f}"], textposition="outside")
                fig.update_layout(title="Contribution per litre (GHS/L)", showlegend=False)
                st.plotly_chart(style_fig(fig, 320, acc), use_container_width=True)

        with mtabs[1]:
            st.markdown("<div class='eyebrow'>Per-site commercial result</div>",
                        unsafe_allow_html=True)
            for rp in ("PMS", "AGO"):
                d = mg[rp]
                if d.empty:
                    continue
                phead(rp)
                top = d.sort_values("contribution").tail(14)
                fig = go.Figure()
                fig.add_bar(y=top["station"], x=top["contribution"], orientation="h",
                            marker_color=COLORS[rp],
                            text=[f"{v:,.0f}" for v in top["contribution"]],
                            textposition="outside", textfont=dict(color=INK, size=11),
                            cliponaxis=False)
                fig.update_layout(title=f"{GLABEL[rp]} — contribution by station (GHS)")
                st.plotly_chart(style_fig(fig, max(280, 32 * len(top)), COLORS[rp]),
                                use_container_width=True)
                show = d.rename(columns={
                    "station": "Station", "volume": "Litres", "avg_price": "Avg price",
                    "avg_cost": "Avg ex-depot", "revenue": "Revenue", "gross_margin": "Gross",
                    "gross_per_litre": "Gross/L", "dealer_cost": "Dealer",
                    "site_opex": "Opex", "loss_cost": "Loss cost",
                    "contribution": "Contribution", "net_per_litre": "Net/L"})
                cols = ["Station", "Litres", "Avg price", "Avg ex-depot", "Revenue", "Gross",
                        "Gross/L", "Dealer", "Opex", "Loss cost", "Contribution", "Net/L"]
                st.dataframe(show[cols].style.format({
                    "Litres": "{:,.0f}", "Avg price": "{:,.2f}", "Avg ex-depot": "{:,.2f}",
                    "Revenue": "{:,.0f}", "Gross": "{:,.0f}", "Gross/L": "{:,.3f}",
                    "Dealer": "{:,.0f}", "Opex": "{:,.0f}", "Loss cost": "{:,.0f}",
                    "Contribution": "{:,.0f}", "Net/L": "{:,.3f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True)

        with mtabs[2]:
            st.markdown("<div class='eyebrow'>Where the cedi goes, per litre</div>",
                        unsafe_allow_html=True)
            rp = st.radio("Grade", ["PMS", "AGO"], horizontal=True, key="mwf")
            d = mg[rp]
            d = d if mfocus == "All stations" else d[d["station"] == mfocus]
            if d.empty or not d["volume"].sum():
                st.info("No litres for that selection.")
            else:
                v = d["volume"].sum()
                pump = d["revenue"].sum() / v
                cost = d["cogs"].sum() / v
                dealer = d["dealer_cost"].sum() / v
                opx = d["site_opex"].sum() / v
                loss = d["loss_cost"].sum() / v
                keep = pump - cost - dealer - opx - loss
                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                    x=["Pump price", "Ex-depot cost", "Dealer margin", "Site opex",
                       "Stock loss", "We keep"],
                    y=[pump, -cost, -dealer, -opx, -loss, keep],
                    text=[f"{pump:,.2f}", f"-{cost:,.2f}", f"-{dealer:,.2f}",
                          f"-{opx:,.2f}", f"-{loss:,.3f}", f"{keep:,.3f}"],
                    textposition="outside",
                    connector={"line": {"color": "rgba(140,140,140,.35)"}},
                    increasing={"marker": {"color": "#1F9D57"}},
                    decreasing={"marker": {"color": "#B00020"}},
                    totals={"marker": {"color": COLORS[rp]}}))
                fig.update_layout(title=f"{GLABEL[rp]} — build-down of one litre (GHS)")
                st.plotly_chart(style_fig(fig, 420, COLORS[rp]), use_container_width=True)
                st.caption("Read left to right: what the customer pays, minus what we paid the "
                           "BDC, minus what the dealer keeps, minus running cost per litre, "
                           "minus the litres that disappeared — what is left is ours.")

        with mtabs[3]:
            st.markdown("<div class='eyebrow'>Margin over time</div>", unsafe_allow_html=True)
            for rp in ("PMS", "AGO"):
                mm = margin_by_month(df, rp, cfg)
                if mm.empty:
                    continue
                phead(rp)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_bar(x=mm["month"], y=mm["gross"], name="Gross margin (GHS)",
                            marker_color=COLORS[rp], opacity=.75)
                fig.add_scatter(x=mm["month"], y=mm["gross_per_litre"], name="GHS per litre",
                                mode="lines+markers", line=dict(color=INK, width=2.5),
                                secondary_y=True)
                fig.update_layout(title=f"{GLABEL[rp]} — monthly margin and unit margin")
                fig.update_yaxes(title_text="GHS", secondary_y=False)
                fig.update_yaxes(title_text="GHS / litre", secondary_y=True, showgrid=False)
                st.plotly_chart(style_fig(fig, 330, COLORS[rp]), use_container_width=True)

        with mtabs[4]:
            st.markdown("<div class='eyebrow'>Litres needed to cover fixed site cost</div>",
                        unsafe_allow_html=True)
            be = allm[allm["breakeven_litres"].notna()]
            if be.empty:
                st.info("Enter a monthly fixed cost per site in **⚙️ Settings → Sites** and the "
                        "break-even volume for each one appears here.")
            else:
                be = be.assign(monthly_rate=be["volume"] / be["days"] * 30)
                be["cover"] = be["monthly_rate"] / be["breakeven_litres"] * 100
                bb = be.sort_values("cover")
                fig = go.Figure()
                fig.add_bar(y=bb["station"] + " · " + bb["product"], x=bb["cover"],
                            orientation="h",
                            marker_color=["#B00020" if c < 100 else "#1F9D57" for c in bb["cover"]],
                            text=[f"{c:,.0f}%" for c in bb["cover"]], textposition="outside",
                            textfont=dict(color=INK, size=11), cliponaxis=False)
                fig.add_vline(x=100, line_dash="dash", line_color=INK)
                fig.update_layout(title="Run-rate volume as % of break-even volume")
                st.plotly_chart(style_fig(fig, max(300, 30 * len(bb)), acc),
                                use_container_width=True)
                show = be.rename(columns={"station": "Station", "product": "Grade",
                                          "net_per_litre": "Net/L", "monthly_rate": "Run-rate L/mo",
                                          "breakeven_litres": "Break-even L/mo", "cover": "Cover %",
                                          "profit": "Profit after fixed (GHS)"})
                st.dataframe(show[["Station", "Grade", "Net/L", "Run-rate L/mo",
                                   "Break-even L/mo", "Cover %", "Profit after fixed (GHS)"]]
                             .style.format({"Net/L": "{:,.3f}", "Run-rate L/mo": "{:,.0f}",
                                            "Break-even L/mo": "{:,.0f}", "Cover %": "{:,.0f}",
                                            "Profit after fixed (GHS)": "{:,.0f}"}, na_rep="—"),
                             use_container_width=True, hide_index=True)

        with mtabs[5]:
            st.markdown("<div class='eyebrow'>Cash tied up in wet stock</div>",
                        unsafe_allow_html=True)
            sv = compute_stock_value(df, cfg, me)
            if sv.empty or sv["stock_value"].isna().all():
                st.info("Stock value needs an ex-depot cost in Settings.")
            else:
                tot = sv["stock_value"].sum(skipna=True)
                doi = (sv["stock_litres"].sum(skipna=True) /
                       sv["avg_daily_sales"].sum(skipna=True)) if sv["avg_daily_sales"].sum() else np.nan
                _cards([("Stock on hand", _n0(sv["stock_litres"].sum(skipna=True)), "L",
                         f"as at {fmt(me)}"),
                        ("Value at cost", _n0(tot), "GHS", "working capital in tanks"),
                        ("Days of inventory", _n2(doi), "days", "network average"),
                        ("Sites dry-risk", _n0(int((sv["days_of_inventory"] < 2).sum())), "",
                         "under 2 days of cover")], acc)
                piv = sv.pivot_table(index="station", columns="product", values="stock_value",
                                     aggfunc="sum").fillna(0)
                piv["__t"] = piv.sum(axis=1)
                piv = piv.sort_values("__t").drop(columns="__t")
                fig = go.Figure()
                for rp in ("PMS", "AGO"):
                    if rp in piv.columns:
                        fig.add_bar(y=piv.index, x=piv[rp], orientation="h", name=rp,
                                    marker_color=COLORS[rp])
                fig.update_layout(barmode="stack", title="Working capital held by site (GHS)")
                st.plotly_chart(style_fig(fig, max(300, 32 * len(piv)), acc),
                                use_container_width=True)
                st.dataframe(sv.rename(columns={
                    "station": "Station", "product": "Grade", "stock_litres": "Litres",
                    "unit_cost": "Cost/L", "stock_value": "Value (GHS)",
                    "avg_daily_sales": "Avg daily L", "days_of_inventory": "Days of stock"})
                    .style.format({"Litres": "{:,.0f}", "Cost/L": "{:,.2f}",
                                   "Value (GHS)": "{:,.0f}", "Avg daily L": "{:,.0f}",
                                   "Days of stock": "{:,.1f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download margin workbook",
                           build_excel({f"margin_{k}": v for k, v in mg.items()}),
                           file_name=f"margins_{ms:%Y%m%d}_{me:%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ══════════════════════════ MODULE · PRICING ═══════════════════════════
    def render_pricing():
        acc = "#3A6EA5"
        _acc(acc)
        st.sidebar.markdown("#### Pricing")
        st.sidebar.markdown("<span class='note'>windows run 1–15 and 16–month end, the way "
                            "NPA prices them.</span>", unsafe_allow_html=True)
        ps, pe = _period("pperiod", "Period", (dmin.date(), dmax.date()))
        _refresh("prefresh")

        _hero("🏷️", "Pricing & Window Control",
              f"{fmt(ps)} → {fmt(pe)} · floors, headroom and price response", badge="WINDOWS")

        ptabs = st.tabs(["Window board", "Floor compliance", "Network price spread",
                         "Price response", "Stock revaluation"])

        with ptabs[0]:
            for rp in ("PMS", "AGO"):
                w = compute_window_pricing(df, rp, cfg, ps, pe)
                if w.empty:
                    continue
                phead(rp)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_bar(x=w["window"], y=w["daily_volume"], name="Litres / day",
                            marker_color="rgba(58,110,165,.35)")
                fig.add_scatter(x=w["window"], y=w["avg_price"], name="Our avg price",
                                mode="lines+markers", line=dict(color=COLORS[rp], width=3),
                                secondary_y=True)
                if w["floor"].notna().any():
                    fig.add_scatter(x=w["window"], y=w["floor"], name="NPA floor",
                                    mode="lines", line=dict(color=INK, width=2, dash="dot"),
                                    secondary_y=True)
                fig.update_layout(title=f"{GLABEL[rp]} — price vs volume by pricing window")
                fig.update_yaxes(title_text="litres / day", secondary_y=False)
                fig.update_yaxes(title_text="GHS / litre", secondary_y=True, showgrid=False)
                st.plotly_chart(style_fig(fig, 350, COLORS[rp]), use_container_width=True)
                show = w.rename(columns={
                    "window": "Window", "days": "Days", "avg_price": "Avg price",
                    "floor": "NPA floor", "headroom": "Headroom", "exdepot": "Ex-depot",
                    "unit_margin": "Unit margin", "volume": "Litres",
                    "daily_volume": "L / day", "price_chg": "Price Δ",
                    "vol_chg_pct": "Volume Δ%", "arc_elasticity": "Elasticity"})
                st.dataframe(show[["Window", "Days", "Avg price", "NPA floor", "Headroom",
                                   "Ex-depot", "Unit margin", "Litres", "L / day",
                                   "Price Δ", "Volume Δ%", "Elasticity"]].style.format({
                    "Avg price": "{:,.2f}", "NPA floor": "{:,.2f}", "Headroom": "{:,.2f}",
                    "Ex-depot": "{:,.2f}", "Unit margin": "{:,.3f}", "Litres": "{:,.0f}",
                    "L / day": "{:,.0f}", "Price Δ": "{:+,.2f}", "Volume Δ%": "{:+,.1f}",
                    "Elasticity": "{:,.2f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True)
            st.caption("Headroom is how far our pump price sits above the NPA minimum. Negative "
                       "headroom means we were pricing under the floor for that window.")

        with ptabs[1]:
            st.markdown("<div class='eyebrow'>Selling below the NPA minimum</div>",
                        unsafe_allow_html=True)
            any_floor = any((cfg.get("floors") or {}).get(rp) for rp in ("PMS", "AGO"))
            if not any_floor:
                st.info("No price floors entered yet. Add each window's NPA minimum in "
                        "**⚙️ Settings → Price floors** and every site-day below it is flagged "
                        "here with the cedi exposure attached.")
            else:
                tot = 0.0
                for rp in ("PMS", "AGO"):
                    br = compute_floor_breaches(df, rp, cfg, ps, pe)
                    phead(rp)
                    if br.empty:
                        st.success(f"No {rp} site-days below the floor in this period.")
                        continue
                    tot += br["exposure"].sum()
                    st.error(f"{len(br)} site-day(s) under the floor across "
                             f"{br['station'].nunique()} site(s) — GHS "
                             f"{br['exposure'].sum():,.0f} of volume sold below the minimum.")
                    st.dataframe(br.assign(date=br["date"].dt.strftime("%d %b %Y")).rename(columns={
                        "date": "Date", "station": "Station", "price": "Our price",
                        "floor": "Floor", "shortfall": "Under by", "volume": "Litres",
                        "exposure": "Exposure (GHS)"}).style.format({
                        "Our price": "{:,.2f}", "Floor": "{:,.2f}", "Under by": "{:,.2f}",
                        "Litres": "{:,.0f}", "Exposure (GHS)": "{:,.0f}"}, na_rep="—"),
                        use_container_width=True, hide_index=True)
                if tot:
                    st.caption("Exposure is the shortfall per litre multiplied by the litres sold "
                               "at that price — the size of the pricing breach, and the number a "
                               "regulator would ask about first.")

        with ptabs[2]:
            st.markdown("<div class='eyebrow'>Are all our sites on the same price?</div>",
                        unsafe_allow_html=True)
            for rp in ("PMS", "AGO"):
                latest, spread = price_dispersion(df, rp)
                if latest.empty:
                    continue
                phead(rp)
                c1, c2 = st.columns([1, 2], gap="large")
                with c1:
                    st.metric("Network spread", f"{spread:,.2f} GHS/L")
                    st.metric("Highest", f"{latest['price'].max():,.2f}")
                    st.metric("Lowest", f"{latest['price'].min():,.2f}")
                with c2:
                    fig = go.Figure()
                    fig.add_bar(y=latest["station"], x=latest["price"], orientation="h",
                                marker_color=COLORS[rp],
                                text=[f"{p:,.2f}" for p in latest["price"]],
                                textposition="outside", textfont=dict(color=INK, size=11),
                                cliponaxis=False)
                    med = latest["price"].median()
                    fig.add_vline(x=med, line_dash="dash", line_color=INK)
                    fig.update_layout(title=f"{GLABEL[rp]} — latest pump price by site "
                                            f"(dashed = network median)")
                    fig.update_xaxes(range=[latest["price"].min() * 0.97,
                                            latest["price"].max() * 1.03])
                    st.plotly_chart(style_fig(fig, max(260, 30 * len(latest)), COLORS[rp]),
                                    use_container_width=True)
            st.caption("A wide spread is either deliberate territory pricing or a site quietly "
                       "discounting. Both are worth knowing; only one is a decision.")

        with ptabs[3]:
            st.markdown("<div class='eyebrow'>What happened to volume when we moved price</div>",
                        unsafe_allow_html=True)
            for rp in ("PMS", "AGO"):
                w = compute_window_pricing(df, rp, cfg, ps, pe).dropna(subset=["price_chg_pct"])
                w = w[w["price_chg_pct"].abs() > 0.2]
                if w.empty:
                    continue
                phead(rp)
                fig = px.scatter(w, x="price_chg_pct", y="vol_chg_pct", text="window",
                                 labels={"price_chg_pct": "price change %",
                                         "vol_chg_pct": "daily volume change %"},
                                 title=f"{GLABEL[rp]} — window-on-window price vs volume move")
                fig.update_traces(marker=dict(size=13, color=COLORS[rp]),
                                  textposition="top center", textfont=dict(size=9, color=INK))
                fig.add_hline(y=0, line_color=AXIS)
                fig.add_vline(x=0, line_color=AXIS)
                st.plotly_chart(style_fig(fig, 380, COLORS[rp]), use_container_width=True)
                med_e = w["arc_elasticity"].median()
                if med_e == med_e:
                    st.markdown(f"<div class='summary'>📈 Across these windows the typical "
                                f"response was <b>{med_e:,.2f}</b> — {elast_brief(med_e)}"
                                "</div>", unsafe_allow_html=True)

        with ptabs[4]:
            st.markdown("<div class='eyebrow'>Gain or loss on stock when depot cost moved</div>",
                        unsafe_allow_html=True)
            for rp in ("PMS", "AGO"):
                rv = stock_revaluation(df, rp, cfg, pe)
                if rv.empty:
                    continue
                phead(rp)
                fig = go.Figure()
                fig.add_bar(x=rv["effective_from"], y=rv["revaluation"],
                            marker_color=["#1F9D57" if v > 0 else "#B00020"
                                          for v in rv["revaluation"]],
                            text=[f"{v:,.0f}" for v in rv["revaluation"]],
                            textposition="outside", textfont=dict(color=INK, size=10))
                fig.update_layout(title=f"{GLABEL[rp]} — revaluation of stock held at each "
                                        f"ex-depot change (GHS)")
                st.plotly_chart(style_fig(fig, 320, COLORS[rp]), use_container_width=True)
                st.dataframe(rv.assign(effective_from=rv["effective_from"].dt.strftime("%d %b %Y"))
                             .rename(columns={"effective_from": "Effective", "old_cost": "Old cost",
                                              "new_cost": "New cost", "delta": "Δ GHS/L",
                                              "stock_held": "Stock held (L)",
                                              "revaluation": "Gain / (loss) GHS"})
                             .style.format({"Old cost": "{:,.2f}", "New cost": "{:,.2f}",
                                            "Δ GHS/L": "{:+,.2f}", "Stock held (L)": "{:,.0f}",
                                            "Gain / (loss) GHS": "{:+,.0f}"}, na_rep="—"),
                             use_container_width=True, hide_index=True)
            st.caption("When the depot price rises, litres already in our tanks were bought "
                       "cheaper — that is a real one-off gain. When it falls, we are carrying "
                       "expensive stock into a cheaper market. Loading ahead of a window is worth "
                       "exactly this number.")

    # ══════════════════════════ MODULE · SUPPLY ════════════════════════════
    def render_supply():
        acc = "#7A0010"
        _acc(acc)
        st.sidebar.markdown("#### Supply")
        st.sidebar.markdown("<span class='note'>load plan uses tank capacity, lead time and "
                            "the last 7 days of sales.</span>", unsafe_allow_html=True)
        ss_, se_ = _period("speriod", "Delivery history", (cs_def, ce_def))
        _refresh("srefresh")

        _hero("🚚", "Supply & Replenishment",
              f"as at {fmt(dmax)} · ullage, reorder points and transit loss", badge="LOGISTICS")

        stabs = st.tabs(["Load plan", "Ullage & cover", "Transit loss", "Delivery log"])

        with stabs[0]:
            st.markdown("<div class='eyebrow'>What to put on the road next</div>",
                        unsafe_allow_html=True)
            caps = sum(1 for s in stations
                       if station_cfg(cfg, s)["PMS_capacity"] or station_cfg(cfg, s)["AGO_capacity"])
            if caps == 0:
                st.warning("No tank capacities on file, so ullage and order quantities can't be "
                           "worked out. Enter each site's tank size in **⚙️ Settings → Sites** — "
                           "days of cover below still works without it.")
            for rp in ("PMS", "AGO"):
                rep = compute_replenishment(df, rp, cfg, dmax)
                if rep.empty:
                    continue
                phead(rp)
                due = rep[rep["order_now"] == True]
                if len(due):
                    total = due["suggested_order"].sum()
                    st.markdown(
                        f"<div class='summary'>🚚 <b>{len(due)}</b> {rp} site(s) are at or below "
                        f"their reorder point: <b>{', '.join(due['station'].head(6))}</b>. "
                        f"Planned lift <b>{total:,.0f} L</b>"
                        + (f" ≈ <b>GHS {due['order_value'].sum():,.0f}</b> at today's depot price."
                           if due["order_value"].notna().any() else ".") + "</div>",
                        unsafe_allow_html=True)
                else:
                    st.success(f"No {rp} site is below its reorder point today.")
                show = rep.rename(columns={
                    "station": "Station", "stock_litres": "Stock now", "capacity": "Capacity",
                    "ullage": "Ullage", "fill_pct": "Tank %", "avg_daily_sales": "Avg daily",
                    "days_cover": "Days cover", "reorder_point": "Reorder at",
                    "order_now": "Order?", "suggested_order": "Suggested L",
                    "truck_plan": "Compartments", "order_value": "Order value (GHS)"})
                st.dataframe(show[["Station", "Stock now", "Capacity", "Tank %", "Ullage",
                                   "Avg daily", "Days cover", "Reorder at", "Order?",
                                   "Suggested L", "Compartments", "Order value (GHS)"]]
                             .style.format({"Stock now": "{:,.0f}", "Capacity": "{:,.0f}",
                                            "Tank %": "{:,.0f}", "Ullage": "{:,.0f}",
                                            "Avg daily": "{:,.0f}", "Days cover": "{:,.1f}",
                                            "Reorder at": "{:,.0f}", "Suggested L": "{:,.0f}",
                                            "Order value (GHS)": "{:,.0f}"}, na_rep="—"),
                             use_container_width=True, hide_index=True)

        with stabs[1]:
            st.markdown("<div class='eyebrow'>How full the network is</div>",
                        unsafe_allow_html=True)
            for rp in ("PMS", "AGO"):
                rep = compute_replenishment(df, rp, cfg, dmax)
                rep = rep[rep["capacity"] > 0] if not rep.empty else rep
                if rep.empty:
                    continue
                phead(rp)
                r = rep.sort_values("fill_pct")
                fig = go.Figure()
                fig.add_bar(y=r["station"], x=r["capacity"], orientation="h", name="Capacity",
                            marker_color="rgba(140,140,140,.25)")
                fig.add_bar(y=r["station"], x=r["stock_litres"], orientation="h", name="In tank",
                            marker_color=COLORS[rp],
                            text=[f"{v:,.0f}%" if v == v else "" for v in r["fill_pct"]],
                            textposition="outside", textfont=dict(color=INK, size=10),
                            cliponaxis=False)
                fig.update_layout(barmode="overlay",
                                  title=f"{GLABEL[rp]} — tank fill (label = % full)")
                st.plotly_chart(style_fig(fig, max(280, 32 * len(r)), COLORS[rp]),
                                use_container_width=True)

        with stabs[2]:
            st.markdown("<div class='eyebrow'>Litres that left the depot but never reached the "
                        "tank</div>", unsafe_allow_html=True)
            rows = []
            for rp in ("PMS", "AGO"):
                dp = compute_delivery_perf(df, rp, ss_, se_)
                if dp.empty:
                    continue
                dealer, opexl, uppf, exd = margin_legs(cfg, rp)
                cost = dated_rate(exd, se_)
                dp = dp.assign(product=rp,
                               loss_value=dp["shortage"] * (cost if cost == cost else np.nan))
                rows.append(dp)
                phead(rp)
                fig = go.Figure()
                d2 = dp.sort_values("loss_pct")
                fig.add_bar(y=d2["station"], x=d2["loss_pct"], orientation="h",
                            marker_color=COLORS[rp],
                            text=[f"{v:,.2f}%" if v == v else "" for v in d2["loss_pct"]],
                            textposition="outside", textfont=dict(color=INK, size=10),
                            cliponaxis=False)
                fig.update_layout(title=f"{GLABEL[rp]} — shortage as % of litres discharged")
                st.plotly_chart(style_fig(fig, max(260, 30 * len(d2)), COLORS[rp]),
                                use_container_width=True)
                st.dataframe(dp.rename(columns={
                    "station": "Station", "drops": "Drops", "discharged": "Discharged L",
                    "avg_drop": "Avg drop", "shortage": "Shortage L", "loss_pct": "Loss %",
                    "short_drops": "Short drops", "avg_gap_days": "Days between drops",
                    "loss_value": "Cost (GHS)"})[["Station", "Drops", "Discharged L", "Avg drop",
                                                  "Shortage L", "Loss %", "Short drops",
                                                  "Days between drops", "Cost (GHS)"]]
                    .style.format({"Discharged L": "{:,.0f}", "Avg drop": "{:,.0f}",
                                   "Shortage L": "{:,.0f}", "Loss %": "{:,.2f}",
                                   "Days between drops": "{:,.1f}", "Cost (GHS)": "{:,.0f}"},
                                  na_rep="—"),
                    use_container_width=True, hide_index=True)
            if rows:
                allp = pd.concat(rows, ignore_index=True)
                tl = allp["shortage"].sum()
                tv = allp["loss_value"].sum(skipna=True)
                st.markdown(f"<div class='summary'>🚚 Transit shortage across the period: "
                            f"<b>{tl:,.0f} L</b>"
                            + (f" ≈ <b>GHS {tv:,.0f}</b>." if tv == tv and tv else ".")
                            + " This is a claim against the transporter or the loading depot, not "
                              "a station loss — it happens before the product reaches the "
                              "forecourt.</div>", unsafe_allow_html=True)

        with stabs[3]:
            st.markdown("<div class='eyebrow'>Every discharge in the period</div>",
                        unsafe_allow_html=True)
            rp = st.radio("Grade", ["PMS", "AGO"], horizontal=True, key="dlog")
            lg = delivery_log(df, rp, ss_, se_)
            if lg.empty:
                st.info("No discharges recorded for that grade in this period.")
            else:
                st.dataframe(lg.assign(date=lg["date"].dt.strftime("%d %b %Y")).rename(columns={
                    "date": "Date", "station": "Station", "discharge": "Discharged L",
                    "shortage": "Shortage L", "dip": "Dip after", "closing": "Book after",
                    "price": "Price"}).style.format({
                    "Discharged L": "{:,.0f}", "Shortage L": "{:,.0f}", "Dip after": "{:,.0f}",
                    "Book after": "{:,.0f}", "Price": "{:,.2f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True, height=460)

    # ══════════════════════ MODULE · CONTROL & COMPLIANCE ══════════════════
    def render_control():
        acc = "#1F9D57"
        _acc(acc)
        st.sidebar.markdown("#### Control period")
        cs_, ce_ = _period("cperiod", "Period", (cs_def, ce_def))
        _refresh("crefresh")

        _hero("🛡️", "Control & Compliance",
              f"{fmt(cs_)} → {fmt(ce_)} · wetstock, returns discipline and statutory volumes",
              badge="CONTROL")

        ctabs = st.tabs(["Wetstock control", "Returns discipline", "Integrity flags",
                         "Statutory volume return"])

        with ctabs[0]:
            st.markdown("<div class='eyebrow'>Cumulative variance against tolerance</div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns([2, 1])
            with c1:
                wsite = st.selectbox("Site", stations, key="wsite")
            with c2:
                wgrade = st.radio("Grade", ["PMS", "AGO"], horizontal=True, key="wgrade")
            series, ctl = wetstock_control(df, wsite, wgrade, cs_, ce_, TOL)
            if series.empty:
                st.info("No usable dip-variance readings for that site and grade in this period.")
            else:
                _cards([("Cumulative variance", _n0(ctl["cum_var"]), "L",
                         "sum of daily dip differences"),
                        ("Tolerance band", f"±{_n0(ctl['band'])}", "L",
                         f"{TOL}% of {_n0(ctl['throughput'])} L throughput"),
                        ("Recent drift", _n2(ctl["drift_lpd"]), "L/day", "last 14 readings"),
                        ("Verdict", ctl["verdict"].split()[0].title(), "",
                         ctl["verdict"])], acc)
                fig = go.Figure()
                fig.add_scatter(x=series["date"], y=series["band"], name="Upper tolerance",
                                mode="lines", line=dict(color=AXIS, width=1, dash="dot"))
                fig.add_scatter(x=series["date"], y=-series["band"], name="Lower tolerance",
                                mode="lines", line=dict(color=AXIS, width=1, dash="dot"),
                                fill="tonexty", fillcolor="rgba(140,140,140,.10)")
                fig.add_scatter(x=series["date"], y=series["cum_var"], name="Cumulative variance",
                                mode="lines", line=dict(color=COLORS[wgrade], width=3))
                fig.add_hline(y=0, line_color=AXIS)
                fig.update_layout(title=f"{wsite} · {GLABEL[wgrade]} — wetstock control chart")
                fig.update_yaxes(title_text="litres, cumulative")
                st.plotly_chart(style_fig(fig, 380, COLORS[wgrade]), use_container_width=True)
                if ctl["breached"] and ctl["cum_var"] < 0:
                    st.error("Cumulative loss has broken the tolerance band and keeps falling. "
                             "That pattern is a leak, a mis-calibrated pump or a systematic "
                             "short-delivery — a one-off error would wander back toward zero.")
                elif ctl["breached"]:
                    st.warning("Cumulative variance is positive beyond tolerance, which usually "
                               "means deliveries are reaching the tank without being booked.")
                else:
                    st.success("Variance is wandering inside tolerance — normal measurement noise.")

            st.divider()
            st.markdown("<div class='eyebrow'>Network control summary</div>",
                        unsafe_allow_html=True)
            rows = []
            for stn in stations:
                for rp in ("PMS", "AGO"):
                    _, c = wetstock_control(df, stn, rp, cs_, ce_, TOL)
                    if c:
                        rows.append({"Station": stn, "Grade": rp, "Cumulative L": c["cum_var"],
                                     "Tolerance ±L": c["band"], "Drift L/day": c["drift_lpd"],
                                     "Status": c["verdict"]})
            if rows:
                cd = pd.DataFrame(rows).sort_values("Cumulative L")
                st.dataframe(cd.style.format({"Cumulative L": "{:,.0f}", "Tolerance ±L": "{:,.0f}",
                                              "Drift L/day": "{:,.1f}"}, na_rep="—"),
                             use_container_width=True, hide_index=True)

        with ctabs[1]:
            st.markdown("<div class='eyebrow'>Are the sites sending their daily returns?</div>",
                        unsafe_allow_html=True)
            sub = compute_submission(df, cs_, ce_)
            if sub.empty:
                st.info("No sites in this period.")
            else:
                silent = sub[sub["days_silent"] >= 3]
                _cards([("Network return rate", f"{sub['rate'].mean():,.0f}", "%",
                         "days submitted vs expected"),
                        ("Sites at 100%", _n0(int((sub["rate"] >= 99.9).sum())), "",
                         "no missing days"),
                        ("Silent 3+ days", _n0(len(silent)), "", "no return recently"),
                        ("Missing days total", _n0(sub["missing_days"].sum()), "",
                         "across the network")], acc)
                if len(silent):
                    st.warning("No returns for 3 days or more: **"
                               + ", ".join(silent["station"]) + "**")
                s2 = sub.sort_values("rate")
                fig = go.Figure()
                fig.add_bar(y=s2["station"], x=s2["rate"], orientation="h",
                            marker_color=["#B00020" if r < 90 else "#C5821C" if r < 99
                                          else "#1F9D57" for r in s2["rate"]],
                            text=[f"{r:,.0f}%" for r in s2["rate"]], textposition="outside",
                            textfont=dict(color=INK, size=11), cliponaxis=False)
                fig.update_layout(title="Daily return submission rate by site")
                fig.update_xaxes(range=[0, 108])
                st.plotly_chart(style_fig(fig, max(280, 30 * len(s2)), acc),
                                use_container_width=True)
                st.dataframe(sub.assign(last_return=sub["last_return"].apply(
                    lambda d: "—" if pd.isna(d) else pd.Timestamp(d).strftime("%d %b %Y"))).rename(
                    columns={"station": "Station", "expected": "Expected days",
                             "submitted": "Submitted", "rate": "Rate %",
                             "missing_days": "Missing", "last_return": "Last return",
                             "days_silent": "Days silent", "missing_list": "Recent gaps"})
                    .style.format({"Rate %": "{:,.0f}", "Days silent": "{:,.0f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True)
                st.caption("A site that stops reporting is invisible to every other number in "
                           "this system — chase the gaps before reading the rankings.")

        with ctabs[2]:
            st.markdown("<div class='eyebrow'>Rows that need an explanation</div>",
                        unsafe_allow_html=True)
            parts = []
            for rp in ("PMS", "AGO"):
                fr = integrity_flags(df, rp, cs_, ce_)
                if not fr.empty:
                    parts.append(fr.assign(grade=rp))
            flg = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if flg.empty:
                st.success("Nothing anomalous in the book for this period.")
            else:
                counts = flg["flag"].value_counts()
                _cards([(k[:22], _n0(v), "", "occurrences") for k, v in counts.head(4).items()], acc)
                st.dataframe(flg.assign(date=flg["date"].dt.strftime("%d %b %Y")).rename(columns={
                    "date": "Date", "station": "Station", "grade": "Grade", "flag": "Flag",
                    "detail": "Detail"})[["Date", "Station", "Grade", "Flag", "Detail"]],
                    use_container_width=True, hide_index=True, height=430)
                st.caption("These are questions, not accusations. Most resolve as a delivery "
                           "entered on the wrong day or a dip taken before the discharge settled.")

        with ctabs[3]:
            st.markdown("<div class='eyebrow'>Monthly volume return</div>",
                        unsafe_allow_html=True)
            months = pd.period_range(dmin.to_period("M"), dmax.to_period("M"), freq="M")
            labels = [p.strftime("%B %Y") for p in months]
            pick = st.selectbox("Return month", labels, index=len(labels) - 1, key="statmonth")
            sel = months[labels.index(pick)]
            ret = statutory_returns(df, cfg, sel.year, sel.month)
            if ret.empty:
                st.info("No data for that month.")
            else:
                tot = ret.groupby("product").agg(received=("received_litres", "sum"),
                                                 sold=("sold_litres", "sum")).reset_index()
                _cards([(f"{r['product']} received", _n0(r["received"]), "L", pick)
                        for _, r in tot.iterrows()]
                       + [(f"{r['product']} sold", _n0(r["sold"]), "L", pick)
                          for _, r in tot.iterrows()], acc)
                st.dataframe(ret.rename(columns={
                    "station": "Station", "product": "Product", "opening_litres": "Opening L",
                    "received_litres": "Received L", "sold_litres": "Sold L",
                    "closing_litres": "Closing L", "days_reported": "Days reported"})
                    .style.format({"Opening L": "{:,.0f}", "Received L": "{:,.0f}",
                                   "Sold L": "{:,.0f}", "Closing L": "{:,.0f}"}, na_rep="—"),
                    use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download volume return (Excel)",
                                   build_excel({f"return_{sel.strftime('%Y_%m')}": ret}),
                                   file_name=f"volume_return_{sel.strftime('%Y_%m')}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument."
                                        "spreadsheetml.sheet")
                st.caption("Opening and closing are physical dips, received is booked discharge "
                           "and sold is metered sales. Reconcile these before they go anywhere "
                           "official.")

    # ══════════════════════════ MODULE · SITE CARD ═════════════════════════
    def render_site():
        acc = "#0F766E"
        _acc(acc)
        st.sidebar.markdown("#### Site review")
        sst, sse = _period("siteperiod", "Period", (cs_def, ce_def))
        who = st.sidebar.selectbox("Site", stations, key="sitepick")
        _refresh("siterefresh")

        sc = station_cfg(cfg, who)
        meta = " · ".join(x for x in [sc.get("region") or "", sc.get("dealer") or "",
                                      f"{fmt(sst)} → {fmt(sse)}"] if x)
        _hero("🏪", who, meta, badge="SITE CARD")

        card = site_scorecard(df, cfg, who, sst, sse)
        tot_vol = sum((card["grades"][rp]["margin"].get("volume") or 0) for rp in ("PMS", "AGO"))
        tot_con = sum((card["grades"][rp]["margin"].get("contribution") or 0) for rp in ("PMS", "AGO"))
        worst = min([card["grades"][rp]["runway"].get("days_to_run_out", np.nan)
                     for rp in ("PMS", "AGO")], default=np.nan)
        loss = sum((card["grades"][rp]["margin"].get("loss_cost") or 0) for rp in ("PMS", "AGO"))
        _cards([("Volume", _n0(tot_vol), "L", "both grades, this period"),
                ("Contribution", _n0(tot_con), "GHS", "after dealer, opex and losses"),
                ("Lowest cover", _n2(worst), "days", "grade closest to dry"),
                ("Stock loss cost", _n0(loss), "GHS", "value of missing litres")], acc)

        for rp in ("PMS", "AGO"):
            g = card["grades"][rp]
            phead(rp)
            m, r, e, c = g["margin"], g["runway"], g["efficiency"], g["control"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Litres", _n0(m.get("volume")))
            c2.metric("Contribution/L", _n2(m.get("net_per_litre")))
            c3.metric("Days cover", _n2(r.get("days_to_run_out")))
            c4.metric("Sell-through", f"{(e.get('turnover_per_day') or 0) * 100:,.1f}%")
            bits = []
            if m.get("volume"):
                bits.append(f"Sold <b>{m['volume']:,.0f} L</b> at an average of "
                            f"<b>{m.get('avg_price', float('nan')):,.2f} GHS/L</b>")
                if m.get("net_per_litre") == m.get("net_per_litre"):
                    bits.append(f"keeping <b>{m['net_per_litre']:,.3f} GHS</b> a litre")
            if r.get("days_to_run_out") == r.get("days_to_run_out"):
                bits.append(f"about <b>{r['days_to_run_out']:,.1f} days</b> of stock left "
                            f"({r.get('risk', '')})")
            if c:
                bits.append(f"wetstock is <b>{c.get('verdict', '')}</b>")
            if bits:
                st.markdown("<div class='summary'>" + ", ".join(bits) + ".</div>",
                            unsafe_allow_html=True)
            hist = df[(df["station"] == who) & (df["product"] == rp) &
                      (df["date"] >= sst) & (df["date"] <= sse)].dropna(subset=["volume"])
            if not hist.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_bar(x=hist["date"], y=hist["volume"], name="Litres",
                            marker_color=COLORS[rp], opacity=.65)
                if hist["price"].notna().any():
                    fig.add_scatter(x=hist["date"], y=hist["price"], name="Price",
                                    mode="lines", line=dict(color=INK, width=2),
                                    secondary_y=True)
                fig.update_layout(title=f"{GLABEL[rp]} — daily volume and price")
                fig.update_yaxes(title_text="litres", secondary_y=False)
                fig.update_yaxes(title_text="GHS/L", secondary_y=True, showgrid=False)
                st.plotly_chart(style_fig(fig, 300, COLORS[rp]), use_container_width=True)

        st.divider()
        st.markdown("<div class='eyebrow'>Site register</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Field": k.replace("_", " ").title(), "Value": v}
                                   for k, v in sc.items()]),
                     use_container_width=True, hide_index=True)

    # ══════════════════════════ MODULE · SETTINGS ══════════════════════════
    def render_settings():
        acc = "#555C66"
        _acc(acc)
        _hero("⚙️", "Settings",
              "the commercial numbers behind every money figure in this system", badge="CONFIG")
        if CONFIG_LOAD.get("source") == "file":
            st.success(f"Loaded from `{CONFIG_LOAD['path']}`")
        elif CONFIG_LOAD.get("source") == "secrets":
            st.success("Loaded from Streamlit secrets (`omc_config`)")
        else:
            st.error("**No config file was found — you are looking at empty defaults.** "
                     + (CONFIG_LOAD.get("error") or ""))
            with st.expander("Paths I looked in"):
                st.code("\n".join(CONFIG_LOAD.get("tried") or ["(none)"]))
                st.markdown(
                    "<span class='note'>Deployed on Streamlit Community Cloud? The file has to be "
                    "<b>committed to the repo</b> — uploading it to a running app does nothing, and "
                    "the container is rebuilt from git on every reboot. Put "
                    "<code>omc_config.json</code> beside this script in the repository, or paste its "
                    "contents into <b>Settings &rarr; Secrets</b> as "
                    "<code>omc_config = &#39;&#39;&#39;{ ... }&#39;&#39;&#39;</code>.</span>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns([1, 3])
        if c1.button("↻ Reload from disk", use_container_width=True, key="cfgreload"):
            st.session_state["omc_cfg"] = load_config()
            st.rerun()
        c2.caption("Nothing here is guessed — enter your own figures from the current pricing "
                   "window and the invoices you actually pay.")
        if config_is_ephemeral():
            st.warning("This looks like a hosted deployment with no persistent disk. **Saving "
                       "writes a file that will vanish on the next reboot.** Edit here, then use "
                       "*Download config* below and commit that file to your repo.")

        etabs = st.tabs(["Commercial model", "Price floors", "Sites", "Logistics & control",
                         "Company", "Raw JSON"])

        with etabs[0]:
            st.markdown("<div class='eyebrow'>Per-litre economics by grade</div>",
                        unsafe_allow_html=True)
            for rp in ("PMS", "AGO"):
                phead(rp)
                ec = cfg["economics"].setdefault(rp, {})
                c1, c2, c3 = st.columns(3)
                ec["dealer_margin"] = c1.number_input(
                    f"{rp} dealer margin (GHS/L)", value=float(ec.get("dealer_margin") or 0),
                    step=0.01, format="%.4f", key=f"dm_{rp}")
                ec["opex_per_litre"] = c2.number_input(
                    f"{rp} opex per litre (GHS/L)", value=float(ec.get("opex_per_litre") or 0),
                    step=0.01, format="%.4f", key=f"ox_{rp}",
                    help="Haulage, marking, site overhead and shrinkage allowance, per litre.")
                ec["uppf_recovery"] = c3.number_input(
                    f"{rp} UPPF recovery (GHS/L)", value=float(ec.get("uppf_recovery") or 0),
                    step=0.01, format="%.4f", key=f"up_{rp}",
                    help="Leave at zero unless you actually claim and receive it.")
                st.markdown(
                    f"<span class='note'>{rp} ex-depot cost is built from two tables: the "
                    "<b>base price</b> the BDC charges per litre, and the <b>duties and levies</b> "
                    "riding on it. They are kept apart because the base moves with every lifting "
                    "while the tax moves when policy moves — so a levy change is one row, not "
                    "hundreds.</span>", unsafe_allow_html=True)
                ec1, ec2 = st.columns(2)
                for col, key, label in ((ec1, "exdepot_base", f"{rp} base GHS/L"),
                                        (ec2, "tax", f"{rp} duties + levies GHS/L")):
                    with col:
                        cur = rate_table_frame(ec.get(key, {}))
                        if cur.empty:
                            cur = pd.DataFrame({"effective_from": pd.Series(dtype="datetime64[ns]"),
                                                "value": pd.Series(dtype=float)})
                        ed = st.data_editor(
                            cur, num_rows="dynamic", use_container_width=True,
                            key=f"{key}_{rp}", height=250,
                            column_config={
                                "effective_from": st.column_config.DateColumn("Effective from"),
                                "value": st.column_config.NumberColumn(label, format="%.4f")})
                        ec[key] = {pd.Timestamp(r["effective_from"]).strftime("%Y-%m-%d"):
                                   float(r["value"]) for _, r in ed.iterrows()
                                   if pd.notna(r.get("effective_from")) and pd.notna(r.get("value"))}
                res = resolved_exdepot(cfg, rp)
                if res:
                    rf = rate_table_frame(res).tail(6)
                    st.markdown(f"<span class='note'>Resolved {rp} ex-depot — base plus tax, most "
                                "recent entries:</span>", unsafe_allow_html=True)
                    st.dataframe(rf.assign(
                        effective_from=rf["effective_from"].dt.strftime("%d %b %Y")).rename(
                        columns={"effective_from": "From", "value": "Ex-depot GHS/L"})
                        .style.format({"Ex-depot GHS/L": "{:,.4f}"}),
                        use_container_width=True, hide_index=True)

        with etabs[1]:
            st.markdown("<div class='eyebrow'>NPA minimum pump price by window</div>",
                        unsafe_allow_html=True)
            st.markdown("<span class='note'>Enter the floor published for each pricing window, "
                        "dated from the day it takes effect. Published start dates slip — the "
                        "first August 2026 window ran from the 4th, not the 1st — so use the "
                        "date on the notice, not the convention.</span>", unsafe_allow_html=True)

            # ---- pull from the NPA site, review, then apply ----
            with st.expander("🌐 Pull from npa.gov.gh/price-floor", expanded=False):
                st.markdown(
                    "<span class='note'>The NPA publishes floors as a web page, not an API. "
                    "This tries to read it; if the page blocks the request or builds its table "
                    "in the browser, paste the notice text instead — the same parser reads it. "
                    "Nothing is written to your config until you press Apply.</span>",
                    unsafe_allow_html=True)
                fc1, fc2 = st.columns([1, 2])
                if fc1.button("🌐 Fetch now", use_container_width=True, key="npafetch"):
                    with st.spinner("Reading npa.gov.gh…"):
                        fr, (flo, fhi), note = fetch_npa_floors()
                    st.session_state["npa_stage"] = fr
                    st.session_state["npa_dates"] = (flo, fhi)
                    st.session_state["npa_note"] = note
                fc2.caption(f"Source: {NPA_FLOOR_URL} — override with the NPA_FLOOR_URL "
                            "environment variable if the address changes.")

                pasted = st.text_area(
                    "…or paste the notice text / page source here",
                    height=130, key="npapaste",
                    placeholder="Under the revised price floors, petrol will be sold at a "
                                "minimum of GH¢14.53 per litre, while diesel is pegged at "
                                "GH¢14.97 per litre…")
                if st.button("Read pasted text", key="npaparse") and pasted.strip():
                    fr = parse_npa_floors(pasted)
                    st.session_state["npa_stage"] = fr
                    st.session_state["npa_dates"] = npa_window_dates(pasted)
                    st.session_state["npa_note"] = (f"Read {len(fr)} product(s) from the pasted "
                                                    "text." if not fr.empty else
                                                    "Couldn't find any product and price in that "
                                                    "text. Type the two figures in below instead.")

                if st.session_state.get("npa_note"):
                    st.info(st.session_state["npa_note"])
                stage = st.session_state.get("npa_stage")
                if stage is not None and not stage.empty:
                    flo, fhi = st.session_state.get("npa_dates", (None, None))
                    eff = st.date_input(
                        "Effective from", value=(flo.date() if flo is not None else date.today()),
                        key="npaeff",
                        help="The day these floors take effect. Everything from this date until "
                             "the next entry is checked against them.")
                    if fhi is not None:
                        st.caption(f"Notice reads as running to {fhi:%d %b %Y}.")
                    st.dataframe(stage.rename(columns={
                        "product": "Product", "price": "Floor", "unit": "Unit",
                        "source": "Read from", "raw": "Matched text"}),
                        use_container_width=True, hide_index=True)
                    usable = stage[stage["product"].isin(["PMS", "AGO"])]
                    if usable.empty:
                        st.warning("Nothing here maps to PMS or AGO, which are the only two "
                                   "grades this system prices.")
                    elif st.button("✓ Apply these floors", key="npaapply",
                                   use_container_width=False):
                        key = pd.Timestamp(eff).strftime("%Y-%m-%d")
                        for _, r in usable.iterrows():
                            cfg.setdefault("floors", {}).setdefault(
                                r["product"], {})[key] = float(r["price"])
                        st.success(f"Staged {len(usable)} floor(s) effective {key}. "
                                   "Check the table below, then press Save settings.")
                        st.session_state["npa_stage"] = None
                        st.rerun()
                    st.caption("Read the matched text before applying. A floor that is wrong by "
                               "ten pesewas turns the compliance report into fiction, and this "
                               "parser is reading a page that was never designed to be read by "
                               "a machine.")

            for rp in ("PMS", "AGO"):
                phead(rp)
                cur = rate_table_frame((cfg.get("floors") or {}).get(rp, {}))
                if cur.empty:
                    cur = pd.DataFrame({"effective_from": pd.Series(dtype="datetime64[ns]"),
                                        "value": pd.Series(dtype=float)})
                fd = st.data_editor(cur, num_rows="dynamic", use_container_width=True,
                                    key=f"flr_{rp}",
                                    column_config={
                                        "effective_from": st.column_config.DateColumn("Window start"),
                                        "value": st.column_config.NumberColumn(
                                            "Floor GHS/L", format="%.4f")})
                cfg.setdefault("floors", {})[rp] = {
                    pd.Timestamp(r["effective_from"]).strftime("%Y-%m-%d"): float(r["value"])
                    for _, r in fd.iterrows()
                    if pd.notna(r.get("effective_from")) and pd.notna(r.get("value"))}

        with etabs[2]:
            st.markdown("<div class='eyebrow'>Site register</div>", unsafe_allow_html=True)
            st.markdown("<span class='note'>Tank capacity drives ullage and order quantities. "
                        "Monthly fixed cost drives break-even. Lead time drives the reorder "
                        "point.</span>", unsafe_allow_html=True)
            reg = pd.DataFrame([dict(station=s, **station_cfg(cfg, s)) for s in stations])
            ed = st.data_editor(reg, use_container_width=True, key="sitereg",
                                disabled=["station"],
                                column_config={
                                    "station": st.column_config.TextColumn("Station"),
                                    "PMS_capacity": st.column_config.NumberColumn(
                                        "PMS tank (L)", format="%.0f"),
                                    "AGO_capacity": st.column_config.NumberColumn(
                                        "AGO tank (L)", format="%.0f"),
                                    "opex_month": st.column_config.NumberColumn(
                                        "Fixed cost / month (GHS)", format="%.0f"),
                                    "lead_time_days": st.column_config.NumberColumn(
                                        "Lead time (days)", format="%.1f"),
                                    "safety_days": st.column_config.NumberColumn(
                                        "Safety stock (days)", format="%.1f"),
                                    "region": st.column_config.TextColumn("Region"),
                                    "dealer": st.column_config.TextColumn("Dealer")})
            def _cast(v):
                if isinstance(v, str) or v is None:
                    return v or ""
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return str(v)
            cfg["stations"] = {str(r["station"]): {k: _cast(v) for k, v in r.items()
                                                   if k != "station"}
                               for _, r in ed.iterrows()}

        with etabs[3]:
            st.markdown("<div class='eyebrow'>Haulage and control limits</div>",
                        unsafe_allow_html=True)
            log = cfg.setdefault("logistics", {})
            c1, c2, c3 = st.columns(3)
            sizes = c1.text_input("BRV compartment sizes (L, comma separated)",
                                  ", ".join(str(int(x)) for x in (log.get("brv_sizes") or [])),
                                  key="brv")
            log["brv_sizes"] = [float(x.strip()) for x in sizes.split(",")
                                if x.strip().replace(".", "").isdigit()]
            log["min_drop"] = c2.number_input("Minimum economic drop (L)",
                                              value=float(log.get("min_drop") or 0), step=500.0,
                                              key="mindrop")
            log["ullage_reserve_pct"] = c3.number_input(
                "Ullage reserve (%)", value=float(log.get("ullage_reserve_pct") or 0),
                step=1.0, key="ullres",
                help="Headroom never planned into a load, for expansion and dip error.")
            ctlc = cfg.setdefault("control", {})
            ctlc["tolerance_pct"] = st.number_input(
                "Wetstock tolerance (% of throughput)",
                value=float(ctlc.get("tolerance_pct") or TOLERANCE_PCT), step=0.05,
                format="%.2f", key="tolpct",
                help="Cumulative variance beyond this band is treated as a control failure "
                     "rather than measurement noise.")

        with etabs[4]:
            comp = cfg.setdefault("company", {})
            c1, c2 = st.columns(2)
            comp["name"] = c1.text_input("Company name", value=comp.get("name", ""), key="cname")
            comp["npa_licence"] = c2.text_input("NPA licence no.",
                                                value=comp.get("npa_licence", ""), key="clic")
            comp["supplier_bdc"] = st.text_input("Supplying BDC",
                                                 value=comp.get("supplier_bdc", ""), key="cbdc")
            st.markdown("<div class='eyebrow'>National monthly consumption (optional)</div>",
                        unsafe_allow_html=True)
            st.markdown("<span class='note'>Enter NPA industry volumes to see our share of the "
                        "market by month.</span>", unsafe_allow_html=True)
            mk = (cfg.setdefault("market", {}).setdefault("national_monthly_litres", {}))
            mrows = pd.DataFrame([{"month": k, "litres": v} for k, v in sorted(mk.items())]) \
                if mk else pd.DataFrame({"month": pd.Series(dtype=str),
                                         "litres": pd.Series(dtype=float)})
            med = st.data_editor(mrows, num_rows="dynamic", use_container_width=True, key="mktvol",
                                 column_config={
                                     "month": st.column_config.TextColumn("Month (YYYY-MM)"),
                                     "litres": st.column_config.NumberColumn("National litres",
                                                                             format="%.0f")})
            cfg["market"]["national_monthly_litres"] = {
                str(r["month"]): float(r["litres"]) for _, r in med.iterrows()
                if pd.notna(r.get("month")) and pd.notna(r.get("litres"))}
            share = compute_market_share(df, cfg)
            if not share.empty:
                fig = px.line(share, x="month", y="share_pct", markers=True,
                              title="Our share of national volume (%)")
                fig.update_traces(line=dict(color=acc, width=3))
                st.plotly_chart(style_fig(fig, 300, acc), use_container_width=True)

        with etabs[5]:
            st.markdown("<div class='eyebrow'>Everything, as stored</div>", unsafe_allow_html=True)
            st.code(json.dumps(cfg, indent=2, sort_keys=True), language="json")

        st.divider()
        c1, cD, c2 = st.columns([1, 1, 2])
        cD.download_button("⬇️ Download config", config_bytes(cfg),
                           file_name="omc_config.json", mime="application/json",
                           use_container_width=True, key="cfgdl")
        if c1.button("💾 Save settings", use_container_width=True, key="savecfg"):
            try:
                p = save_config(cfg)
                st.session_state["omc_cfg"] = cfg
                st.success(f"Saved to {p}. Every money figure in the system now uses these.")
            except Exception as e:
                st.error(f"Couldn't write the config file: {e}")
        c2.caption("Changes apply to this session immediately; save to keep them for next time.")

    # ═══════════════════════════ module dispatch ═══════════════════════════
    MODULES = {
        "⛽ Stocks & Sales": None,
        "💰 Margins": render_margins,
        "🏷️ Pricing": render_pricing,
        "🚚 Supply": render_supply,
        "🛡️ Control": render_control,
        "🏪 Site card": render_site,
        "🏦 Banking": render_banking,
        "⚙️ Settings": render_settings,
    }
    module = st.sidebar.radio("Module", list(MODULES), key="module")
    st.sidebar.divider()
    if MODULES[module] is not None:
        MODULES[module]()
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
                    "Efficiency", "Variance", "Rankings", "🔮 Forecast", "🚨 Alerts", "Trends",
                    "💸 Money", "📄 Report"])

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

    # ============================ COST OF LOSSES ($) ============================
    with tabs[10]:
        st.markdown("<div class='eyebrow'>What your stock losses cost · current period</div>",
                    unsafe_allow_html=True)
        st.subheader("💸 Cost of losses")

        med_price = {rp: (float(df[df["product"] == rp]["price"].dropna().median())
                          if df[df["product"] == rp]["price"].notna().any() else np.nan)
                     for rp in real_products}

        def _price(station, rp):
            s = df[(df["station"] == station) & (df["product"] == rp)].dropna(subset=["price"])
            return float(s.sort_values("date")["price"].iloc[-1]) if len(s) else med_price.get(rp, np.nan)

        rows = []
        for rp in real_products:
            vv = compute_variance(df, rp, pd.DataFrame(), cur_s, cur_e, STANDARD[rp])
            for _, r in vv.iterrows():
                nv = r["dip_variance"]
                days = int(r["days"]) if not np.isnan(r["days"]) else 0
                loss_l = max(-nv, 0.0) if not np.isnan(nv) else 0.0
                price = _price(r["station"], rp)
                money = loss_l * price if not np.isnan(price) else np.nan
                annual = ((loss_l / days * 365 * price) if (days and not np.isnan(price)) else np.nan)
                rows.append({"station": r["station"], "product": rp, "loss_l": loss_l,
                             "price": price, "money": money, "annual": annual})
        md = pd.DataFrame(rows)
        tot_loss = md["loss_l"].sum()
        tot_money = md["money"].sum(skipna=True)
        tot_annual = md["annual"].sum(skipna=True)
        cb = compute_banking(banking_frame(df))
        unbanked = (float(cb["outstanding"].clip(lower=0).sum())
                    if (not cb.empty and cb["outstanding"].notna().any()) else 0.0)

        kpi_row([
            ("Litres lost", f0(tot_loss), "L", f"{fmt(cur_s)} → {fmt(cur_e)}", accent),
            ("Cost this period", f0(tot_money), "GHS", "litres lost × pump price", accent),
            ("Annualized loss", f0(tot_annual), "GHS/yr", "at the current loss rate", accent),
            ("Unbanked cash now", f0(unbanked), "GHS", "sales not yet deposited", "#2563EB"),
        ])

        if tot_loss <= 0:
            st.success("No net stock losses in this period — nothing is leaking. 🎉")
        else:
            worst = md.dropna(subset=["annual"]).sort_values("annual", ascending=False)
            wline = ""
            if len(worst) and worst.iloc[0]["annual"] > 0:
                w = worst.iloc[0]
                wline = (f" The biggest single leak is <b>{w['station']} {w['product']}</b> at about "
                         f"<b>GHS {w['annual']:,.0f}/year</b>.")
            st.markdown(
                f"<div class='summary'>💸 At the current rate, stock losses are costing roughly "
                f"<b>GHS {tot_money:,.0f}</b> this period — about <b>GHS {tot_annual:,.0f} a year</b>. "
                f"Recovering half would save ~<b>GHS {tot_annual/2:,.0f}/year</b>.{wline}"
                + (f" Separately, <b>GHS {unbanked:,.0f}</b> of cash is still sitting unbanked."
                   if unbanked > 0 else "") + "</div>", unsafe_allow_html=True)

            piv = md.pivot_table(index="station", values="money", columns="product",
                                 aggfunc="sum").fillna(0.0)
            piv["__t"] = piv.sum(axis=1)
            piv = piv[piv["__t"] > 0].sort_values("__t").drop(columns="__t")
            if not piv.empty:
                fig = go.Figure()
                for rp in real_products:
                    if rp in piv.columns:
                        fig.add_bar(y=piv.index, x=piv[rp], orientation="h", name=rp,
                                    marker_color=PCOL[rp],
                                    text=[f"{v:,.0f}" if v > 0 else "" for v in piv[rp]],
                                    textposition="inside", textfont=dict(color="white", size=10))
                fig.update_layout(barmode="stack",
                                  title="Cost of stock losses by station (GHS, this period)")
                fig.update_xaxes(title_text="GHS lost")
                st.plotly_chart(style_fig(fig, max(300, 36 * len(piv)), accent),
                                use_container_width=True)

            show = md[md["loss_l"] > 0].copy().sort_values("annual", ascending=False)
            show = show.rename(columns={
                "station": "Station", "product": "Product", "loss_l": "Litres lost",
                "price": "Price (GHS/L)", "money": "Cost this period (GHS)",
                "annual": "Annualized (GHS/yr)"})
            st.dataframe(show.style.format({
                "Litres lost": "{:,.1f}", "Price (GHS/L)": "{:,.2f}",
                "Cost this period (GHS)": "{:,.0f}", "Annualized (GHS/yr)": "{:,.0f}"}, na_rep="—"),
                use_container_width=True, hide_index=True)
        st.caption("Loss = net negative dip variance (gains and excluded delivery days don't count), "
                   "valued at each station's latest pump price. Annualized = current daily loss "
                   "rate × 365. Unbanked cash is shown alongside as the other pool of money at risk.")

    # ============================ REPORT ============================
    with tabs[11]:
        _lb = logo_bytes()
        if _lb:
            st.image(_lb, width=86)
        st.markdown("<div class='eyebrow'>Stakeholder report · PMS &amp; AGO · monthly</div>",
                    unsafe_allow_html=True)
        st.subheader("📄 Monthly performance report")
        st.caption("Ranks every station on the three pillars — Target vs Actual realized, "
                   "Variance, and Efficiency — for PMS (red) and AGO (green) separately, in one "
                   "file. Pick a month below; the report covers the whole network for that month, "
                   "regardless of the grade/station chosen in the sidebar.")

        # ---- month picker ----
        rmonths = pd.period_range(dmin.to_period("M"), dmax.to_period("M"), freq="M")
        rlabels = [p.strftime("%B %Y") for p in rmonths]
        rpick = st.selectbox("Report month", rlabels, index=len(rlabels) - 1, key="rep_month")
        rsel = rmonths[rlabels.index(rpick)]
        rep_cs = max(pd.Timestamp(rsel.start_time.date()), dmin)
        rep_ce = min(pd.Timestamp(rsel.end_time.date()), dmax)

        # ---- TARGET BASELINE: 1 Jan of the report year → last day of the month before ----
        rep_bs = pd.Timestamp(rsel.year, 1, 1)
        rep_be = pd.Timestamp(rsel.start_time.date()) - pd.Timedelta(days=1)
        jan_no_base = rep_be < rep_bs           # month of interest is January → no in-year baseline
        if jan_no_base:
            target_note = (f"No target basis: {rpick} is the first month of {rsel.year}, so there "
                           f"are no earlier {rsel.year} months to build a target from.")
        else:
            target_note = (f"Target = 2 × median of monthly sales from {fmt(rep_bs)} to "
                           f"{fmt(rep_be)} (1 Jan of the year → the month before the report month).")
        st.caption(f"Reporting on **{rpick}** ({fmt(rep_cs)} → {fmt(rep_ce)}).  {target_note}")

        # ---- compute both grades for the chosen month ----
        rmonth_df = df_all[(df_all["date"] >= rep_cs) & (df_all["date"] <= rep_ce)]
        per_product = {}
        for g in ("PMS", "AGO"):
            tg = compute_targets(df_all, g, rep_bs, rep_be, rep_cs, rep_ce)
            vr = compute_variance(df_all, g, pd.DataFrame(), rep_cs, rep_ce, STANDARD[g])
            ef = compute_efficiency(rmonth_df, g)
            tgt_r, var_r, eff_r = build_pillars(tg, vr, ef)
            per_product[g] = {"kpis": report_kpis(tgt_r, var_r, eff_r, STANDARD[g]),
                              "tgt": tgt_r, "var": var_r, "eff": eff_r, "std": STANDARD[g]}
        kP, kA = per_product["PMS"]["kpis"], per_product["AGO"]["kpis"]

        # ---- headline row (PMS red / AGO green) ----
        kpi_row([
            ("PMS realized", f0(kP["actual"]), "L",
             ("—" if np.isnan(kP["attainment"]) else f"{kP['attainment']:.0f}% · {kP['verdict']}"),
             PCOL["PMS"]),
            ("AGO realized", f0(kA["actual"]), "L",
             ("—" if np.isnan(kA["attainment"]) else f"{kA['attainment']:.0f}% · {kA['verdict']}"),
             PCOL["AGO"]),
            ("Total target", f0(kP["target"] + kA["target"]), "L", "PMS + AGO", INK),
            ("Flagged", f"{kP['flagged'] + kA['flagged']}", "", "outside ±10 L/day", INK),
        ])

        # ---- plain-English headline + summary (drive the PDF cover) ----
        _p = lambda k: ("—" if np.isnan(k["attainment"]) else f"{k['attainment']:.0f}%")
        tot_a, tot_t = kP["actual"] + kA["actual"], kP["target"] + kA["target"]
        ov = "" if tot_t <= 0 else f", {tot_a / tot_t * 100:.0f}% of target"
        headline = f"The network sold {f0(tot_a)} L in {rpick}{ov}."
        sb = [f"PMS sold {f0(kP['actual'])} L ({_p(kP)} of target) and AGO sold "
              f"{f0(kA['actual'])} L ({_p(kA)})."]
        _pv = lambda v: "—" if v is None or np.isnan(v) else f"{v:.0f}%"
        if kP["tgt_top"] and kA["tgt_top"]:
            sb.append(f"The strongest stations were {kP['tgt_top']} on PMS "
                      f"({_pv(kP['tgt_top_v'])}) and {kA['tgt_top']} on AGO "
                      f"({_pv(kA['tgt_top_v'])}); the ones to watch were "
                      f"{kP['tgt_bot']} and {kA['tgt_bot']}.")
        fl = kP["flagged"] + kA["flagged"]
        sb.append((f"All stations kept stock variance within the ±10 L/day standard."
                   if fl == 0 else
                   f"{fl} grade-station(s) breached the ±10 L/day variance standard "
                   f"(see the variance page)."))
        summary = " ".join(sb)
        st.markdown(f"<div class='summary'>📋 <b>{headline}</b> {summary}</div>",
                    unsafe_allow_html=True)

        # ---- on-screen ranking previews: PMS (red) then AGO (green), bold station names ----
        def rep_chart(g, key, valcol, mode):
            color = PCOL[g]
            d = per_product[g][key].dropna(subset=[valcol]).iloc[::-1]
            if d.empty:
                st.info(f"No {g} data for {rpick}.")
                return
            if mode == "target":
                txt = [f"{v:.0f}%" for v in d[valcol]]
            elif mode == "variance":
                txt = [f"{v:+.1f}" for v in d[valcol]]
            else:
                txt = [f"{v:.1f} d" for v in d[valcol]]
            ylab = [f"<b>{s}</b>" for s in d["station"]]      # bold station names
            fig = go.Figure(go.Bar(x=d[valcol], y=ylab, orientation="h",
                                   marker_color=color, text=txt, textposition="outside",
                                   textfont=dict(color=INK, size=11), cliponaxis=False))
            if mode == "target":
                fig.add_vline(x=100, line_dash="dash", line_color=INK, annotation_text="target")
            elif mode == "variance":
                fig.add_vline(x=0, line_color=INK)
                fig.add_vline(x=STANDARD[g], line_dash="dash", line_color=INK,
                              annotation_text=f"+{STANDARD[g]:.0f}")
                fig.add_vline(x=-STANDARD[g], line_dash="dash", line_color=INK,
                              annotation_text=f"-{STANDARD[g]:.0f}")
            fig.update_layout(title=f"{PLABEL[g]} — ranked",
                              font=dict(family="Candara, Calibri, Segoe UI, sans-serif"))
            st.plotly_chart(style_fig(fig, max(230, 30 * len(d)), color),
                            use_container_width=True)

        st.markdown("<div class='eyebrow'>1 · Target vs Actual realized</div>",
                    unsafe_allow_html=True)
        rep_chart("PMS", "tgt", "attainment_pct", "target")
        rep_chart("AGO", "tgt", "attainment_pct", "target")

        st.markdown("<div class='eyebrow'>2 · Variance · stock control</div>",
                    unsafe_allow_html=True)
        rep_chart("PMS", "var", "avg_daily_var", "variance")
        rep_chart("AGO", "var", "avg_daily_var", "variance")

        st.markdown("<div class='eyebrow'>3 · Efficiency · sell-through</div>",
                    unsafe_allow_html=True)
        rep_chart("PMS", "eff", "days_to_stockout", "efficiency")
        rep_chart("AGO", "eff", "days_to_stockout", "efficiency")

        # ---- downloads: PDF (both grades) + WhatsApp ----
        pstr = rsel.strftime("%Y_%m")
        meta = {
            "period": rpick,
            "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
            "cover_sub": f"{len(stations)} stations · {rpick} · whole network",
            "footer": f"Spartan Fuel Analytics · sheet {used_sheet} · GHS · confidential",
            "headline": headline,
            "summary": summary,
            "target_note": target_note,
            "_buffer": io.BytesIO(),
        }
        wa_text = build_whatsapp_text(meta, per_product)

        st.markdown("<div class='eyebrow'>PDF report · PMS &amp; AGO · with logo</div>",
                    unsafe_allow_html=True)
        try:
            pdf_bytes = build_report_pdf(meta, per_product)
            st.download_button("⬇ Download PDF report", pdf_bytes,
                               f"spartan_report_{pstr}.pdf", "application/pdf")
        except Exception as e:
            st.caption(f"PDF export unavailable ({e}). Install matplotlib &amp; pillow: "
                       "`pip install matplotlib pillow`.")

        st.markdown("<div class='eyebrow'>WhatsApp brief · copy &amp; send with the PDF</div>",
                    unsafe_allow_html=True)
        st.caption("Tap the copy icon (top-right of the box), paste into WhatsApp, then attach "
                   "the PDF above. The *asterisks* become bold in WhatsApp.")
        st.code(wa_text, language=None)
        st.download_button("⬇ Download WhatsApp text (.txt)", wa_text,
                           f"spartan_whatsapp_{pstr}.txt", "text/plain")


    st.caption("All figures from the MASTER sheet · prices in GHS · target = 2× median "
               "baseline month, measured against the current period.")


if __name__ == "__main__":
    main()