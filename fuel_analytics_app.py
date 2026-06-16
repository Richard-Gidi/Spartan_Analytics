"""
Spartan Fuel — Petroleum Marketing Analytics
=============================================
Reads the MASTER sheet of a Google Sheet (link supplied via a .env variable
called GOOGLE) and produces a marketing-analytics workbench for a fuel-retail
network. PMS = petrol (red), AGO = diesel (green).

Setup
-----
1) Create a file named  .env  next to this script containing:

       GOOGLE=https://docs.google.com/spreadsheets/d/<your-id>/edit?usp=sharing

   The sheet's link-sharing must be set to "Anyone with the link – Viewer".

2) Install and run:

       pip install -r requirements.txt
       streamlit run fuel_analytics_app.py

Only the MASTER tab is used. The target has its own date window; no other
analysis exposes date controls. The three tuning values below are fixed in
code (not shown in the dashboard).
"""

import io
import os
import re
import math
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

# ───────────────────────── fixed configuration (not shown in UI) ────────────
SHEET_NAME           = "MASTER"     # only this tab is analysed
RUNWAY_WINDOW        = 7            # days in the rolling sales average for runway
PRICE_EVENT_WINDOW   = 14           # days before/after a price change in the event study
RANK_W_ATTAIN        = 0.40         # ranking weight: target attainment
RANK_W_VOLUME        = 0.35         # ranking weight: throughput
RANK_W_DISCIPLINE    = 0.25         # ranking weight: stock discipline
EXCLUDE_ZERO         = True         # exclude stock-out / blank days from medians & elasticity

PCOL  = {"PMS": "#D62828", "AGO": "#1B7F4B"}      # PMS red, AGO green
PSOFT = {"PMS": "#FCE7E7", "AGO": "#E2F1E8"}
PLABEL = {"PMS": "PMS · Petrol", "AGO": "AGO · Diesel"}
SCALE = {"PMS": [[0, "#FBEAEA"], [1, "#D62828"]],
         "AGO": [[0, "#E6F2EA"], [1, "#1B7F4B"]]}


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
    """rows = list of lists from the MASTER sheet → tidy long DataFrame
    (one row per date / station / product)."""
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
            })
    df = pd.DataFrame(recs)
    if not df.empty:
        df = df.sort_values(["station", "product", "date"]).reset_index(drop=True)
    return df


# ──────────────────────────── google sheet loader ──────────────────────────
def gsheet_export_url(link):
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", link) or re.search(r"[?&]id=([a-zA-Z0-9-_]+)", link)
    if not m:
        raise ValueError("Couldn't find a spreadsheet ID in the GOOGLE link.")
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=xlsx"


def load_master(link):
    """Download the workbook from the share link and return (tidy_df, used_sheet)."""
    import requests
    url = gsheet_export_url(link)
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    content = r.content
    if content[:2] != b"PK":      # xlsx is a zip; an HTML login page is not
        raise PermissionError(
            "The sheet isn't publicly readable. In Google Sheets set "
            "Share → General access → 'Anyone with the link – Viewer'.")
    xls = pd.ExcelFile(io.BytesIO(content))
    used = next((s for s in xls.sheet_names if s.strip().upper() == SHEET_NAME),
                xls.sheet_names[0])
    raw = pd.read_excel(xls, sheet_name=used, header=None, dtype=object)
    return build_records(raw.values.tolist()), used


# ───────────────────────────── core analytics ──────────────────────────────
def _slice(df, product, start, end):
    m = (df["product"] == product) & (df["date"] >= start) & (df["date"] <= end)
    return df.loc[m]


def selling_vols(s, exclude_zero=EXCLUDE_ZERO):
    v = s["volume"].dropna()
    return v[v > 0] if exclude_zero else v


def compute_targets(df, product, base_start, base_end, cur_start, cur_end,
                    exclude_zero=EXCLUDE_ZERO):
    base = _slice(df, product, base_start, base_end)
    cur = _slice(df, product, cur_start, cur_end)
    out = []
    for st in sorted(df["station"].unique()):
        bs = base[base["station"] == st]
        cs = cur[cur["station"] == st]
        vb = selling_vols(bs, exclude_zero)
        median_daily = float(np.median(vb)) if len(vb) else np.nan
        daily_target = median_daily * 2 if not np.isnan(median_daily) else np.nan
        cv = cs["volume"].dropna()
        cur_days = int(cv.shape[0])
        actual_total = float(cv.sum()) if cur_days else 0.0
        target_total = daily_target * cur_days if not np.isnan(daily_target) else np.nan
        attainment = (actual_total / target_total * 100
                      if target_total and not np.isnan(target_total) and target_total > 0
                      else np.nan)
        gap = actual_total - target_total if not np.isnan(target_total) else np.nan
        out.append({
            "station": st, "base_days": int(len(vb)),
            "median_daily": median_daily, "daily_target": daily_target,
            "cur_days": cur_days, "actual_total": actual_total,
            "target_total": target_total, "attainment_pct": attainment,
            "gap_litres": gap,
        })
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
            arc = dQ = np.nan
            if not np.isnan(qb) and not np.isnan(qa) and (qa + qb) > 0:
                dQ = (qa - qb) / ((qa + qb) / 2)
                arc = dQ / dP if dP != 0 else np.nan
            rows.append({
                "date": d, "old_price": prev, "new_price": row["price"],
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
        if np.isnan(runway):
            risk = "no estimate"
        elif runway < 1.5:
            risk = "critical"
        elif runway < 3:
            risk = "low"
        elif runway < 6:
            risk = "watch"
        else:
            risk = "healthy"
        out.append({"station": st, "as_of": last["date"], "stock_litres": stock,
                    "stock_source": src, "avg_daily_sales": avg,
                    "days_to_run_out": runway, "risk": risk})
    res = pd.DataFrame(out)
    if not res.empty:
        res = res.sort_values("days_to_run_out", na_position="last").reset_index(drop=True)
    return res


def compute_variance(df, product, targets_df, cur_start, cur_end):
    cur = _slice(df, product, cur_start, cur_end)
    tmap = targets_df.set_index("station") if not targets_df.empty else None
    out = []
    for st in sorted(df["station"].unique()):
        cs = cur[cur["station"] == st]
        throughput = float(cs["volume"].dropna().sum())
        dip_var = float(cs["dip_var"].dropna().sum())
        shortage = float(cs["shortage"].dropna().sum())
        tgt_var = tgt_var_pct = np.nan
        if tmap is not None and st in tmap.index:
            tgt_var = tmap.loc[st, "gap_litres"]
            tt = tmap.loc[st, "target_total"]
            if tt and not np.isnan(tt) and tt != 0:
                tgt_var_pct = tgt_var / tt * 100
        loss_pct = (dip_var / throughput * 100) if throughput else np.nan
        out.append({"station": st, "throughput": throughput,
                    "target_variance": tgt_var, "target_var_pct": tgt_var_pct,
                    "dip_variance": dip_var, "stock_loss_pct": loss_pct,
                    "delivery_shortage": shortage})
    return pd.DataFrame(out)


def _minmax(series):
    s = series.astype(float)
    lo, hi = np.nanmin(s.values), np.nanmax(s.values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo) * 100


def compute_rankings(targets_df, variance_df,
                     w_attain=RANK_W_ATTAIN, w_volume=RANK_W_VOLUME,
                     w_disc=RANK_W_DISCIPLINE):
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


def analyst_summary(product, targets_df, runway_df, rankings_df):
    if targets_df.empty:
        return "No data in the selected windows."
    tot_a = targets_df["actual_total"].sum()
    tot_t = targets_df["target_total"].sum(skipna=True)
    overall = tot_a / tot_t * 100 if tot_t else np.nan
    bits, pl = [], PLABEL[product]
    if not np.isnan(overall):
        verdict = ("ahead of" if overall >= 100 else
                   "tracking toward" if overall >= 75 else "behind")
        bits.append(f"Network {pl} sold <b>{tot_a:,.0f} L</b> against a target of "
                    f"<b>{tot_t:,.0f} L</b> — <b>{overall:.0f}%</b> of target, {verdict} plan.")
    if not rankings_df.empty:
        top, bot = rankings_df.iloc[0], rankings_df.iloc[-1]
        bits.append(f"Strongest station: <b>{top['station']}</b> · weakest: "
                    f"<b>{bot['station']}</b>.")
    if not runway_df.empty:
        crit = runway_df[runway_df["risk"].isin(["critical", "low"])]
        if len(crit):
            bits.append(f"⚠ <b>{len(crit)}</b> station(s) under ~3 days of cover "
                        f"({', '.join(crit['station'].head(4))}) — replenish first.")
        else:
            bits.append("All stations hold healthy stock cover.")
    return " ".join(bits)


def default_windows(dmin, dmax):
    """Baseline = Jan 1 (this year) → last day of previous month.
    Current   = first day of this month → latest date in sheet.

    If the sheet hasn't reached the current month yet (stale data), the rule is
    anchored on the latest month present so the defaults still land on a useful
    period. Everything is clamped to the available data range. Editable in the UI.
    """
    today = date.today()
    anchor = today if date(today.year, today.month, 1) <= dmax.date() else dmax.date()
    fom = date(anchor.year, anchor.month, 1)          # first of the (anchor) month
    prev_end = fom - timedelta(days=1)                # last day of previous month
    ys = date(anchor.year, 1, 1)                       # beginning of the year
    lo, hi = dmin.date(), dmax.date()
    cl = lambda d: min(max(d, lo), hi)
    bs, be = cl(ys), cl(prev_end)
    cs, ce = cl(fom), hi
    if bs > be:
        bs, be = lo, hi
    if cs > ce:
        cs = ce
    return (bs, be), (cs, ce)


# ─────────────────────────────────── theme ─────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--ink:#16191F;--muted:#6B6F76;--faint:#9A9DA3;--line:#E7E4DB;--paper:#F4F2EC;}
.stApp{background:var(--paper);}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1280px;}
h1,h2,h3,h4{font-family:'Sora',sans-serif;color:var(--ink);letter-spacing:-.01em;}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
.hero{background:linear-gradient(125deg,#13171E 0%,#202B33 100%);border-radius:20px;
      padding:26px 30px;color:#fff;box-shadow:0 14px 40px rgba(20,24,31,.18);margin-bottom:16px;}
.hero h1{color:#fff;font-size:25px;margin:0 0 4px;font-weight:700;}
.hero .meta{color:#9aa1aa;font-family:'IBM Plex Mono',monospace;font-size:12.5px;}
.hero .badge{display:inline-block;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
      color:#fff;border-radius:6px;padding:3px 9px;font-family:'IBM Plex Mono',monospace;
      font-size:11px;letter-spacing:.12em;margin-left:8px;vertical-align:middle;}
.summary{background:#fff;border-radius:14px;padding:16px 20px;line-height:1.65;font-size:15px;
      color:#2A2E35;box-shadow:0 8px 24px rgba(20,24,31,.06);border-left:4px solid var(--acc);margin-bottom:18px;}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:2px 0 8px;}
.kpi{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px 18px 16px;
      position:relative;overflow:hidden;box-shadow:0 8px 24px rgba(20,24,31,.05);}
.kpi .l{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.11em;
      text-transform:uppercase;color:var(--faint);}
.kpi .v{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:29px;color:var(--ink);
      margin-top:9px;line-height:1;font-variant-numeric:tabular-nums;}
.kpi .v .u{font-size:13px;color:var(--faint);margin-left:4px;font-weight:500;}
.kpi .s{font-size:12px;color:var(--muted);margin-top:8px;min-height:15px;}
.kpi .tick{position:absolute;left:0;bottom:0;height:4px;width:100%;}
.eyebrow{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.14em;
      text-transform:uppercase;color:var(--acc);font-weight:600;margin:6px 0 -4px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:9px 16px;border-radius:10px 10px 0 0;
      font-family:'Sora',sans-serif;font-weight:600;font-size:13.5px;color:var(--muted);}
.stTabs [aria-selected="true"]{background:var(--ink);color:#fff!important;}
[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums;}
[data-testid="stMetricLabel"]{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.08em;text-transform:uppercase;}
.note{color:var(--faint);font-size:12px;font-family:'IBM Plex Mono',monospace;}
</style>
"""


def style_fig(fig, height=340, accent="#16191F"):
    fig.update_layout(
        height=height, margin=dict(l=12, r=12, t=48, b=12),
        font=dict(family="Inter, sans-serif", size=12, color="#3A3F47"),
        title=dict(font=dict(family="Sora, sans-serif", size=15, color="#16191F")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.14, x=1, xanchor="right", font=dict(size=11)),
        colorway=[accent, "#9A9DA3", "#C5821C", "#3A6EA5"])
    fig.update_xaxes(gridcolor="#ECEAE3", zeroline=False, linecolor="#DAD6CC")
    fig.update_yaxes(gridcolor="#ECEAE3", zeroline=False, linecolor="#DAD6CC")
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
        st.markdown(CSS, unsafe_allow_html=True)
        st.title("⛽ Spartan Fuel — Marketing Analytics")
        st.error("No data source found. Create a **.env** file next to this app with:\n\n"
                 "`GOOGLE=https://docs.google.com/spreadsheets/d/<your-id>/edit?usp=sharing`\n\n"
                 "and set the sheet's sharing to *Anyone with the link – Viewer*.")
        st.stop()

    @st.cache_data(ttl=600, show_spinner="Reading the MASTER sheet from Google…")
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

    stations = sorted(df["station"].unique())
    dmin, dmax = df["date"].min(), df["date"].max()
    (bs_def, be_def), (cs_def, ce_def) = default_windows(dmin, dmax)

    # ───── sidebar : only product, station, and the TARGET window ─────
    with st.sidebar:
        st.markdown("#### View")
        product = st.radio("Fuel grade", ["PMS", "AGO"],
                           format_func=lambda p: PLABEL[p], horizontal=True)
        focus = st.selectbox("Station focus", ["All stations"] + stations)
        st.divider()
        st.markdown("#### Target window")
        st.markdown("<span class='note'>median × 2 over the baseline = the daily target; "
                    "actual total is measured over the current period.</span>",
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

    # ───── compute ─────
    targets = compute_targets(df, product, base_s, base_e, cur_s, cur_e)
    runway = compute_runway(df, product, dmax)              # as-of latest date (hidden window)
    variance = compute_variance(df, product, targets, cur_s, cur_e)
    rankings = compute_rankings(targets, variance)

    # ───── hero + summary ─────
    fmt = lambda d: pd.Timestamp(d).strftime("%d %b %Y")
    st.markdown(
        f"<div class='hero'><h1>⛽ Spartan Fuel — Marketing Analytics"
        f"<span class='badge'>{used_sheet}</span></h1>"
        f"<div class='meta'>{len(stations)} stations · {fmt(dmin)} → {fmt(dmax)} · "
        f"viewing {PLABEL[product]} · target baseline {fmt(base_s)}→{fmt(base_e)} · "
        f"current {fmt(cur_s)}→{fmt(cur_e)}</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='summary' style='--acc:{accent}'>📋 "
                f"{analyst_summary(product, targets, runway, rankings)}</div>",
                unsafe_allow_html=True)

    def kpi_row(items):
        cards = "".join(
            f"<div class='kpi'><div class='l'>{l}</div>"
            f"<div class='v'>{v}<span class='u'>{u}</span></div>"
            f"<div class='s'>{s}</div><div class='tick' style='background:{a}'></div></div>"
            for l, v, u, s, a in items)
        st.markdown(f"<div class='kpi-row'>{cards}</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Targets vs Actual", "Price Sensitivity",
                    "Days to Run Out", "Variance", "Rankings", "Trends"])

    # ============================ OVERVIEW ============================
    with tabs[0]:
        if focus == "All stations":
            row = None
            tot_a = targets["actual_total"].sum()
            tot_t = targets["target_total"].sum(skipna=True)
            med = float(np.nanmedian(targets["median_daily"].values)) if len(targets) else np.nan
            dtgt = float(np.nansum(targets["daily_target"].values))
            ctx = "All stations"
        else:
            row = targets[targets["station"] == focus]
            tot_a = float(row["actual_total"].iloc[0]) if len(row) else 0
            tot_t = float(row["target_total"].iloc[0]) if len(row) else np.nan
            med = float(row["median_daily"].iloc[0]) if len(row) else np.nan
            dtgt = float(row["daily_target"].iloc[0]) if len(row) else np.nan
            ctx = focus
        att = tot_a / tot_t * 100 if tot_t and not np.isnan(tot_t) and tot_t > 0 else np.nan

        st.markdown(f"<div class='eyebrow'>{ctx} · {PLABEL[product]}</div>", unsafe_allow_html=True)
        f0 = lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.0f}"
        kpi_row([
            ("Actual total sold", f0(tot_a), "L", "in the current period", accent),
            ("Target total", f0(tot_t), "L", "2× median × operating days", "#16191F"),
            ("Attainment", "—" if np.isnan(att) else f"{att:,.0f}", "%",
             status_label(att), accent),
            ("Daily target", f0(dtgt), "L/day", "twice the baseline median", accent),
        ])

        st.write("")
        g1, g2 = st.columns([1, 1.45], gap="large")
        with g1:
            gv = 0 if np.isnan(att) else att
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=gv, number={"suffix": "%", "font": {"size": 38}},
                gauge={"axis": {"range": [0, max(150, gv + 15)]},
                       "bar": {"color": accent, "thickness": 0.28},
                       "bgcolor": "#F4F2EC",
                       "steps": [{"range": [0, 75], "color": "#EFEAE2"},
                                 {"range": [75, 100], "color": PSOFT[product]}],
                       "threshold": {"line": {"color": "#16191F", "width": 3},
                                     "thickness": 0.85, "value": 100}}))
            st.plotly_chart(style_fig(gauge, 300, accent), use_container_width=True)
            st.markdown("<div class='note' style='text-align:center'>100% = hitting twice the "
                        "median every operating day</div>", unsafe_allow_html=True)
        with g2:
            top = targets.dropna(subset=["actual_total"]).head(12).iloc[::-1]
            fig = go.Figure()
            fig.add_bar(y=top["station"], x=top["target_total"], orientation="h",
                        name="Target", marker_color="rgba(22,25,31,.18)")
            fig.add_bar(y=top["station"], x=top["actual_total"], orientation="h",
                        name="Actual", marker_color=accent)
            fig.update_layout(barmode="overlay", title="Actual vs target by station")
            st.plotly_chart(style_fig(fig, 320, accent), use_container_width=True)

        st.write("")
        share = targets[targets["actual_total"] > 0]
        if not share.empty:
            st.markdown("<div class='eyebrow'>Volume share</div>", unsafe_allow_html=True)
            donut = px.pie(share, names="station", values="actual_total", hole=0.58)
            donut.update_traces(textposition="outside", textinfo="label+percent",
                                marker=dict(line=dict(color="#F4F2EC", width=2)))
            donut.update_layout(showlegend=False,
                                colorway=px.colors.sequential.Reds[2:] if product == "PMS"
                                else px.colors.sequential.Greens[2:])
            st.plotly_chart(style_fig(donut, 360, accent), use_container_width=True)

    # ============================ TARGETS ============================
    with tabs[1]:
        st.markdown("<div class='eyebrow'>Target = twice the baseline median</div>",
                    unsafe_allow_html=True)
        st.subheader("Actual total sold vs target total")
        view = targets.copy()
        view["status"] = view["attainment_pct"].apply(status_label)
        show = view.rename(columns={
            "station": "Station", "base_days": "Baseline days",
            "median_daily": "Median daily (L)", "daily_target": "Daily target (L)",
            "cur_days": "Operating days", "actual_total": "Actual total (L)",
            "target_total": "Target total (L)", "attainment_pct": "Attainment %",
            "gap_litres": "Gap (L)", "status": "Status"})
        st.dataframe(show.style.format({
            "Median daily (L)": "{:,.0f}", "Daily target (L)": "{:,.0f}",
            "Actual total (L)": "{:,.0f}", "Target total (L)": "{:,.0f}",
            "Attainment %": "{:,.0f}%", "Gap (L)": "{:,.0f}"}, na_rep="—"),
            use_container_width=True, hide_index=True,
            height=min(560, 80 + 36 * len(show)))
        att = targets.dropna(subset=["attainment_pct"]).sort_values("attainment_pct").iloc[::-1]
        if not att.empty:
            fig = px.bar(att.iloc[::-1], x="attainment_pct", y="station", orientation="h",
                         labels={"attainment_pct": "Attainment %", "station": ""},
                         title="Attainment % by station")
            fig.update_traces(marker_color=[accent if a >= 100 else
                                            (PSOFT[product] if a >= 75 else "#C9C5BC")
                                            for a in att.iloc[::-1]["attainment_pct"]])
            fig.add_vline(x=100, line_dash="dash", line_color="#16191F",
                          annotation_text="target")
            st.plotly_chart(style_fig(fig, max(280, 32 * len(att)), accent),
                            use_container_width=True)
        st.download_button("⬇ Download targets (CSV)", show.to_csv(index=False),
                           f"targets_{product}.csv", "text/csv")

    # ============================ PRICE SENSITIVITY (full history) ============================
    with tabs[2]:
        st.markdown("<div class='eyebrow'>Across all available history</div>",
                    unsafe_allow_html=True)
        if focus == "All stations":
            st.info("Choose a single **Station focus** in the sidebar — elasticity needs "
                    "one station's price history.")
        else:
            el = elasticity(df, focus, product, dmin, dmax)
            st.subheader(f"{focus} · {PLABEL[product]} — price sensitivity")
            if np.isnan(el["elasticity"]):
                st.warning("This station shows no price variation in the data, so elasticity "
                           "can't be estimated yet.")
            else:
                E, absE = el["elasticity"], abs(el["elasticity"])
                lab = ("Elastic demand" if absE >= 1.1 else
                       "Inelastic demand" if absE <= 0.9 else "Near unit-elastic")
                kpi_row([
                    ("Price elasticity", f"{E:.2f}", "", lab, accent),
                    ("Litres per +GHS 0.10",
                     "—" if np.isnan(el["per_10pesewa"]) else f"{el['per_10pesewa']:,.0f}",
                     "L", "linear sensitivity", "#16191F"),
                    ("Model fit", "—" if np.isnan(el["r2"]) else f"{el['r2']*100:.0f}", "%",
                     "log-log R²", accent),
                    ("Price levels", f"{el['n_prices']}", "", "distinct prices seen", accent),
                ])
                desc = ("Buyers here react strongly to price." if absE >= 1.1 else
                        "Volume holds up well against price." if absE <= 0.9 else
                        "Volume moves about 1-for-1 with price.")
                st.caption(desc)

            st.write("")
            c1, c2 = st.columns([1.25, 1], gap="large")
            with c1:
                s = df[(df["product"] == product) & (df["station"] == focus)].dropna(
                    subset=["price", "volume"])
                s = s[(s["price"] > 0) & (s["volume"] > 0)]
                if not s.empty:
                    fig = px.scatter(s, x="price", y="volume",
                                     labels={"price": "Price (GHS/L)", "volume": "Volume (L/day)"},
                                     title="Daily volume vs price — demand curve")
                    fig.update_traces(marker=dict(color=accent, size=8, opacity=0.65,
                                                  line=dict(width=0)))
                    m, b = np.polyfit(s["price"], s["volume"], 1)
                    xs = np.array([s["price"].min(), s["price"].max()])
                    fig.add_scatter(x=xs, y=m * xs + b, mode="lines", name="Trend",
                                    line=dict(color="#16191F", dash="dash", width=2))
                    st.plotly_chart(style_fig(fig, 360, accent), use_container_width=True)
            with c2:
                pl = price_levels(df, focus, product, dmin, dmax)
                if not pl.empty:
                    fig = go.Figure()
                    for _, r in pl.iterrows():
                        fig.add_shape(type="line", x0=r["avg_daily"], x1=r["avg_daily"],
                                      y0=0, y1=0, line=dict(width=0))
                    fig.add_trace(go.Scatter(
                        x=pl["avg_daily"], y=pl["price"].astype(str), mode="markers",
                        marker=dict(color=accent, size=13),
                        error_x=dict(type="data", array=[0] * len(pl))))
                    for _, r in pl.iterrows():
                        fig.add_shape(type="line", x0=0, x1=r["avg_daily"],
                                      y0=str(r["price"]), y1=str(r["price"]),
                                      line=dict(color="#D8D4CB", width=2))
                    fig.update_layout(title="Avg daily volume by price level",
                                      xaxis_title="Avg daily (L)", yaxis_title="Price (GHS/L)")
                    st.plotly_chart(style_fig(fig, 360, accent), use_container_width=True)

            # price vs volume over time (dual axis)
            s2 = df[(df["product"] == product) & (df["station"] == focus)].sort_values("date")
            if not s2.empty and s2["price"].notna().any():
                st.markdown("<div class='eyebrow'>Price steps vs daily volume</div>",
                            unsafe_allow_html=True)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_bar(x=s2["date"], y=s2["volume"], name="Volume",
                            marker_color=PSOFT[product], secondary_y=False)
                fig.add_scatter(x=s2["date"], y=s2["price"], name="Price",
                                mode="lines", line=dict(color=accent, width=2.5, shape="hv"),
                                secondary_y=True)
                fig.update_yaxes(title_text="Volume (L/day)", secondary_y=False)
                fig.update_yaxes(title_text="Price (GHS/L)", secondary_y=True)
                st.plotly_chart(style_fig(fig, 320, accent), use_container_width=True)

            ev = price_events(df, focus, product, dmin, dmax)
            st.markdown("<div class='eyebrow'>Price-change event study</div>",
                        unsafe_allow_html=True)
            if focus != "All stations":
                if ev.empty:
                    st.caption("No price changes recorded for this station.")
                else:
                    evs = ev.rename(columns={
                        "date": "Date", "old_price": "Old", "new_price": "New",
                        "price_chg_pct": "Price Δ%", "avg_before": "Avg before (L)",
                        "avg_after": "Avg after (L)", "vol_chg_pct": "Volume Δ%",
                        "arc_elasticity": "Arc elasticity"})
                    evs["Date"] = pd.to_datetime(evs["Date"]).dt.strftime("%d %b %Y")
                    st.dataframe(evs.style.format({
                        "Old": "{:,.2f}", "New": "{:,.2f}", "Price Δ%": "{:+.1f}%",
                        "Avg before (L)": "{:,.0f}", "Avg after (L)": "{:,.0f}",
                        "Volume Δ%": "{:+.1f}%", "Arc elasticity": "{:.2f}"}, na_rep="—"),
                        use_container_width=True, hide_index=True)

    # ============================ RUNWAY ============================
    with tabs[3]:
        st.markdown("<div class='eyebrow'>Stock cover · rolling-average method</div>",
                    unsafe_allow_html=True)
        st.subheader(f"Days to run out — {PLABEL[product]}")
        st.caption(f"Latest tank stock ÷ {RUNWAY_WINDOW}-day rolling-average daily sales. "
                   "Physical dip is used where available, else book closing stock.")
        if runway.empty:
            st.info("No stock readings detected for this product.")
        else:
            colormap = {"critical": "#7A0010", "low": "#D62828", "watch": "#C5821C",
                        "healthy": "#1B7F4B", "no estimate": "#B7B4AC"}
            plotr = runway.dropna(subset=["days_to_run_out"]).iloc[::-1]
            if not plotr.empty:
                fig = px.bar(plotr, x="days_to_run_out", y="station", orientation="h",
                             color="risk", color_discrete_map=colormap,
                             labels={"days_to_run_out": "Days of cover", "station": "",
                                     "risk": ""}, title="Stock cover by station")
                fig.add_vline(x=3, line_dash="dot", line_color="#16191F",
                              annotation_text="3-day floor")
                st.plotly_chart(style_fig(fig, max(280, 34 * len(plotr)), accent),
                                use_container_width=True)
            rv = runway.copy()
            rv["as_of"] = pd.to_datetime(rv["as_of"]).dt.strftime("%d %b %Y")
            show = rv.rename(columns={
                "station": "Station", "as_of": "As of", "stock_litres": "Stock (L)",
                "stock_source": "Stock source", "avg_daily_sales": "Avg daily sales (L)",
                "days_to_run_out": "Days to run out", "risk": "Risk"})
            st.dataframe(show.style.format({
                "Stock (L)": "{:,.0f}", "Avg daily sales (L)": "{:,.0f}",
                "Days to run out": "{:,.1f}"}, na_rep="—"),
                use_container_width=True, hide_index=True)

    # ============================ VARIANCE ============================
    with tabs[4]:
        st.markdown("<div class='eyebrow'>Plan, stock & delivery control</div>",
                    unsafe_allow_html=True)
        st.subheader(f"Variance analysis — {PLABEL[product]}")
        st.caption("Target variance (actual − target), dip variance (physical vs book stock), "
                   "and tanker delivery shortage / overage — over the current period.")
        if variance.empty:
            st.info("No variance data in this window.")
        else:
            vv = variance.copy()
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig = px.bar(vv, x="target_variance", y="station", orientation="h",
                             title="Target variance (L)",
                             labels={"target_variance": "Litres vs target", "station": ""})
                fig.update_traces(marker_color=[accent if x >= 0 else "#C9C5BC"
                                                for x in vv["target_variance"].fillna(0)])
                fig.add_vline(x=0, line_color="#16191F")
                st.plotly_chart(style_fig(fig, max(280, 32 * len(vv)), accent),
                                use_container_width=True)
            with c2:
                fig = px.bar(vv, x="dip_variance", y="station", orientation="h",
                             title="Cumulative dip variance (L) — stock control",
                             labels={"dip_variance": "Physical − book (L)", "station": ""})
                fig.update_traces(marker_color=["#7A0010" if abs(x) > 0 else accent
                                                for x in vv["dip_variance"].fillna(0)])
                fig.add_vline(x=0, line_color="#16191F")
                st.plotly_chart(style_fig(fig, max(280, 32 * len(vv)), accent),
                                use_container_width=True)
            show = vv.rename(columns={
                "station": "Station", "throughput": "Throughput (L)",
                "target_variance": "Target variance (L)", "target_var_pct": "Target var %",
                "dip_variance": "Dip variance (L)", "stock_loss_pct": "Stock loss %",
                "delivery_shortage": "Delivery shortage (L)"})
            st.dataframe(show.style.format({
                "Throughput (L)": "{:,.0f}", "Target variance (L)": "{:,.0f}",
                "Target var %": "{:+.1f}%", "Dip variance (L)": "{:,.1f}",
                "Stock loss %": "{:+.2f}%", "Delivery shortage (L)": "{:,.0f}"}, na_rep="—"),
                use_container_width=True, hide_index=True)

    # ============================ RANKINGS ============================
    with tabs[5]:
        st.markdown("<div class='eyebrow'>Composite performance index</div>",
                    unsafe_allow_html=True)
        st.subheader(f"Station rankings — {PLABEL[product]}")
        st.caption(f"Index blends attainment {RANK_W_ATTAIN:.0%}, throughput "
                   f"{RANK_W_VOLUME:.0%} and stock discipline {RANK_W_DISCIPLINE:.0%}.")
        if rankings.empty:
            st.info("No data to rank in this window.")
        else:
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            rk = rankings.copy()
            rk["rank"] = rk["rank"].apply(lambda r: f"{medals.get(r, '')} {int(r)}".strip())
            fig = px.bar(rankings.head(15).iloc[::-1], x="score", y="station",
                         orientation="h", title="Performance index",
                         labels={"score": "Index (0–100)", "station": ""})
            fig.update_traces(marker_color=accent)
            st.plotly_chart(style_fig(fig, max(280, 34 * min(15, len(rankings))), accent),
                            use_container_width=True)
            show = rk.rename(columns={
                "rank": "Rank", "station": "Station", "score": "Index",
                "total_volume": "Volume (L)", "attainment_pct": "Attainment %",
                "stock_loss_pct": "Stock loss %", "rank_volume": "Vol rank",
                "rank_attain": "Attain rank"})
            cols = ["Rank", "Station", "Index", "Volume (L)", "Attainment %",
                    "Stock loss %", "Vol rank", "Attain rank"]
            st.dataframe(show[cols].style.format({
                "Index": "{:,.0f}", "Volume (L)": "{:,.0f}", "Attainment %": "{:,.0f}%",
                "Stock loss %": "{:+.2f}%", "Vol rank": "{:.0f}", "Attain rank": "{:.0f}"},
                na_rep="—"), use_container_width=True, hide_index=True)

    # ============================ TRENDS ============================
    with tabs[6]:
        st.markdown("<div class='eyebrow'>Full history</div>", unsafe_allow_html=True)
        st.subheader(f"Trends — {PLABEL[product]} · {focus}")
        if focus == "All stations":
            s = (df[df["product"] == product].groupby("date", as_index=False)
                 .agg(volume=("volume", "sum")))
            s["price"] = np.nan
        else:
            s = df[(df["product"] == product) & (df["station"] == focus)][
                ["date", "volume", "price"]]
        s = s.sort_values("date")
        if s.empty:
            st.info("No data for this selection.")
        else:
            sv = s["volume"].where(s["volume"] > 0) if EXCLUDE_ZERO else s["volume"]
            med = float(np.nanmedian(sv.dropna())) if sv.notna().any() else np.nan
            target = med * 2 if not np.isnan(med) else np.nan
            s["ma"] = s["volume"].rolling(7, min_periods=1).mean()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=s["date"], y=s["volume"], name="Daily volume",
                        marker_color=PSOFT[product], secondary_y=False)
            fig.add_scatter(x=s["date"], y=s["ma"], name="7-day average",
                            line=dict(color=accent, width=2.5), secondary_y=False)
            if not np.isnan(target):
                fig.add_hline(y=target, line_dash="dash", line_color="#16191F",
                              annotation_text="target (2× median)", secondary_y=False)
            if focus != "All stations" and s["price"].notna().any():
                fig.add_scatter(x=s["date"], y=s["price"], name="Price",
                                line=dict(color="#16191F", width=1.5, shape="hv"),
                                secondary_y=True)
                fig.update_yaxes(title_text="Price (GHS/L)", secondary_y=True)
            # shade current target window
            fig.add_vrect(x0=cur_s, x1=cur_e, fillcolor="rgba(22,25,31,.05)",
                          line_width=0, annotation_text="current", annotation_position="top left")
            fig.update_yaxes(title_text="Volume (L/day)", secondary_y=False)
            fig.update_layout(title="Daily volume, target & price")
            st.plotly_chart(style_fig(fig, 380, accent), use_container_width=True)

            # network heatmap (station × week)
            st.markdown("<div class='eyebrow'>Weekly volume heatmap</div>",
                        unsafe_allow_html=True)
            hm = df[df["product"] == product].copy()
            hm["week"] = hm["date"].dt.to_period("W").dt.start_time
            piv = hm.pivot_table(index="station", columns="week", values="volume",
                                 aggfunc="sum")
            if not piv.empty:
                fig = go.Figure(go.Heatmap(
                    z=piv.values, x=[d.strftime("%d %b") for d in piv.columns],
                    y=list(piv.index), colorscale=SCALE[product],
                    colorbar=dict(title="L/wk")))
                st.plotly_chart(style_fig(fig, max(260, 30 * len(piv)), accent),
                                use_container_width=True)

    st.caption("All figures derived from the MASTER sheet · prices in GHS · "
               "targets recompute from the target window; other views use full history.")


if __name__ == "__main__":
    main()