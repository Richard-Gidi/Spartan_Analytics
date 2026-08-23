"""
extract_npa_floors.py — read NPA price-floor notices into CSV / Excel
=====================================================================

Point it at a folder of NPA price-floor PDFs. It writes:

    npa_price_floors.xlsx      Wide / Long / Review / Files sheets
    npa_price_floors_long.csv  one row per file x product x price basis
    npa_floors_config.json     PMS + AGO + LPG ex-pump floors for omc_config.json

We only care about Gasoline (Petrol), Gasoil (Diesel) and LPG. NPA lists MGO
Local and Kerosene on the same notice; those rows are skipped — though their
positions are still used to work out the table geometry, since more row labels
means a better read.

Why this is more careful than it looks
--------------------------------------
NPA has issued these notices in at least four different shapes, and you cannot
tell which one you are holding by looking at it:

  1. A proper PDF table with a text layer            (June 2025)
  2. Text, but serialised so badly that the usual
     extractors return alphabet soup                 (April & June 2024)
  3. The table flattened into an IMAGE, no text      (Nov 2025, Aug 2026)
  4. TWO price columns — Ex-Refinery AND Ex-Pump     (the 2024 notices)

Shape 4 is the dangerous one. Reading "the number in the Petrol row" off the
April 2024 notice gives 9.80, the ex-refinery floor. The ex-pump floor is
13.02 — a third higher, and the wrong figure looks entirely normal. So this
script never takes "the first number in the row": it works out which column is
which from the heading and labels every figure with its basis.

Shape 3 is the imprecise one. OCR on NPA's display font misreads decimals: in
testing it read a kerosene floor of 9.21 as "971", and one preprocessing
variant read an LPG floor of 8.93 as "6.95" six times over, perfectly
consistently. So OCR figures are cross-checked between two independent crop
strategies and trusted only when both agree; anything else is surfaced for a
human rather than written into the config.

Reading order, best first:
    char grid  -> exact, column-aware, handles shapes 1, 2 and 4
    pdf table  -> exact, column-aware
    text lines -> exact, single column only
    OCR        -> approximate, cross-checked, shape 3 only

Install
-------
    pip install pdfplumber pypdfium2 pandas openpyxl
    pip install pytesseract Pillow numpy        # only for the OCR path

OCR also needs the Tesseract program itself (a normal Windows installer, not a
pip package): https://github.com/UB-Mannheim/tesseract/wiki
Install it, then add it to PATH or pass --tesseract. Without it the script still
runs: text-based notices extract perfectly and image-only ones are listed as
needing manual entry.

Usage
-----
    python extract_npa_floors.py
    python extract_npa_floors.py --folder "C:\\Users\\GIDI\\Desktop\\APPS\\Spartan Analytics"
    python extract_npa_floors.py --merge-config omc_config.json
    python extract_npa_floors.py --no-ocr
    python extract_npa_floors.py --tesseract "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_FOLDER = r"C:\Users\GIDI\Desktop\APPS\Spartan Analytics\npa"

# code, the word NPA uses, display name, unit
PRODUCTS = [
    ("PMS", "petrol", "Gasoline (Petrol)", "GHS/L"),
    ("AGO", "diesel", "Gasoil (Diesel)",   "GHS/L"),
    ("LPG", "lpg",    "LPG",               "GHS/kg"),
]
LABEL_TO_CODE = {lbl: code for code, lbl, _, _ in PRODUCTS}
APP_CODES = ("PMS", "AGO", "LPG")

# Products on the notice we don't trade. Skipped for extraction...
IGNORED_LABELS = ("mgo", "local", "kerosene", "marine")
# ...but their row positions still help locate the figure column during OCR.
LAYOUT_LABELS = {"petrol": "PMS", "diesel": "AGO", "lpg": "LPG",
                 "mgo": "_MGO", "kerosene": "_KERO"}

# Two possible price bases. Only ex-pump is a retail compliance floor;
# ex-refinery is the BIDEC-side number, captured because it is worth having.
BASIS_PUMP, BASIS_REF = "ex_pump", "ex_refinery"

# "9 .81" happens — these PDFs sometimes split a number with a stray space.
NUM_TOKEN = re.compile(r"\d{1,3}\s?\.\s?\d{1,2}")
PRICE_RE = re.compile(r"^\d{1,3}\.\d{1,2}$")
PRICE_MIN, PRICE_MAX = 0.50, 500.0

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]

OCR_CFGS = ("--psm 7  -c tessedit_char_whitelist=0123456789.",
            "--psm 13 -c tessedit_char_whitelist=0123456789.",
            "--psm 6  -c tessedit_char_whitelist=0123456789.")

# A figure reaches the config file only if it came off a text layer, or two
# independent OCR strategies agreed on it.
TRUSTED = ("char grid", "pdf table", "pdf text", "ocr agreed")


# ───────────────────────────────── helpers ─────────────────────────────────
def _num(tok):
    try:
        v = float(str(tok).replace(" ", "").replace(",", ""))
    except (TypeError, ValueError):
        return None
    return v if PRICE_MIN <= v <= PRICE_MAX else None


def _month_num(word):
    w = str(word).strip().lower()[:3]
    for i, m in enumerate(MONTHS, start=1):
        if m.startswith(w):
            return i
    return None


def _mk(y, m, d):
    try:
        if m == 2 and d > 28:
            d = 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
        if d > 30 and m in (4, 6, 9, 11):
            d = 30
        return pd.Timestamp(date(y, m, min(d, 31)))
    except ValueError:
        return None


def parse_period(text, filename=""):
    """The window the notice covers. Header first, then the filename — NPA names
    its files consistently, which has rescued more than one badly built PDF."""
    blob = f"{text}\n{filename}".replace("\u2013", "-").replace("\u2014", "-")
    blob = re.sub(r"(?i)(\d{1,2})\s*(st|nd|rd|th)\b", r"\1", blob)
    pat = "|".join(MONTHS)
    m = re.search(rf"(\d{{1,2}})\s*[-to]{{1,3}}\s*(\d{{1,2}})[\s\-]+({pat})\w*[\s\-]+(\d{{4}})",
                  blob, re.I)
    if m:
        d1, d2, mon, yr = m.groups()
        i = _month_num(mon)
        if i:
            return _mk(int(yr), i, int(d1)), _mk(int(yr), i, int(d2))
    m = re.search(rf"({pat})\w*[\s\-]+(\d{{1,2}})\s*[-to]{{1,3}}\s*(?:(?:{pat})\w*[\s\-]+)?"
                  rf"(\d{{1,2}})[,\s\-]+(\d{{4}})", blob, re.I)
    if m:
        mon, d1, d2, yr = m.groups()
        i = _month_num(mon)
        if i:
            return _mk(int(yr), i, int(d1)), _mk(int(yr), i, int(d2))
    return None, None


def parse_issue_date(text):
    t = re.sub(r"(?i)(\d{1,2})\s*(st|nd|rd|th)\b", r"\1", text.replace("\u2013", "-"))
    pat = "|".join(MONTHS)
    hits = re.findall(rf"(\d{{1,2}})\s+({pat})\w*\s+(\d{{4}})", t, re.I)
    if not hits:
        return None
    d, mon, yr = hits[-1]                    # the sign-off sits at the foot
    i = _month_num(mon)
    return _mk(int(yr), i, int(d)) if i else None


def _assign_basis(tokens, headers):
    """tokens: [(value, x_left, x_right)] found in one product row.
    headers:  {basis: x_left of that column's heading}, possibly empty.

    One figure means one column, and a single-column notice is always ex-pump.
    Two or more means the ex-refinery/ex-pump layout, and each figure is placed
    under the heading it sits nearest — never by assuming an order.
    """
    if not tokens:
        return {}
    if len(tokens) == 1:
        return {BASIS_PUMP: tokens[0][0]}
    if headers:
        out = {}
        for basis, hx in headers.items():
            best = min(tokens, key=lambda t: abs(((t[1] + t[2]) / 2) - hx))
            out[basis] = best[0]
        if len(set(out.values())) == len(out):
            return out
    # no usable heading: ex-pump is the rightmost column on every notice seen
    ordered = sorted(tokens, key=lambda t: t[2])
    return {BASIS_REF: ordered[-2][0], BASIS_PUMP: ordered[-1][0]}


# ──────────────── layer 1: rebuild the table from character x/y ─────────────
def _char_rows(chars, tol_ratio=0.6):
    """Group characters into visual rows, then rebuild each row's text from
    geometry rather than from the PDF's own space glyphs.

    That last part matters more than it sounds. The June 2024 notice emits a
    space glyph at exactly the same x as the '1' in '13.22'. Sort by x and you
    get '1 3.22'; any number regex then happily returns 3.22, and a floor that
    should read 13.22 is out by ten cedis while looking perfectly plausible.
    Ignoring the space glyphs and inserting our own gaps from the character
    boxes removes the problem at its root.
    """
    if not chars:
        return []
    real = [c for c in chars if c.get("text") and not str(c["text"]).isspace()]
    if not real:
        return []
    groups, current, last_top = [], [], None
    for c in sorted(real, key=lambda c: (c["top"], c["x0"])):
        size = c.get("size") or 10.0
        if last_top is None or abs(c["top"] - last_top) <= tol_ratio * size:
            current.append(c)
            last_top = c["top"] if last_top is None else last_top
        else:
            groups.append(current)
            current, last_top = [c], c["top"]
    if current:
        groups.append(current)

    rows = []
    for g in groups:
        cs = sorted(g, key=lambda c: c["x0"])
        text, refs, prev = [], [], None
        for c in cs:
            if prev is not None:
                gap = c["x0"] - prev["x1"]
                if gap > 0.28 * (c.get("size") or 10.0):
                    text.append(" ")
                    refs.append(None)
            text.append(str(c["text"]))
            refs.append(c)
            prev = c
        rows.append(("".join(text), refs))
    return rows


def from_chars(page):
    """Some of these PDFs carry a full text layer but serialise it so badly that
    extract_text() returns one letter per line. The characters still know where
    they sit on the page, so we regroup them into rows by y and read across by
    x. Exact, and it sees both price columns."""
    rows = _char_rows(page.chars)
    if not rows:
        return {}

    headers = {}
    for line, refs in rows:
        flat, flat_refs = [], []
        for ch, ref in zip(line, refs):
            if not re.match(r"[\s\-]", ch):
                flat.append(ch.lower())
                flat_refs.append(ref)
        flat = "".join(flat)
        for basis, needle in ((BASIS_REF, "exrefinery"), (BASIS_PUMP, "expump")):
            if basis in headers:
                continue
            pos = flat.find(needle)
            if pos >= 0 and flat_refs[pos] is not None:
                headers[basis] = flat_refs[pos]["x0"]

    out = {}
    for line, refs in rows:
        low = line.lower()
        if any(w in low for w in IGNORED_LABELS):
            continue
        code = next((LABEL_TO_CODE[k] for k in LABEL_TO_CODE
                     if re.match(rf"^\s*{k}\b", low)), None)
        if not code or code in out:
            continue
        tokens = []
        for m in NUM_TOKEN.finditer(line):
            v = _num(m.group(0))
            if v is None:
                continue
            lo = next((refs[i] for i in range(m.start(), m.end()) if refs[i]), None)
            hi = next((refs[i] for i in range(m.end() - 1, m.start() - 1, -1) if refs[i]), None)
            if lo is None or hi is None:
                continue
            tokens.append((v, lo["x0"], hi["x1"]))
        got = _assign_basis(tokens, headers)
        if got:
            out[code] = got
    return out if len(out) >= 2 else {}


# ─────────────────────── layer 2: pdfplumber's own tables ──────────────────
def from_tables(page):
    for table in (page.extract_tables() or []):
        headers, out = {}, {}
        for row in table:
            cells = [str(c or "").strip() for c in row]
            flat = [re.sub(r"[\s\-]", "", c).lower() for c in cells]
            for i, c in enumerate(flat):
                if "exrefinery" in c and BASIS_REF not in headers:
                    headers[BASIS_REF] = i
                if "expump" in c and BASIS_PUMP not in headers:
                    headers[BASIS_PUMP] = i
            joined = " ".join(cells).lower()
            if any(w in joined for w in IGNORED_LABELS):
                continue
            code = next((LABEL_TO_CODE[k] for c in cells
                         for k in LABEL_TO_CODE if k in c.lower()), None)
            if not code or code in out:
                continue
            if headers:
                vals = {}
                for b, i in headers.items():
                    if i < len(cells):
                        m = NUM_TOKEN.search(cells[i])
                        v = _num(m.group(0)) if m else None
                        if v is not None:
                            vals[b] = v
                if vals:
                    out[code] = vals
                    continue
            tokens = [(_num(m.group(0)), j, j) for j, c in enumerate(cells)
                      for m in NUM_TOKEN.finditer(c)]
            tokens = [t for t in tokens if t[0] is not None]
            got = _assign_basis(tokens, {})
            if got:
                out[code] = got
        if len(out) >= 2:
            return out
    return {}


# ────────────────────────── layer 3: plain text lines ──────────────────────
def from_text(text):
    out = {}
    for line in text.splitlines():
        low = line.strip().lower()
        if any(w in low for w in IGNORED_LABELS):
            continue
        code = next((LABEL_TO_CODE[k] for k in LABEL_TO_CODE
                     if re.match(rf"^{k}\b", low)), None)
        if not code or code in out:
            continue
        tokens = [(_num(m.group(0)), m.start(), m.end()) for m in NUM_TOKEN.finditer(line)]
        tokens = [t for t in tokens if t[0] is not None]
        got = _assign_basis(tokens, {})
        if got:
            out[code] = got
    return out if len(out) >= 2 else {}


# ────────────────────── layer 4: OCR, cross-checked ────────────────────────
def _render(pdf_path, page_index, dpi=300):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return doc[page_index].render(scale=dpi / 72).to_pil()
    finally:
        doc.close()


def _readings(strip):
    import pytesseract
    from PIL import Image, ImageOps
    import numpy as np
    votes = {}
    for scale in (3, 4, 6):
        s = strip.resize((strip.width * scale, strip.height * scale), Image.LANCZOS)
        s = ImageOps.autocontrast(s.convert("L"))
        binar = Image.fromarray(((np.array(s) > 150) * 255).astype("uint8"))
        for v in (s, binar):
            for cfg in OCR_CFGS:
                try:
                    raw = pytesseract.image_to_string(v, config=cfg)
                except Exception:
                    continue
                t = re.sub(r"[^\d.]", "", raw.strip())
                if PRICE_RE.match(t) and PRICE_MIN <= float(t) <= PRICE_MAX:
                    votes[t] = votes.get(t, 0) + 1
    return votes


def _winner(votes):
    if not votes:
        return None, 0.0
    total = sum(votes.values())
    best, n = max(votes.items(), key=lambda kv: kv[1])
    return best, round(n / total, 2)


def from_ocr(pdf_path, page_index, log):
    """Two independent ways of cutting out the figure, cross-checked.

    A  the strip immediately right of the product label, out to the page edge
    C  the figure column, bounded on the right by the table's own rule line

    Both anchor on things OCR reads reliably — the product word, a solid rule —
    and they fail differently, so agreement between them means something. Where
    an Ex-Pump heading is found, both crops start there, so an image-only
    two-column notice cannot hand back the ex-refinery figure by mistake. If the
    notice clearly has two columns but the heading is not legible, we refuse to
    guess rather than risk it.
    """
    try:
        import pytesseract
        import numpy as np
        from PIL import ImageOps
    except ImportError:
        log.append("OCR skipped: pytesseract / Pillow / numpy not installed")
        return {}
    try:
        page_img = _render(pdf_path, page_index)
    except Exception as e:
        log.append(f"OCR skipped: could not render the page ({e})")
        return {}
    try:
        data = pytesseract.image_to_data(page_img, output_type=pytesseract.Output.DICT)
    except Exception as e:
        log.append(f"OCR unavailable — is Tesseract installed and on PATH? ({e})")
        return {}

    W, H = page_img.size
    layout, pump_x, ref_seen = {}, None, False
    for i, raw in enumerate(data["text"]):
        w = re.sub(r"[^a-z]", "", str(raw).lower())
        code = LAYOUT_LABELS.get(w)
        if code and code not in layout:
            layout[code] = (data["left"][i], data["top"][i],
                            data["width"][i], data["height"][i])
        if w.startswith("expump") and pump_x is None:
            pump_x = data["left"][i]
        if w.startswith("exrefinery"):
            ref_seen = True
    if not layout:
        log.append("OCR found no product labels on this page")
        return {}
    anchors = {k: v for k, v in layout.items() if not k.startswith("_")}
    if not anchors:
        log.append("OCR found the table but none of our three grades in it")
        return {}
    if ref_seen and pump_x is None:
        log.append("two price columns, but the Ex-Pump heading was not legible")
        return {c: {"value": None, "confidence": 0.0,
                    "status": "ocr ambiguous columns",
                    "detail": "could not tell the ex-refinery column from the ex-pump one"}
                for c in anchors}

    grey = np.array(page_img.convert("L"))
    dark = grey < 160
    tops = [v[1] for v in layout.values()]
    hts = [v[3] for v in layout.values()]
    y_lo, y_hi = max(min(tops) - 20, 0), min(max(tops) + max(hts) + 20, H)
    x_lo = max(v[0] + v[2] for v in layout.values()) + 15
    if pump_x is not None:
        x_lo = max(x_lo, pump_x - 20)
    col_density = dark[y_lo:y_hi, :].mean(axis=0)
    rules = [x for x in range(x_lo, W) if col_density[x] > 0.70]
    x_hi = max(rules) - 6 if rules else W
    if x_hi - x_lo < 40:
        x_hi = W

    out = {}
    for code, (x, y, w, h) in anchors.items():
        pad = max(int(h * 0.45), 8)
        top, bot = max(y - pad, 0), min(y + h + pad, H)
        a_left = min(x + w + int(h * 0.4), W - 1)
        if pump_x is not None:
            a_left = max(a_left, x_lo)
        strip_a = page_img.crop((a_left, top, W, bot)).convert("L")
        strip_c = ImageOps.expand(page_img.crop((x_lo, top, x_hi, bot)).convert("L"),
                                  border=30, fill=255)
        wa, ca = _winner(_readings(strip_a))
        wc, cc = _winner(_readings(strip_c))
        if wa and wc and wa == wc:
            out[code] = {"value": float(wa), "confidence": round(min(ca, cc), 2),
                         "status": "ocr agreed", "detail": f"both methods read {wa}"}
        elif wa and wc:
            out[code] = {"value": None, "confidence": 0.0, "status": "ocr conflict",
                         "detail": f"methods disagreed ({wa} vs {wc})"}
        elif wa or wc:
            v, c = (wa, ca) if wa else (wc, cc)
            out[code] = {"value": float(v), "confidence": round(c * 0.6, 2),
                         "status": "ocr single method",
                         "detail": f"only one method read it ({v})"}
        else:
            out[code] = {"value": None, "confidence": 0.0, "status": "ocr unreadable",
                         "detail": "no legible figure"}
    return out


# ─────────────────────────────── one document ──────────────────────────────
def read_pdf(pdf_path, allow_ocr=True):
    import pdfplumber

    path = Path(pdf_path)
    log, text_all = [], ""
    found, method = {}, None

    with pdfplumber.open(str(path)) as pdf:
        for pidx, page in enumerate(pdf.pages):
            text_all += "\n" + (page.extract_text() or "")

            for fn, name in ((from_chars, "char grid"), (from_tables, "pdf table")):
                got = fn(page)
                if got:
                    found, method = got, name
                    break
            if method:
                break

            got = from_text(text_all)
            if got:
                found, method = got, "pdf text"
                break

            if allow_ocr:
                got = from_ocr(path, pidx, log)
                if got:
                    found, method = got, "ocr"
                    break

    if not method:
        method = "none"
        if not allow_ocr:
            log.append("no text layer and OCR was disabled (--no-ocr)")
        elif not log:
            log.append("no table, no text layer and nothing legible on the page")

    w_start, w_end = parse_period(text_all, path.name)
    issued = parse_issue_date(text_all)

    rows = []
    for code, _lbl, name, unit in PRODUCTS:
        if code not in found:
            continue
        rec = found[code]
        if method == "ocr":
            entries = [(BASIS_PUMP, rec["value"], rec["confidence"],
                        rec["status"], rec["detail"])]
        else:
            entries = [(b, v, 1.0, method, "read from the document's own text")
                       for b, v in rec.items()]
        for basis, value, conf, status, detail in entries:
            if status in TRUSTED:
                note = "" if conf >= 0.95 else f"agreed, but only {conf:.0%} of readings"
            elif status == "ocr single method":
                note = (f"CONFIRM — {detail}; shown so you can tick it, "
                        "not loaded into the app")
            elif status in ("ocr conflict", "ocr ambiguous columns"):
                note = f"NEEDS REVIEW — {detail}; open the PDF and type it in"
            else:
                note = "NEEDS REVIEW — could not read this figure; open the PDF and type it in"
            rows.append({
                "source_file": path.name,
                "window_start": w_start,
                "window_end": w_end,
                "window_label": (f"{w_start:%d %b %Y} - {w_end:%d %b %Y}"
                                 if w_start is not None and w_end is not None else ""),
                "issue_date": issued,
                "product_code": code,
                "product": name,
                "basis": basis,
                "unit": unit,
                "floor": value,
                "status": status,
                "confidence": conf,
                "trusted": status in TRUSTED,
                "note": note,
            })

    # An ex-pump floor below the ex-refinery floor is arithmetically impossible —
    # the pump price carries margins and levies on top. If we see it, the columns
    # were mixed up and neither figure can be trusted.
    for code in {r["product_code"] for r in rows}:
        pair = {r["basis"]: r for r in rows if r["product_code"] == code}
        p, q = pair.get(BASIS_PUMP), pair.get(BASIS_REF)
        if p and q and p["floor"] is not None and q["floor"] is not None \
                and p["floor"] < q["floor"]:
            for r in (p, q):
                r["trusted"] = False
                r["note"] = ("NEEDS REVIEW — ex-pump reads below ex-refinery, which cannot "
                             "be right; the two columns were probably swapped")

    pump = [r for r in rows if r["basis"] == BASIS_PUMP]
    status_row = {
        "source_file": path.name,
        "window_label": (f"{w_start:%d %b %Y} - {w_end:%d %b %Y}"
                         if w_start is not None and w_end is not None else "NOT FOUND"),
        "method": method,
        "columns": ("ex-refinery + ex-pump" if any(r["basis"] == BASIS_REF for r in rows)
                    else "ex-pump only"),
        "trusted": sum(1 for r in pump if r["trusted"]),
        "confirm": sum(1 for r in pump if r["status"] == "ocr single method"),
        "review": sum(1 for r in pump if r["floor"] is None),
        "log": "; ".join(log),
    }
    if w_start is None:
        status_row["log"] = "; ".join(
            log + ["window dates not found — keep NPA's own filename, or set them by hand"]
        ).strip("; ")
    return rows, status_row


# ───────────────────────────────── outputs ─────────────────────────────────
def build_wide(long_df):
    if long_df.empty:
        return pd.DataFrame()
    codes = [c for c, _, _, _ in PRODUCTS]
    w = long_df.pivot_table(index=["window_start", "window_end", "window_label",
                                   "issue_date", "source_file"],
                            columns=["basis", "product_code"], values="floor",
                            aggfunc="first")
    w.columns = [c if b == BASIS_PUMP else f"{c}_exref" for b, c in w.columns]
    w = w.reset_index()
    for c in codes:
        if c not in w.columns:
            w[c] = pd.NA
    exref = [f"{c}_exref" for c in codes if f"{c}_exref" in w.columns]
    w = w[["window_start", "window_end", "window_label", "issue_date"] + codes
          + exref + ["source_file"]].sort_values("window_start").reset_index(drop=True)
    for c in codes:
        w[f"{c}_change"] = pd.to_numeric(w[c], errors="coerce").diff().round(4)
    return w


def build_config(long_df):
    """The slice the analytics app consumes: ex-pump floors only, keyed by the
    day the window opens. Anything not trusted is deliberately left out."""
    cfg = {"floors": {c: {} for c in APP_CODES}}
    if long_df.empty:
        return cfg
    ok = long_df[(long_df["basis"] == BASIS_PUMP) & long_df["floor"].notna()
                 & long_df["window_start"].notna()
                 & long_df["trusted"].fillna(False).astype(bool)]
    ok = ok[ok["product_code"].isin(APP_CODES)].sort_values(["window_start", "confidence"])
    for _, r in ok.iterrows():
        key = pd.Timestamp(r["window_start"]).strftime("%Y-%m-%d")
        cfg["floors"][r["product_code"]][key] = float(r["floor"])
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description="Extract NPA ex-pump price floors from PDF notices.")
    ap.add_argument("--folder", default=DEFAULT_FOLDER, help="folder holding the PDFs")
    ap.add_argument("--out-dir", default=None, help="where to write (default: --folder)")
    ap.add_argument("--no-ocr", action="store_true",
                    help="text-based notices only; image-only ones get listed, not read")
    ap.add_argument("--tesseract", default=None, help="full path to tesseract.exe")
    ap.add_argument("--pattern", default="*.pdf")
    ap.add_argument("--merge-config", default=None,
                    help="path to omc_config.json — merges trusted ex-pump floors "
                         "straight in (a .bak copy is made first)")
    args = ap.parse_args()

    if args.tesseract:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = args.tesseract
        except ImportError:
            print("! --tesseract given but pytesseract isn't installed")

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"! Folder not found: {folder}")
        print('  Pass the right one with --folder "C:\\path\\to\\pdfs"')
        return 1
    out_dir = Path(args.out_dir) if args.out_dir else folder
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(folder.glob(args.pattern))
    if not pdfs:
        print(f"! No PDFs matching {args.pattern} in {folder}")
        return 1

    print(f"Reading {len(pdfs)} file(s) from {folder}\n")
    rows, statuses, seen_hash = [], [], {}
    for p in pdfs:
        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
        except Exception:
            digest = None
        if digest and digest in seen_hash:
            print(f"  [dup]     {p.name}")
            print(f"            byte-identical to {seen_hash[digest]} — skipped")
            continue
        if digest:
            seen_hash[digest] = p.name
        try:
            r, s = read_pdf(p, allow_ocr=not args.no_ocr)
        except Exception as e:
            print(f"  [FAIL]    {p.name}: {e}")
            statuses.append({"source_file": p.name, "window_label": "", "method": "error",
                             "columns": "", "trusted": 0, "confirm": 0, "review": 0,
                             "log": str(e)})
            continue
        rows.extend(r)
        statuses.append(s)
        mark = ("[ok]" if s["trusted"] and not s["confirm"] and not s["review"]
                else "[CHECK]" if s["trusted"] or s["confirm"] else "[NOTHING]")
        print(f"  {mark:9s} {p.name}")
        print(f"            {s['window_label']} | {s['method']} | {s['columns']} | "
              f"{s['trusted']} trusted, {s['confirm']} to confirm, {s['review']} to type in"
              + (f" | {s['log']}" if s["log"] else ""))

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print("\n! Nothing extracted.")
        return 1
    long_df = long_df.sort_values(["window_start", "product_code", "basis"]) \
                     .reset_index(drop=True)

    csv_path = out_dir / "npa_price_floors_long.csv"
    if csv_path.exists():
        try:
            prev = pd.read_csv(csv_path, parse_dates=["window_start", "window_end",
                                                      "issue_date"])
            long_df = (pd.concat([prev, long_df], ignore_index=True)
                         .sort_values(["trusted", "confidence"])
                         .drop_duplicates(subset=["window_start", "product_code", "basis"],
                                          keep="last")
                         .sort_values(["window_start", "product_code", "basis"])
                         .reset_index(drop=True))
        except Exception as e:
            print(f"\n  (couldn't merge the previous CSV, writing fresh: {e})")

    wide = build_wide(long_df)
    review = long_df[(long_df["note"].fillna("") != "")
                     | (~long_df["trusted"].fillna(False).astype(bool))]
    files_df = pd.DataFrame(statuses)

    long_df.to_csv(csv_path, index=False)
    xlsx_path = out_dir / "npa_price_floors.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        wide.to_excel(xw, sheet_name="Wide", index=False)
        long_df.to_excel(xw, sheet_name="Long", index=False)
        (review if not review.empty
         else pd.DataFrame({"note": ["Nothing flagged - every figure read cleanly."]})
         ).to_excel(xw, sheet_name="Review", index=False)
        files_df.to_excel(xw, sheet_name="Files", index=False)

    cfg = build_config(long_df)
    json_path = out_dir / "npa_floors_config.json"
    json_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nWrote:\n  {xlsx_path}\n  {csv_path}\n  {json_path}")

    if args.merge_config:
        target = Path(args.merge_config)
        try:
            live = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
            if target.exists():
                target.with_suffix(target.suffix + ".bak").write_text(
                    json.dumps(live, indent=2, sort_keys=True), encoding="utf-8")
            live.setdefault("floors", {})
            added = 0
            for code in APP_CODES:
                live["floors"].setdefault(code, {})
                for k, v in cfg["floors"][code].items():
                    if live["floors"][code].get(k) != v:
                        added += 1
                    live["floors"][code][k] = v
            target.write_text(json.dumps(live, indent=2, sort_keys=True), encoding="utf-8")
            print(f"  merged {added} floor(s) into {target} (backup: {target.name}.bak)")
        except Exception as e:
            print(f"  ! could not merge into {target}: {e}")

    pump = long_df[long_df["basis"] == BASIS_PUMP]
    n_trust = int(pump["trusted"].fillna(False).astype(bool).sum())
    n_conf = int((pump["status"] == "ocr single method").sum())
    n_blank = int(pump["floor"].isna().sum())
    n_ref = int((long_df["basis"] == BASIS_REF).sum())
    print(f"\n{len(pump)} ex-pump figure(s) across {pump['window_start'].nunique()} window(s):")
    print(f"  {n_trust} trusted   (document's own text, or two OCR methods agreeing) -> in the JSON")
    print(f"  {n_conf} to confirm (one OCR method only)                                -> not in the JSON")
    print(f"  {n_blank} to type in (unreadable)                                         -> not in the JSON")
    if n_ref:
        print(f"Also captured {n_ref} ex-refinery figure(s) — the BIDEC-side number, kept for "
              f"reference and never fed to the app.")
    if n_conf or n_blank:
        print("\nSettle those against the PDFs in the Review sheet. It is a one-minute job, and "
              "it is the difference between a compliance report you can defend and one you "
              "cannot.")
    print("\nTo load into the app: --merge-config omc_config.json, or paste the \"floors\" "
          "block from npa_floors_config.json, or type them into Settings -> Price floors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())