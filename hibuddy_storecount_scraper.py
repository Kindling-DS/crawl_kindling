#!/usr/bin/env python3
"""hibuddy_storecount_scraper.py (v5)

v5 focus: PRE-FILTER BY CATEGORY + BRAND (LEFT SIDEBAR)
------------------------------------------------------
Keeps all v4 match-verification logic, and adds optional left-side filters on /products/all:

1) Map AGLC "Format" -> Hibuddy main Category checkbox:
   - Dried Flower, Milled Flower -> Flower
   - Pre-Roll -> Pre-Rolls
   - Vape -> Vapes
   - Concentrate or Extract, Oil or Spray -> Extracts
   - Edible, Beverage, Beverage - Non-liquid, Capsule or Soft Gel -> Edibles
   - Topical -> Topicals

2) Apply Brand filter via left sidebar Brand section (when detectable).

Why it helps:
- Smaller, cleaner candidate grid -> you can lower threshold / verify fewer candidates
- Lower risk of "same brand, wrong item" (and fewer false positives overall)

Speed / weekend-run mode:
- Sorts input by (Format, Brand Name) by default to reduce filter switching.
- Reuses filters for consecutive rows with same (Category, Brand).

Output:
- Adds columns: hibuddy_category_filter, hibuddy_search_query, filters_applied

IMPORTANT
---------
You are responsible for ensuring this complies with hibuddy.ca Terms / robots / policies.
Keep the request rate LOW.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


HIBUDDY_PRODUCTS_ALL_URL = "https://hibuddy.ca/products/all?orderby=added"

# When True, skip attempting to change "Rows per page" to VIEW ALL on retailer tables.
DISABLE_VIEW_ALL = False


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _tokenize(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(s)))


def _sleep(min_s: float, max_s: float) -> None:
    time.sleep(random.uniform(min_s, max_s))


def _maybe_click_age_gate(page) -> None:
    try:
        btn = page.get_by_role("button", name=re.compile(r"^\s*yes\s*$", re.I))
        if btn.count() > 0 and btn.first.is_visible():
            btn.first.click()
    except Exception:
        pass


# -----------------------------
# Speed + reliability helpers
# -----------------------------

_BAD_STATUS_CODES = {502, 503, 504, 520, 521, 522, 523, 524, 525, 526}

def _enable_resource_blocking(context) -> None:
    """Block heavy resources (images/media/fonts) to speed up scraping.
    Keep CSS+JS so the site still works.
    """
    def _route(route, request):
        try:
            rtype = request.resource_type
            if rtype in ("image", "media", "font"):
                return route.abort()
            url = request.url.lower()
            if url.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg', '.mp4', '.webm', '.mov', '.avi', '.woff', '.woff2', '.ttf', '.otf')):
                return route.abort()
        except Exception:
            pass
        return route.continue_()
    try:
        context.route("**/*", _route)
    except Exception:
        # route() can fail in some rare cases; ignore and continue normally
        pass


def _page_looks_like_gateway(page) -> bool:
    """Detect common Cloudflare / 502 gateway pages."""
    try:
        title = (page.title() or "").lower()
        if "bad gateway" in title or "error" in title:
            return True
    except Exception:
        pass
    try:
        # Body text is usually tiny on gateway pages; cap timeouts
        body = page.locator("body").inner_text(timeout=1200).lower()
        if "bad gateway" in body:
            return True
        if "cloudflare" in body and ("error" in body or "gateway" in body):
            return True
        if "checking your browser" in body and "cloudflare" in body:
            return True
        if "gateway time-out" in body or "gateway timeout" in body:
            return True
    except Exception:
        pass
    return False


def _recover_from_gateway(page, base_url: str) -> None:
    """Best-effort recovery if we hit a gateway / Cloudflare error."""
    # Try reload
    try:
        page.reload(wait_until="domcontentloaded", timeout=45000)
        _maybe_click_age_gate(page)
        if not _page_looks_like_gateway(page):
            return
    except Exception:
        pass

    # Try going back once
    try:
        page.go_back(wait_until="domcontentloaded", timeout=45000)
        _maybe_click_age_gate(page)
        if not _page_looks_like_gateway(page):
            return
    except Exception:
        pass

    # Fallback: go to the products grid
    try:
        page.goto(base_url, wait_until="domcontentloaded", timeout=45000)
        _maybe_click_age_gate(page)
    except Exception:
        pass


def _safe_goto(page, url: str, base_url: str = HIBUDDY_PRODUCTS_ALL_URL, attempts: int = 3, timeout_ms: int = 45000):
    """goto() with retries + gateway recovery."""
    last_err = None
    for i in range(attempts):
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            _maybe_click_age_gate(page)
            if resp is not None and getattr(resp, "status", None) in _BAD_STATUS_CODES:
                raise RuntimeError(f"HTTP {resp.status}")
            if _page_looks_like_gateway(page):
                raise RuntimeError("gateway_page")
            return resp
        except Exception as e:
            last_err = e
            _recover_from_gateway(page, base_url=base_url)
            # small backoff
            try:
                page.wait_for_timeout(700 + i * 900)
            except Exception:
                _sleep(0.7 + i * 0.7, 1.4 + i * 0.9)
            continue
    if last_err:
        raise last_err
    return None


def _safe_search_to_grid(page, query: str, attempts: int = 3) -> bool:
    """Run a header search and wait for product cards to appear (with gateway recovery)."""
    last_err = None
    for i in range(attempts):
        try:
            search_input = _find_header_search_input(page)
            search_input.click()
            search_input.fill("")
            search_input.fill(query)
            search_input.press("Enter")
            page.wait_for_selector('a[href^="/product/"]', timeout=16000)
            if _page_looks_like_gateway(page):
                raise RuntimeError("gateway_page")
            return True
        except Exception as e:
            last_err = e
            _recover_from_gateway(page, base_url=HIBUDDY_PRODUCTS_ALL_URL)
            try:
                page.wait_for_timeout(900 + i * 900)
            except Exception:
                _sleep(0.9 + i * 0.8, 1.6 + i * 1.0)
            continue
    return False



def _ensure_location_is_set_once(page, interactive: bool) -> None:
    if not interactive:
        return
    print("\n=== One-time setup ===")
    print("A browser window opened.")
    print("1) Click YES on age gate if needed")
    print("2) Set location: Calgary, Alberta, Canada")
    print("3) Set radius: 50 km")
    print("4) Confirm products grid is visible")
    input("Press Enter here to start scraping... ")


def _find_header_search_input(page):
    primary = page.locator('input[placeholder*="Search Product"]')
    if primary.count() > 0 and primary.first.is_visible() and primary.first.is_editable():
        return primary.first

    inputs = page.locator("input")
    best = None
    best_y = 1e9
    for i in range(inputs.count()):
        el = inputs.nth(i)
        try:
            if not el.is_visible() or not el.is_editable():
                continue
            ph = (el.get_attribute("placeholder") or "").lower()
            aria = (el.get_attribute("aria-label") or "").lower()
            if "search categories" in ph:
                continue
            if ("search" not in ph) and ("search" not in aria):
                continue
            bbox = el.bounding_box()
            if not bbox:
                continue
            if bbox["y"] < best_y:
                best_y = bbox["y"]
                best = el
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Could not find the header search input.")
    return best


# -------------------------
# Left sidebar filter helpers
# -------------------------

FORMAT_TO_HIBUDDY_CATEGORY = {
    "Dried Flower": "Flower",
    "Milled Flower": "Flower",
    "Pre-Roll": "Pre-Rolls",
    "Vape": "Vapes",
    "Concentrate or Extract": "Extracts",
    "Oil or Spray": "Extracts",
    "Edible": "Edibles",
    "Beverage": "Edibles",
    "Beverage - Non-liquid": "Edibles",
    "Capsule or Soft Gel": "Edibles",
    "Topical": "Topicals",
}


def _map_format_to_category(fmt: str) -> str:
    fmt = (fmt or "").strip()
    return FORMAT_TO_HIBUDDY_CATEGORY.get(fmt, "")


def _sidebar_scope(page):
    """Return a locator scoped to the left filter sidebar (best effort)."""
    # Prefer anchoring on the Category search input (very stable in the UI)
    try:
        sidebar = page.locator(
            "xpath=//input[contains(translate(@placeholder,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'search categories')]/ancestor::*[contains(.,'Brand') and contains(.,'Category') and contains(.,'Subcategory')][1]"
        )
        if sidebar.count() > 0:
            return sidebar.first
    except Exception:
        pass

    # Fallback: container that includes multiple known filter section titles
    try:
        sidebar = page.locator("div").filter(has_text=re.compile(r"\bBrand\b", re.I)).filter(
            has_text=re.compile(r"\bSubcategory\b", re.I)
        ).filter(has_text=re.compile(r"\bSizes\b", re.I))
        if sidebar.count() > 0:
            return sidebar.first
    except Exception:
        pass

    return page


def _click_checkbox_by_label_text(page, label_text: str) -> bool:
    """Click/check a checkbox option by label text within the LEFT SIDEBAR only."""
    if not label_text:
        return False

    scope = _sidebar_scope(page)

    rx_exact = re.compile(rf"^\s*{re.escape(label_text)}\s*$", re.I)
    rx_prefix = re.compile(rf"^\s*{re.escape(label_text)}\b", re.I)

    for rx in (rx_exact, rx_prefix):
        # 1) Prefer rows that actually contain a checkbox input
        try:
            rows = scope.locator("css=*:has(input[type='checkbox'])").filter(has_text=rx)
            for i in range(min(rows.count(), 30)):
                row = rows.nth(i)
                if not row.is_visible():
                    continue
                row.scroll_into_view_if_needed()
                cb = row.locator("input[type='checkbox']").first
                try:
                    if cb.count() > 0:
                        cb.check(timeout=2500)
                        try:
                            if cb.is_checked():
                                return True
                        except Exception:
                            return True
                    else:
                        row.click(timeout=2500)
                        return True
                except Exception:
                    try:
                        row.click(timeout=2500)
                        return True
                    except Exception:
                        pass
        except Exception:
            pass

        # 2) Fallback: label elements inside sidebar
        try:
            lab = scope.locator("label").filter(has_text=rx)
            for i in range(min(lab.count(), 10)):
                el = lab.nth(i)
                if el.is_visible():
                    el.scroll_into_view_if_needed()
                    el.click(timeout=2500)
                    return True
        except Exception:
            pass

        # 3) Last resort: click the text inside the sidebar (avoid header nav)
        try:
            t = scope.get_by_text(rx).first
            if t.count() > 0 and t.first.is_visible():
                t.first.scroll_into_view_if_needed()
                t.first.click(timeout=2500)
                return True
        except Exception:
            pass

    return False


def _expand_filter_section(page, section_title: str) -> None:
    """Expand a left-sidebar accordion section.

    Hibuddy acts like an accordion: clicking an already-open section header collapses it.
    Category is often expanded by default after a search, so we must NOT click it if open.
    """
    try:
        scope = _sidebar_scope(page)
    except Exception:
        scope = page

    title = (section_title or "").strip().lower()
    if not title:
        return

    # Special-case Category: if the category search box is visible, it's already open.
    if title == "category":
        try:
            cat_search = scope.locator('input[placeholder*="Search categories" i]')
            if cat_search.count() > 0 and cat_search.first.is_visible():
                return
        except Exception:
            pass

    try:
        hdr = scope.get_by_text(re.compile(rf"^\s*{re.escape(section_title)}\s*$", re.I)).first
        if hdr.count() > 0 and hdr.is_visible():
            hdr.scroll_into_view_if_needed()
            hdr.click()
            _sleep(0.2, 0.5)
    except Exception:
        pass


def _apply_category_filter(page, category_label: str) -> bool:
    if not category_label:
        return False

    _expand_filter_section(page, "Category")
    scope = _sidebar_scope(page)

    # Best-effort: enforce a single Category selection (uncheck other main categories)
    main_cats = ["Flower", "Vapes", "Pre-Rolls", "Extracts", "Edibles", "Topicals"]
    try:
        for lab in main_cats:
            rx = re.compile(rf"^\s*{re.escape(lab)}\s*$", re.I)
            rows = scope.locator("css=*:has(input[type='checkbox'])").filter(has_text=rx)
            if rows.count() == 0:
                continue
            row = rows.first
            cb = row.locator("input[type='checkbox']").first
            if cb.count() == 0:
                continue
            try:
                checked = cb.is_checked()
            except Exception:
                checked = None
            if lab.lower() == category_label.lower():
                try:
                    cb.check(timeout=2500)
                except Exception:
                    try:
                        row.click(timeout=2500)
                    except Exception:
                        pass
            else:
                # uncheck others (if they were checked)
                if checked:
                    try:
                        cb.uncheck(timeout=2500)
                    except Exception:
                        try:
                            row.click(timeout=2500)
                        except Exception:
                            pass
    except Exception:
        pass

    ok = _click_checkbox_by_label_text(page, category_label)
    _sleep(0.3, 0.7)
    return ok


def _apply_brand_filter(page, brand: str) -> bool:
    brand = (brand or "").strip()
    if not brand:
        return False

    _expand_filter_section(page, "Brand")

    # Try to find a brand-search input (distinct from category search).
    try:
        inp = page.locator('input[placeholder*="brand" i]')
        if inp.count() > 0 and inp.first.is_visible() and inp.first.is_editable():
            inp.first.click()
            inp.first.fill("")
            inp.first.fill(brand)
            _sleep(0.4, 0.8)
    except Exception:
        pass

    try:
        lab = page.locator("label").filter(has_text=re.compile(rf"^\s*{re.escape(brand)}\s*$", re.I))
        if lab.count() > 0:
            for i in range(min(lab.count(), 10)):
                el = lab.nth(i)
                if el.is_visible():
                    el.click()
                    _sleep(0.4, 0.8)
                    return True
    except Exception:
        pass

    try:
        t = page.get_by_text(re.compile(rf"^\s*{re.escape(brand)}\s*$", re.I)).first
        if t.count() > 0 and t.is_visible():
            t.click()
            _sleep(0.4, 0.8)
            return True
    except Exception:
        pass

    return False


def _apply_left_filters(page, category_label: str, brand: str, enable_category: bool, enable_brand: bool) -> Tuple[bool, bool]:
    """Apply left sidebar filters (v13: category only)."""
    cat_ok = False
    if enable_category and category_label:
        cat_ok = _apply_category_filter(page, category_label)
    return cat_ok, False


def _strip_size_from_name(pname: str) -> str:
    s = pname or ""
    s = re.sub(r"\b\d+\s*[x×]\s*\d+(?:\.\d+)?\s*(g|ml)\b", " ", s, flags=re.I)
    s = re.sub(r"\b\d+(?:\.\d+)?\s*(g|ml)\b", " ", s, flags=re.I)
    s = re.sub(r"\b(oz)\b", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _parse_mg(text: str, kind: str) -> List[int]:
    out = []
    for m in re.finditer(rf"(\d{{1,4}})\s*mg\s*{kind}\b", text, flags=re.I):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass
    return out


def _parse_total_mg_in_parens(text: str, kind: str) -> List[int]:
    out = []
    for m in re.finditer(r"\(([^)]{0,60})\)", text):
        chunk = m.group(1)
        mm = re.search(rf"([\d,]{{1,7}})\s*mg\s*{kind}\b", chunk, flags=re.I)
        if mm:
            try:
                out.append(int(mm.group(1).replace(",", "")))
            except Exception:
                pass
    return out


def _parse_unit_counts(text: str) -> List[int]:
    out = []
    for m in re.finditer(r"\b(\d{1,3})\s*[x×]\s*(\d+(?:\.\d+)?)\s*(g|ml)\b", text, flags=re.I):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"\b[x×]\s*(\d{1,4})\s*(softgels|softgel|capsules|caps|gels|joints|pre-rolled|prerolls)\b", text, flags=re.I):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"\b(\d{1,4})\s*(softgels|softgel|capsules|caps|gels|joints)\b", text, flags=re.I):
        try:
            out.append(int(m.group(1)))
        except Exception:
            pass

    uniq, seen = [], set()
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq




def _fmt_num_for_size(x) -> str:
    """Format numeric values from Excel/pandas for size matching.

    - 355.0 -> "355"
    - 3.50 -> "3.5"
    """
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        s = str(f).rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(x).strip()

def _build_size_regexes(net_content, uom, pname: str) -> List[re.Pattern]:
    patterns: List[re.Pattern] = []

    def add(pat: str):
        patterns.append(re.compile(pat, re.I))

    if pd.notna(net_content) and pd.notna(uom):
        val = _fmt_num_for_size(net_content)
        unit = str(uom).strip()
        if val and unit:
            add(rf"\b{re.escape(val)}\s*{re.escape(unit)}\b")

    s = pname or ""

    for m in re.finditer(r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(g|ml)\b", s, flags=re.I):
        a, b, unit = m.group(1), m.group(2), m.group(3)
        add(rf"\b{re.escape(a)}\s*[x×]\s*{re.escape(b)}\s*{re.escape(unit)}\b")

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(g|ml)\b", s, flags=re.I):
        val, unit = m.group(1), m.group(2)
        add(rf"\b{re.escape(val)}\s*{re.escape(unit)}\b")

    for m in re.finditer(r"\b(\d{1,4})\s*(softgels|softgel|capsules|caps|gels)\b", s, flags=re.I):
        n = m.group(1)
        add(rf"\b{re.escape(n)}\s*(caps|capsules|softgels|softgel|gels)\b")

    uniq, seen = [], set()
    for p in patterns:
        key = p.pattern.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _expected_spec(brand: str, pname: str, net, uom, size_primary: str = "", size_candidates=None) -> ExpectedSpec:
    s = pname or ""
    if size_candidates is None:
        size_candidates = []
    return ExpectedSpec(
        brand=brand or "",
        pname=pname or "",
        net=net,
        uom=uom,
        size_regexes=_build_size_regexes(net, uom, pname),
        thc_mg=_parse_mg(s, "thc"),
        cbd_mg=_parse_mg(s, "cbd"),
        total_thc_mg=_parse_total_mg_in_parens(s, "thc"),
        total_cbd_mg=_parse_total_mg_in_parens(s, "cbd"),
        unit_counts=_parse_unit_counts(s),
        size_primary=size_primary or "",
        size_candidates=list(size_candidates) if size_candidates else [],
    )


# -------------------------
# Candidate scoring (v4 kept)
# -------------------------


@dataclass
class ExpectedSpec:
    brand: str
    pname: str
    net: object
    uom: object
    size_regexes: List[re.Pattern]
    thc_mg: List[int]
    cbd_mg: List[int]
    total_thc_mg: List[int]
    total_cbd_mg: List[int]
    unit_counts: List[int]

    size_primary: str = ""
    size_candidates: List[str] = field(default_factory=list)

@dataclass
class Candidate:
    href: str
    card_text: str
    rough_score: float


def _score_card_text(card_text: str, exp: ExpectedSpec) -> float:
    t = _norm(card_text)
    brand_n = _norm(exp.brand)
    base_name = _strip_size_from_name(exp.pname)
    base_n = _norm(base_name)

    score = 0.0

    if brand_n and brand_n in t:
        score += 20.0

    ct = _tokenize(card_text)
    bt = _tokenize(exp.brand)
    nt = _tokenize(base_name)
    score += 2.5 * len((bt | nt) & ct)

    try:
        import difflib
        score += 14.0 * difflib.SequenceMatcher(None, base_n, t).ratio()
    except Exception:
        pass

    for v in exp.thc_mg[:2]:
        if str(v) in card_text:
            score += 2.0
    for v in exp.cbd_mg[:2]:
        if str(v) in card_text:
            score += 1.5

    return score


def _collect_candidates_from_grid(page, exp: ExpectedSpec, limit: int) -> List[Candidate]:
    links = page.locator('a[href^="/product/"]')
    n = links.count()
    if n == 0:
        return []

    out: List[Candidate] = []
    for i in range(min(n, limit)):
        a = links.nth(i)
        try:
            txt = a.inner_text(timeout=2000)
            href = a.get_attribute("href")
        except Exception:
            continue
        if not href:
            continue
        sc = _score_card_text(txt or "", exp)
        out.append(Candidate(href=href, card_text=txt or "", rough_score=sc))

    out.sort(key=lambda c: c.rough_score, reverse=True)
    return out


def _extract_product_title(page) -> str:
    for sel in ["h1", "h2"]:
        try:
            h = page.locator(sel).first
            if h.count() > 0 and h.is_visible():
                t = h.inner_text(timeout=2000).strip()
                if t:
                    return t
        except Exception:
            pass

    try:
        body = page.inner_text("body", timeout=5000)
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if lines:
            return lines[0][:160]
    except Exception:
        pass
    return ""


def _extract_size_options(page) -> List[str]:
    opts = []
    try:
        clickable = page.locator("a, button")
        n = clickable.count()
        for i in range(min(n, 220)):
            el = clickable.nth(i)
            try:
                if not el.is_visible():
                    continue
                txt = (el.inner_text(timeout=500) or "").strip()
                if not txt:
                    continue
                if re.search(r"\b(\d+\s*[x×]\s*\d+(?:\.\d+)?\s*(g|ml)|\d+(?:\.\d+)?\s*(g|ml)|\d+\s*(caps|capsules|softgels|softgel|gels))\b", txt, flags=re.I):
                    if txt not in opts:
                        opts.append(txt)
            except Exception:
                continue
    except Exception:
        pass
    return opts


def _normalize_size_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("×", "x")
    s = re.sub(r"\s+", "", s)
    return s


def _find_size_clickables(page) -> List[Tuple[str, object]]:
    """Return likely size-option clickables on a product page (tabs/chips)."""
    patt = re.compile(
        r"^(\s*\d+(?:\.\d+)?\s*(g|ml)\s*|"
        r"\s*\d+\s*[x×]\s*\d+(?:\.\d+)?\s*(g|ml)\s*|"
        r"\s*\d{1,4}\s*(caps|capsules|softgels|softgel|gels)\s*)$",
        re.I,
    )
    seen = set()
    out = []
    loc = page.locator("button, a, [role='tab'], [role='button']")
    try:
        n = loc.count()
    except Exception:
        n = 0
    for i in range(min(n, 260)):
        el = loc.nth(i)
        try:
            if not el.is_visible():
                continue
            txt = (el.inner_text(timeout=500) or "").strip()
            if not txt or len(txt) > 32:
                continue
            if not patt.match(txt):
                continue
            key = _normalize_size_label(txt)
            if key in seen:
                continue
            seen.add(key)
            out.append((txt, el))
        except Exception:
            continue
    return out


def _choose_best_size_label(exp: "ExpectedSpec", labels: List[str]) -> Optional[str]:
    if not labels:
        return None

    # v16: if the input file provides a Hibuddy size token, prefer exact chip/tab match
    try:
        pref: List[str] = []
        sp = getattr(exp, "size_primary", "") or ""
        if sp:
            pref.append(sp)
        for sc in (getattr(exp, "size_candidates", None) or []):
            if sc:
                pref.append(str(sc))
        if pref:
            pref_norm = [_normalize_size_label(x) for x in pref if x]
            for lab in labels:
                if _normalize_size_label(lab) in pref_norm:
                    return lab
    except Exception:
        pass

    for rx in getattr(exp, "size_regexes", []) or []:
        for lab in labels:
            if rx.search(lab):
                return lab

    try:
        exp_val = _fmt_num_for_size(exp.net)
        exp_unit = str(exp.uom).strip().lower()
        if exp_val and exp_unit:
            target = _normalize_size_label(f"{exp_val}{exp_unit}")
            for lab in labels:
                if _normalize_size_label(lab) == target:
                    return lab
    except Exception:
        pass

    return None


def _inject_size_param(url: str, size_label: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        qs["size"] = [size_label]
        new_q = urllib.parse.urlencode(qs, doseq=True)
        return urllib.parse.urlunparse(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_q, parsed.fragment)
        )
    except Exception:
        return url


def _wait_for_size_refresh(page, before_sentence: str) -> None:
    t0 = time.time()
    while time.time() - t0 < 8.0:
        try:
            page.wait_for_load_state("networkidle", timeout=800)
        except Exception:
            pass
        try:
            p = page.locator("p").filter(has_text=re.compile(r"retailers\s+within", re.I)).first
            if p and p.count() > 0:
                now = p.inner_text(timeout=800)
                if before_sentence and now and now != before_sentence:
                    return
        except Exception:
            pass
        _sleep(0.15, 0.25)


def _click_size_label(page, desired_label: str) -> Optional[str]:
    if not desired_label:
        return None
    desired_norm = _normalize_size_label(desired_label)

    before_sentence = ""
    try:
        p = page.locator("p").filter(has_text=re.compile(r"retailers\s+within", re.I)).first
        if p and p.count() > 0:
            before_sentence = p.inner_text(timeout=1500)
    except Exception:
        pass

    candidates = _find_size_clickables(page)
    if not candidates:
        return None

    for txt, el in candidates:
        if _normalize_size_label(txt) == desired_norm:
            try:
                el.click()
                _sleep(0.5, 1.0)
            except Exception:
                pass
            _wait_for_size_refresh(page, before_sentence)
            return txt

    for txt, el in candidates:
        n = _normalize_size_label(txt)
        if desired_norm in n or n in desired_norm:
            try:
                el.click()
                _sleep(0.5, 1.0)
            except Exception:
                pass
            _wait_for_size_refresh(page, before_sentence)
            return txt

    return None



def _parse_mg_from_text(text: str) -> Dict[str, List[int]]:
    return {
        "thc": _parse_mg(text, "thc") + _parse_total_mg_in_parens(text, "thc"),
        "cbd": _parse_mg(text, "cbd") + _parse_total_mg_in_parens(text, "cbd"),
    }


def _numeric_match_score(expected: List[int], candidate: List[int]) -> float:
    if not expected:
        return 0.0
    if not candidate:
        return 0.0

    exp_set = set(expected)
    cand_set = set(candidate)

    score = 0.0
    score += 6.0 * len(exp_set & cand_set)

    mismatches = list(cand_set - exp_set)
    if mismatches:
        score -= 10.0 * min(len(mismatches), 3)

    return score


def _count_match_score(expected_counts: List[int], cand_counts: List[int]) -> float:
    if not expected_counts:
        return 0.0
    if not cand_counts:
        return 0.0
    exp_set = set(expected_counts)
    cand_set = set(cand_counts)
    sc = 0.0
    sc += 5.0 * len(exp_set & cand_set)
    mism = list(cand_set - exp_set)
    if mism:
        sc -= 8.0 * min(len(mism), 3)
    return sc


def _size_match_score(exp: ExpectedSpec, size_options: List[str], url: str) -> Tuple[float, bool, Optional[str]]:
    if not exp.size_regexes:
        return (0.0, False, None)

    all_texts = list(size_options)
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        if "size" in qs and qs["size"]:
            all_texts.append(qs["size"][0])
    except Exception:
        pass

    for rx in exp.size_regexes:
        for txt in all_texts:
            if rx.search(txt):
                return (18.0, True, txt)

    return (-22.0, False, None)


def _verify_candidate_on_product_page(page, exp: ExpectedSpec, cand: Candidate) -> Tuple[float, bool, Optional[str], str]:
    url = f"https://hibuddy.ca{cand.href}"
    _safe_goto(page, url)
    _maybe_click_age_gate(page)
    _sleep(0.7, 1.4)

    title = _extract_product_title(page)
    sizes = _extract_size_options(page)

    chosen_size_txt = None
    desired_size = _choose_best_size_label(exp, sizes)
    if desired_size:
        chosen_size_txt = _click_size_label(page, desired_size) or chosen_size_txt

    final_url = page.url
    if chosen_size_txt:
        final_url = _inject_size_param(final_url, chosen_size_txt)

    score = cand.rough_score

    if _norm(exp.brand) and _norm(exp.brand) in _norm(title):
        score += 6.0

    base_name = _strip_size_from_name(exp.pname)
    try:
        import difflib
        score += 18.0 * difflib.SequenceMatcher(None, _norm(base_name), _norm(title)).ratio()
    except Exception:
        pass

    size_sc, size_ok, size_txt = _size_match_score(exp, sizes, final_url)
    score += size_sc
    if chosen_size_txt:
        size_txt = chosen_size_txt

    cand_nums = _parse_mg_from_text(title)
    score += _numeric_match_score(exp.thc_mg + exp.total_thc_mg, cand_nums.get("thc", []))
    score += _numeric_match_score(exp.cbd_mg + exp.total_cbd_mg, cand_nums.get("cbd", []))

    cand_counts = _parse_unit_counts(title)
    score += _count_match_score(exp.unit_counts, cand_counts)

    return score, size_ok, size_txt, final_url


# -------------------------
# Store count + prices
# -------------------------

def _extract_store_count_from_sentence(page) -> Optional[int]:
    try:
        p = page.locator("p").filter(has_text=re.compile(r"retailers\s+within", re.I)).first
        if p and p.count() > 0:
            txt = p.inner_text(timeout=3000)
            m = re.search(r"\b(\d{1,4})\s+retailers?\s+within\b", txt, flags=re.I)
            if m:
                return int(m.group(1))
    except Exception:
        pass

    try:
        body = page.inner_text("body", timeout=8000)
    except Exception:
        return None

    m2 = re.search(r"\b(\d{1,4})\s+retailers?\s+within\b", body, flags=re.I)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None


def _select_view_all_rows_per_page(page) -> None:
    if DISABLE_VIEW_ALL:
        return
    try:
        sel = page.locator('select[aria-label="Rows per page:"]')
        if sel.count() == 0:
            return
        options = sel.locator("option").all_text_contents()
        if any(("VIEW ALL" in (o or "").upper()) for o in options):
            sel.select_option(label=re.compile(r"VIEW\s+ALL", re.I))
            return
        vals = []
        for opt in sel.locator("option").element_handles():
            try:
                v = opt.get_attribute("value")
                if v and v.isdigit():
                    vals.append(int(v))
            except Exception:
                continue
        if vals:
            sel.select_option(value=str(max(vals)))
    except Exception:
        return


def _extract_price_value(text: str) -> Optional[float]:
    m = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def _extract_min_max_price_from_table(page) -> Tuple[Optional[float], Optional[float]]:
    """Extract min/max price from the retailer table.

    Notes:
    - Hibuddy retailer list is typically sorted by ascending price.
    - We try 'Rows per page: View all' first.
    - If we still appear to have pagination (e.g., stuck at ~10 rows), we jump to the last page
      (or click next repeatedly) and take the last row's price as max.
    """
    try:
        table = page.locator(".rdt_Table")
        if table.count() == 0:
            return None, None

        rows = page.locator(".rdt_TableBody .rdt_TableRow")
        if rows.count() == 0:
            return None, None

        # Min price should be on the first row (sorted ascending)
        first_price_cell = rows.first.locator('[data-column-id="2"]')
        min_txt = first_price_cell.inner_text(timeout=3000)
        min_price = _extract_price_value(min_txt)

        # Try to expand rows per page to VIEW ALL / max
        _select_view_all_rows_per_page(page)
        _sleep(0.6, 1.3)

        # Re-evaluate after changing rows-per-page
        rows = page.locator(".rdt_TableBody .rdt_TableRow")
        if rows.count() == 0:
            return min_price, None

        # If we got more than a typical page, take last row now.
        if rows.count() > 12:
            last_price_cell = rows.last.locator('[data-column-id="2"]')
            max_txt = last_price_cell.inner_text(timeout=3000)
            return min_price, _extract_price_value(max_txt)

        # Otherwise, use pagination to reach the last page.
        try:
            first_btn = page.locator('button[aria-label="First Page"]')
            if first_btn.count() > 0 and first_btn.first.is_enabled():
                first_btn.first.click()
                _sleep(0.5, 1.0)
        except Exception:
            pass

        try:
            last_btn = page.locator('button[aria-label="Last Page"]')
            if last_btn.count() > 0 and last_btn.first.is_enabled():
                last_btn.first.click()
                _sleep(0.8, 1.5)
            else:
                next_btn = page.locator('button[aria-label="Next Page"]')
                guard = 0
                while next_btn.count() > 0 and next_btn.first.is_enabled() and guard < 500:
                    next_btn.first.click()
                    _sleep(0.5, 1.1)
                    guard += 1
        except Exception:
            pass

        rows = page.locator(".rdt_TableBody .rdt_TableRow")
        if rows.count() == 0:
            return min_price, None

        last_price_cell = rows.last.locator('[data-column-id="2"]')
        max_txt = last_price_cell.inner_text(timeout=3000)
        return min_price, _extract_price_value(max_txt)

    except Exception:
        return None, None


def _write_debug(page, debug_dir: Path, sku: str, step: str) -> None:
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        page.screenshot(path=str(debug_dir / f"{ts}_{sku}_{step}.png"), full_page=True)
        (debug_dir / f"{ts}_{sku}_{step}.html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


def run(args) -> None:
    global DISABLE_VIEW_ALL
    DISABLE_VIEW_ALL = not bool(getattr(args, 'enable_view_all', False))
    print(f"Settings: verify_top_k={args.verify_top_k} early_stop_score={getattr(args,'early_stop_score',60)} match_threshold={args.match_threshold} block_images={getattr(args,'block_images',True)}")
    # Force-disable sidebar filtering for speed and to preserve input ordering.
    args.enable_left_filters = False
    args.enable_category_filter = False

    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = pd.read_excel(inp)

    if args.only_available_cases_gt0:
        if "Available Cases" not in df.columns:
            raise KeyError("Column 'Available Cases' not found in input.")
        df = df[df["Available Cases"].fillna(0) > 0].copy()

    if args.sort_for_efficiency:
        for col in ["Format", "Brand Name"]:
            if col not in df.columns:
                df[col] = ""
        df["_fmt_sort"] = df["Format"].fillna("").astype(str)
        df["_brand_sort"] = df["Brand Name"].fillna("").astype(str)
        df.sort_values(by=["_fmt_sort", "_brand_sort"], inplace=True, kind="stable")

    done = set()
    if out.exists() and args.resume:
        try:
            prev = pd.read_csv(out, dtype=str)
            if "AGLC SKU" in prev.columns:
                done = set(prev["AGLC SKU"].astype(str).tolist())
        except Exception:
            pass

    header = [
        "AGLC SKU",
        "Brand Name",
        "Product Name",
        "Format",
        "Net Content",
        "Content UOM",
        "hibuddy_category_filter",
        "hibuddy_search_query",
        "filters_applied",
        "hibuddy_retailers_within_radius",
        "hibuddy_min_price",
        "hibuddy_max_price",
        "hibuddy_product_url",
        "hibuddy_size_selected",
        "match_score",
        "status",
    ]
    if not out.exists() or not args.resume:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    required_base = ["AGLC SKU", "Brand Name", "Net Content", "Content UOM"]


    for col in required_base:


        if col not in df.columns:


            raise KeyError(f"Required column missing: {col}")



    # Accept either Product Name or SKU Description


    if "Product Name" not in df.columns and "SKU Description" not in df.columns:


        raise KeyError("Required column missing: Product Name (or SKU Description)")



    # Normalize so both exist


    if "Product Name" not in df.columns and "SKU Description" in df.columns:


        df["Product Name"] = df["SKU Description"].astype(str)


    if "SKU Description" not in df.columns and "Product Name" in df.columns:


        df["SKU Description"] = df["Product Name"].astype(str)
    df["AGLC SKU"] = df["AGLC SKU"].astype(str)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=args.headless,
            viewport={"width": 1500, "height": 950},
        )
        page = context.new_page()

        _safe_goto(page, HIBUDDY_PRODUCTS_ALL_URL)
        _maybe_click_age_gate(page)
        _ensure_location_is_set_once(page, interactive=args.interactive_setup)

        if args.block_images:
            _enable_resource_blocking(context)
            print("Speed mode: blocking images/media/fonts during scraping.")


        processed = 0

        # Sidebar filters are disabled in this build for speed.

        for _, row in df.iterrows():
            sku = str(row["AGLC SKU"])
            if sku in done:
                continue

            brand = str(row["Brand Name"])
            pname = str(row["Product Name"])
            net = row["Net Content"]
            uom = row["Content UOM"]
            fmt = str(row["Format"]) if "Format" in df.columns else ""

            sku_desc = str(row["SKU Description"]) if "SKU Description" in df.columns else pname
            # If you used the shaper (or manually edited it), these columns can drive matching precisely:
            size_primary = ""
            if "hibuddy_size_primary" in df.columns:
                size_primary = str(row.get("hibuddy_size_primary", "")).strip()

            size_candidates = []
            if "hibuddy_size_candidates" in df.columns:
                raw = str(row.get("hibuddy_size_candidates", "") or "")
                # accept either ";" or "," separated lists
                parts = [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]
                size_candidates = parts

            exp = _expected_spec(brand, sku_desc, net, uom, size_primary=size_primary, size_candidates=size_candidates)

            query = ""
            if "hibuddy_search_query" in df.columns:
                query = str(row.get("hibuddy_search_query", "")).strip()
            if not query:
                query = f"{brand} {sku_desc}".strip()
            store_count = None
            min_price = None
            max_price = None
            hibuddy_url = ""
            chosen_size = ""
            match_score = ""
            status = "ok"
            category_label = ""

            try:
                ok = _safe_search_to_grid(page, query, attempts=3)
                if not ok:
                    raise RuntimeError("search_failed_or_gateway")
                _sleep(args.min_delay, args.max_delay)

                candidates = _collect_candidates_from_grid(page, exp, limit=args.grid_candidate_limit)
                if not candidates:
                    status = "no_results_grid"
                    if debug_dir:
                        _write_debug(page, debug_dir, sku, "no_results_grid")
                else:
                    best = None
                    for cand in candidates[:args.verify_top_k]:
                        sc, size_ok, size_txt, url = _verify_candidate_on_product_page(page, exp, cand)
                        if best is None or sc > best[0]:
                            best = (sc, size_ok, size_txt or "", url)

                        # Speed: if we already found a strong match, stop opening more candidates
                        if sc >= args.early_stop_score:
                            if (not args.require_size_match) or (not exp.size_regexes) or size_ok:
                                break

                        _sleep(0.5, 1.1)

                    if best is None:
                        status = "no_product_match"
                    else:
                        sc, size_ok, size_txt, url = best
                        match_score = f"{sc:.2f}"
                        chosen_size = size_txt or ""

                        accept = sc >= args.match_threshold
                        if args.require_size_match and exp.size_regexes and not size_ok:
                            accept = False

                        # Always keep the best candidate URL (even if below threshold)
                        hibuddy_url = url

                        if not accept:
                            # Keep URL + score so you can review the "best available" match.
                            status = "below_threshold"

                        # Ensure we're on the WINNING candidate page before extracting store count / prices.
                        try:
                            _safe_goto(page, hibuddy_url)
                            _maybe_click_age_gate(page)
                            _sleep(0.6, 1.2)
                            # Ensure correct size variant is selected (Hibuddy may not change URL on size switch)
                            try:
                                if size_txt:
                                    _click_size_label(page, size_txt)
                            except Exception:
                                pass

                        except Exception:
                            pass

                        store_count = _extract_store_count_from_sentence(page)
                        if args.capture_prices:
                            min_price, max_price = _extract_min_max_price_from_table(page)

                        if store_count is None:
                            status = "no_store_count_found"
                            if debug_dir:
                                _write_debug(page, debug_dir, sku, "no_store_count")

            except PlaywrightTimeoutError:
                status = "timeout"
                if debug_dir:
                    _write_debug(page, debug_dir, sku, "timeout")
            except Exception as e:
                status = f"error:{type(e).__name__}"
                if debug_dir:
                    _write_debug(page, debug_dir, sku, "error")
            filters_applied_str = ""

            with out.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    sku,
                    brand,
                    pname,
                    fmt,
                    net,
                    uom,
                    category_label,
                    query,
                    filters_applied_str,
                    store_count,
                    min_price,
                    max_price,
                    hibuddy_url,
                    chosen_size,
                    match_score,
                    status,
                ])

            processed += 1
            if args.max_items and processed >= args.max_items:
                break

            _sleep(args.min_delay, args.max_delay)

        context.close()


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Alberta Assortment Excel (.xlsx)")
    ap.add_argument("--output", default="hibuddy_calgary_storecounts_v16.csv", help="Output CSV")
    ap.add_argument("--profile-dir", default="hibuddy_profile", help="Persistent browser profile directory")
    ap.add_argument("--headless", action="store_true", help="Run browser headless (recommended after setup)")
    ap.set_defaults(block_images=True)
    ap.add_argument("--block-images", dest="block_images", action="store_true", help="(Default) Block images/media/fonts after setup to speed up")
    ap.add_argument("--no-block-images", dest="block_images", action="store_false", help="Do not block images/media/fonts")

    ap.add_argument("--enable-view-all", action="store_true", default=False, help="Try to set Rows per page to View all on retailer tables (slower). Default is off for speed.")
    ap.add_argument("--interactive-setup", action="store_true", help="Let you set Calgary+radius once, then continue")
    ap.add_argument("--only-available-cases-gt0", action="store_true", default=True, help="Default: Available Cases > 0")
    ap.add_argument("--resume", action="store_true", default=True, help="Skip SKUs already written to output CSV")
    ap.add_argument("--min-delay", type=float, default=2.8, help="Min random delay between steps (seconds)")
    ap.add_argument("--max-delay", type=float, default=6.5, help="Max random delay between steps (seconds)")
    ap.add_argument("--max-items", type=int, default=0, help="Stop after N items (0 = no limit)")
    ap.add_argument("--debug-dir", default="", help="Folder to save screenshot+html on failures")

    ap.add_argument("--grid-candidate-limit", type=int, default=60, help="How many grid cards to score")
    ap.add_argument("--verify-top-k", type=int, default=5, help="How many top candidates to open/verify")
    ap.add_argument("--early-stop-score", type=float, default=60.0, help="Stop verifying more candidates early if score reaches this value (default: 60)")

    ap.add_argument("--match-threshold", type=float, default=42.0, help="Minimum strict score to accept a match")
    ap.add_argument("--require-size-match", action="store_true", default=True, help="Require a size match when size info exists")
    ap.add_argument("--capture-prices", action="store_true", default=True, help="Capture min/max price")

    ap.add_argument("--enable-left-filters", action="store_true", default=False, help="Use left sidebar filters when possible")
    ap.add_argument("--enable-category-filter", action="store_true", default=False, help="Apply category filter based on AGLC Format")
    ap.add_argument("--enable-brand-filter", action="store_true", default=False, help="(Optional) Apply brand filter via sidebar. Default OFF; we search using Brand+SKU Description instead.")
    ap.add_argument("--sort-for-efficiency", action="store_true", default=False, help="Sort rows by (Format, Brand) to reduce filter switching")

    args = ap.parse_args(argv)
    if args.max_items == 0:
        args.max_items = None
    if args.debug_dir == "":
        args.debug_dir = None
    return args


if __name__ == "__main__":
    try:
        run(parse_args(sys.argv[1:]))
        print("Done.")
    except KeyboardInterrupt:
        print("\nStopped by user. You can rerun with --resume to continue.")
        raise
