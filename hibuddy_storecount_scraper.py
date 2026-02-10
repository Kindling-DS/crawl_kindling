#!/usr/bin/env python3
from __future__ import annotations

"""
hibuddy_storecount_scraper_v33_candidates_products_plural_size_priority.py

What this fixes (vs v28 you were running)
----------------------------------------
1) **Clicks / navigates to REAL product pages** by only collecting links that start with:
      /product/   (singular)
   Your v28 was using /products/ (plural) which includes listing pages like:
      /products/deals, /products/flower, etc.
   That is why your hibuddy_product_url was repeatedly https://hibuddy.ca/products/deals.

2) Captures **ALL retailer rows** on the product page.
   - Writes ONE output row PER retailer ("exploded" dataset) with:
       store_name, address_full, address_street, price, distance
   - Also keeps the full list as a JSON column (repeated per row) for debugging/back-compat:
       hibuddy_store_prices_json

3) Always scans the first top-K product candidates (default 5) and picks the best-scoring one.
   This avoids missing the correct product that is right next to the top search result.

4) Also outputs the cheapest (“best”) retailer as convenience columns.

5) Keeps your existing workflow:
   - Search on /products/all
   - Collect candidate products from the grid
   - Verify top-K by opening each product page
   - Click requested size if possible
   - Extract retailer count + retailer table

Notes
-----
- Hibuddy DOM can change. This code uses fallbacks.
- Best results if you do --interactive-setup once and set:
    Location + radius, and "Rows per page: VIEW ALL" on a product page.
"""

import argparse
import csv
import json
import os
import urllib.parse
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright

HIBUDDY_PRODUCTS_ALL_URL = "https://hibuddy.ca/products/all"
_BAD_STATUS_CODES = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526}


def _sleep(a: float = 0.25, b: float = 0.6) -> None:
    time.sleep(random.uniform(a, b))


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("×", "x")
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> set:
    s = _norm(s)
    toks = re.findall(r"[a-z0-9]+", s)
    return set(toks)


def _normalize_size_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("×", "x")
    s = re.sub(r"\s+", "", s)
    return s

def _size_matches_requested(size_primary: str, size_selected: str, size_active: str, size_options: List[str]) -> bool:
    """
    True if the requested size matches anything Hibuddy shows for this candidate.

    We compare against:
      - size_selected (what we clicked)
      - size_active   (active tab)
      - size_options  (all detected tabs/options)
    using normalized equality + light containment.
    """
    if not size_primary:
        return False
    want = _normalize_size_label(size_primary)
    if not want:
        return False

    candidates: List[str] = []
    if size_selected:
        candidates.append(size_selected)
    if size_active and size_active not in candidates:
        candidates.append(size_active)
    for o in (size_options or []):
        if o and o not in candidates:
            candidates.append(o)

    for c in candidates:
        if _normalize_size_label(c) == want:
            return True

    for c in candidates:
        n = _normalize_size_label(c)
        if not n:
            continue
        if want in n or n in want:
            return True

    return False


def _page_looks_like_gateway(page) -> bool:
    try:
        t = (page.title() or "").lower()
        if "bad gateway" in t or ("error" in t and "cloudflare" in t):
            return True
    except Exception:
        pass
    try:
        body = page.locator("body")
        if body.count() > 0:
            txt = body.inner_text(timeout=1500).lower()
            if "bad gateway" in txt or "cloudflare" in txt:
                return True
            if re.search(r"\b502\b.*bad gateway", txt, flags=re.I):
                return True
    except Exception:
        pass
    return False


def _recover_from_gateway(page, base_url: str = HIBUDDY_PRODUCTS_ALL_URL) -> None:
    try:
        page.reload(wait_until="domcontentloaded", timeout=20000)
        _sleep(0.6, 1.2)
        if not _page_looks_like_gateway(page):
            return
    except Exception:
        pass

    try:
        page.go_back(wait_until="domcontentloaded", timeout=20000)
        _sleep(0.6, 1.2)
        if not _page_looks_like_gateway(page):
            return
    except Exception:
        pass

    try:
        page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
        _sleep(0.6, 1.2)
    except Exception:
        pass


def _safe_goto(page, url: str, attempts: int = 3) -> None:
    last_err = None
    for i in range(attempts):
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if resp is not None and getattr(resp, "status", None) in _BAD_STATUS_CODES:
                raise RuntimeError(f"HTTP {resp.status}")
            if _page_looks_like_gateway(page):
                raise RuntimeError("gateway_page")
            return
        except Exception as e:
            last_err = e
            _recover_from_gateway(page, base_url=HIBUDDY_PRODUCTS_ALL_URL)
            try:
                page.wait_for_timeout(700 + i * 900)
            except Exception:
                _sleep(0.7 + i * 0.7, 1.4 + i * 0.9)
    if last_err:
        raise last_err


def _maybe_click_age_gate(page) -> None:
    try:
        btn = page.get_by_role("button", name=re.compile(r"^\s*yes\s*$", re.I))
        if btn.count() > 0 and btn.first.is_visible():
            btn.first.click()
    except Exception:
        pass


def _enable_resource_blocking(context) -> None:
    def _route(route, request):
        try:
            rtype = request.resource_type
            if rtype in ("image", "media", "font"):
                return route.abort()
            url = request.url.lower()
            if url.endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".gif",
                    ".svg",
                    ".mp4",
                    ".webm",
                    ".mov",
                    ".avi",
                    ".woff",
                    ".woff2",
                    ".ttf",
                    ".otf",
                )
            ):
                return route.abort()
        except Exception:
            pass
        return route.continue_()

    try:
        context.route("**/*", _route)
    except Exception:
        pass


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
    return best


def _safe_search_to_grid(page, query: str, attempts: int = 3) -> bool:
    """
    Search Hibuddy and ensure we have navigable product candidates.

    We try two entry points:
      1) /products/all using the header search box
      2) fallback direct /products/search?q=<query>

    Candidate detection is done by scanning anchors and normalizing hrefs via
    _normalize_product_href (supports both '/product/...' and some '/products/<slug>').

    On /products/search pages, Hibuddy may default to Deals/Popular views; we try to
    click an "All Products" tab/button if it exists.
    """
    def _ensure_all_products_view() -> None:
        # Best-effort: click "All Products" (or similar) if present
        try:
            for role in ("button", "link"):
                btn = page.get_by_role(role, name=re.compile(r"^\s*all\s+products\s*$", re.I))
                if btn.count() > 0 and btn.first.is_visible():
                    btn.first.click()
                    _sleep(0.25, 0.5)
                    break
        except Exception:
            pass

    def _has_candidate_links() -> bool:
        try:
            anchors = page.locator("main a[href]")
            if anchors.count() == 0:
                anchors = page.locator("a[href]")
            n = min(anchors.count(), 250)
            for i in range(n):
                try:
                    href = anchors.nth(i).get_attribute("href") or ""
                    if _normalize_product_href(href):
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    for _ in range(attempts):
        try:
            # (1) stable start: /products/all
            if not page.url.startswith(HIBUDDY_PRODUCTS_ALL_URL):
                _safe_goto(page, HIBUDDY_PRODUCTS_ALL_URL)
                _maybe_click_age_gate(page)
                _sleep(0.3, 0.7)

            inp = _find_header_search_input(page)
            if inp is not None:
                inp.click()
                inp.fill("")
                _sleep(0.05, 0.15)
                inp.type(query, delay=random.randint(15, 35))
                _sleep(0.15, 0.3)
                inp.press("Enter")

                page.wait_for_load_state("domcontentloaded", timeout=25000)
                _sleep(0.6, 1.2)
                try:
                    page.mouse.wheel(0, 1400)
                    _sleep(0.2, 0.4)
                except Exception:
                    pass

                _ensure_all_products_view()
                if _has_candidate_links():
                    return True

            # (2) fallback direct search page
            q = urllib.parse.quote_plus(query)
            search_url = f"https://hibuddy.ca/products/search?q={q}"
            _safe_goto(page, search_url)
            _maybe_click_age_gate(page)
            page.wait_for_load_state("domcontentloaded", timeout=25000)
            _sleep(0.6, 1.2)

            try:
                page.mouse.wheel(0, 1600)
                _sleep(0.2, 0.5)
            except Exception:
                pass

            _ensure_all_products_view()
            if _has_candidate_links():
                return True

        except Exception:
            _recover_from_gateway(page, base_url=HIBUDDY_PRODUCTS_ALL_URL)
            _sleep(0.6, 1.1)

    return False

@dataclass
class ExpectedSpec:
    search_query: str
    size_primary: str


@dataclass
class Candidate:
    title: str
    href: str
    rough_score: float


def _extract_product_title(page) -> str:
    try:
        h1 = page.locator("h1")
        if h1.count() > 0:
            t = (h1.first.inner_text(timeout=3000) or "").strip()
            if t:
                return t
    except Exception:
        pass
    try:
        for sel in ["h2", "h3"]:
            h = page.locator(sel)
            if h.count() > 0:
                t = (h.first.inner_text(timeout=2000) or "").strip()
                if t:
                    return t
    except Exception:
        pass
    return ""


def _find_size_clickables(page) -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []
    try:
        clickable = page.locator("a, button")
        n = clickable.count()
        for i in range(min(n, 260)):
            el = clickable.nth(i)
            try:
                if not el.is_visible():
                    continue
                txt = (el.inner_text(timeout=500) or "").strip()
                if not txt:
                    continue
                if re.search(
                    r"\b(\d+(?:\.\d+)?\s*(g|ml)|\d+\s*[x×]\s*\d+(?:\.\d+)?\s*(g|ml)|\d+\s*cap|\d+\s*caps|\d+\s*pack)\b",
                    txt,
                    flags=re.I,
                ):
                    out.append((txt, el))
            except Exception:
                continue
    except Exception:
        pass
    return out


def _looks_like_product_page(page) -> bool:
    """
    Hard gate to reject navigation/listing pages.
    A real product page almost always has at least one of:
    - the "X retailers within" sentence
    - the retailer table (.rdt_Table)
    - size chips/buttons
    """
    try:
        if page.locator(".rdt_Table").count() > 0:
            return True
    except Exception:
        pass

    try:
        if page.locator("p").filter(has_text=re.compile(r"retailers\s+within", re.I)).count() > 0:
            return True
    except Exception:
        pass

    try:
        if _find_size_clickables(page):
            return True
    except Exception:
        pass

    return False


def _rough_match_score(exp: ExpectedSpec, cand_title: str) -> float:
    q = _norm(exp.search_query)
    t = _norm(cand_title)
    if not q or not t:
        return 0.0
    if q == t:
        return 90.0

    q_toks = _tokenize(q)
    t_toks = _tokenize(t)
    if not q_toks or not t_toks:
        return 0.0
    inter = len(q_toks & t_toks)
    union = len(q_toks | t_toks)
    jacc = inter / max(union, 1)
    score = 100.0 * jacc
    if q_toks <= t_toks:
        score += 12.0
    if len(t_toks) <= 3 and jacc < 0.4:
        score -= 8.0
    return max(0.0, min(100.0, score))



def _normalize_product_href(href: str) -> Optional[str]:
    """
    Normalize candidate href into a relative path we can navigate to.

    Hibuddy has (at least) two patterns in the wild:
      - Product detail:   /product/<...>        (singular)
      - Sometimes:        /products/<slug>      (plural)  (can be product OR category/deals)

    We therefore accept:
      - any href containing '/product/'  -> normalize to '/product/...'
      - '/products/<slug>'              -> keep as-is ONLY if <slug> is not a known listing slug

    Final validation happens later via _looks_like_product_page(page).
    """
    href = (href or "").strip()
    if not href:
        return None

    path = href
    query = ""

    # Absolute URL -> path + query
    if href.startswith("http://") or href.startswith("https://"):
        try:
            from urllib.parse import urlparse

            p = urlparse(href)
            path = p.path or ""
            query = p.query or ""
        except Exception:
            return None
    else:
        # Split query for relative hrefs
        if "?" in href:
            path, query = href.split("?", 1)
        else:
            path, query = href, ""

    # Drop fragment
    path = path.split("#", 1)[0]
    if not path:
        return None

    # Reject known search/listing endpoints outright
    low_path = path.lower()
    if low_path.startswith("/products/search") or low_path.startswith("/products/all"):
        return None

    # (1) Prefer singular /product/
    if "/product/" in path:
        i = path.find("/product/")
        norm = path[i:]
        if not norm.startswith("/product/"):
            return None
        # keep query if present (sometimes used for variants)
        if query:
            norm = norm + "?" + query
        return norm

    # (2) Accept /products/<slug> ONLY if not a listing/category slug
    if path.startswith("/products/"):
        parts = [p for p in path.split("/") if p]
        # Expect at least: ["products", "<slug>"]
        if len(parts) < 2:
            return None
        slug = parts[1].lower()

        # Common non-product slugs (listing pages)
        blacklist = {
            "deals",
            "deal",
            "flower",
            "extracts",
            "vapes",
            "pre-rolls",
            "pre-roll",
            "prerolls",
            "edibles",
            "topicals",
            "accessories",
            "gear",
            "brands",
            "stores",
            "store",
            "locations",
            "location",
            "popular",
            "new",
            "news",
        }
        if slug in blacklist:
            return None

        # If it's deeper than /products/<slug>, keep it, but still apply gate later.
        norm = path
        if query:
            norm = norm + "?" + query
        return norm

    return None

def _collect_candidates_from_grid(page, exp: ExpectedSpec, limit: int = 10) -> List[Candidate]:
    """
    Collect candidate links from the current result page.

    IMPORTANT:
    - We do NOT assume candidates are only '/product/...'. On /products/search?q=... Hibuddy
      sometimes uses '/products/<slug>' product links.
    - We normalize hrefs via _normalize_product_href and later hard-gate with _looks_like_product_page.

    Implementation:
    - Prefer anchors inside <main> if present to reduce nav/footer noise.
    - Iterate a bounded number of anchors and keep unique normalized hrefs.
    """
    out: List[Candidate] = []

    anchors = page.locator("main a[href]")
    try:
        if anchors.count() == 0:
            anchors = page.locator("a[href]")
    except Exception:
        anchors = page.locator("a[href]")

    n = min(anchors.count(), 400)
    seen = set()

    for i in range(n):
        a = anchors.nth(i)
        try:
            raw_href = a.get_attribute("href") or ""
            href = _normalize_product_href(raw_href)
            if not href or href in seen:
                continue
            seen.add(href)

            # Prefer a compact title from child headings, otherwise fallback to anchor text.
            txt = ""
            try:
                h = a.locator("h1, h2, h3").first
                if h.count() > 0:
                    txt = (h.inner_text(timeout=500) or "").strip()
            except Exception:
                pass
            if not txt:
                try:
                    txt = (a.inner_text(timeout=800) or "").strip()
                except Exception:
                    txt = ""

            txt = re.sub(r"\s+", " ", txt)

            # Some anchors wrap a whole card; if the text is too long, trim to first line-ish
            if len(txt) > 160:
                txt = txt[:160]

            if len(txt) < 4:
                alt = (a.get_attribute("aria-label") or a.get_attribute("title") or "").strip()
                alt = re.sub(r"\s+", " ", alt)
                txt = alt if alt else txt

            if len(txt) < 4:
                continue

            score = _rough_match_score(exp, txt)
            out.append(Candidate(title=txt, href=href, rough_score=score))
        except Exception:
            continue

    out.sort(key=lambda c: c.rough_score, reverse=True)
    return out[:limit]

def _wait_for_size_refresh(page, before_sentence: str) -> None:
    t0 = time.time()
    while time.time() - t0 < 10.0:
        try:
            page.wait_for_load_state("networkidle", timeout=900)
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

    # exact normalized match
    for txt, el in candidates:
        if _normalize_size_label(txt) == desired_norm:
            try:
                el.click()
                _sleep(0.35, 0.75)
            except Exception:
                pass
            _wait_for_size_refresh(page, before_sentence)
            return txt

    # containment match fallback
    for txt, el in candidates:
        n = _normalize_size_label(txt)
        if desired_norm and (desired_norm in n or n in desired_norm):
            try:
                el.click()
                _sleep(0.35, 0.75)
            except Exception:
                pass
            _wait_for_size_refresh(page, before_sentence)
            return txt

    return None



def _extract_size_options_and_active(page) -> Tuple[List[str], str]:
    """Extract available size/weight tabs (e.g., '7x0.5g') and the active one.

    Hibuddy commonly renders sizes as:
      <a class="tab ... tab-active">7x0.5g</a>
    but may also use buttons.

    Returns:
      (options_list, active_label)
    """
    options: List[str] = []
    active = ""

    # Prefer dedicated tab controls
    try:
        tabs = page.locator("a.tab, button.tab")
        n = min(tabs.count(), 120)
        for i in range(n):
            el = tabs.nth(i)
            try:
                if not el.is_visible():
                    continue
                txt = (el.inner_text(timeout=500) or "").strip()
                txt = re.sub(r"\s+", " ", txt)
                if not txt:
                    continue
                # Keep only things that look like a size/weight/pack/caps label
                if not re.search(
                    r"(\d+\s*[x×]\s*\d+(?:\.\d+)?\s*(g|ml)|\d+(?:\.\d+)?\s*(g|ml)|\d+\s*cap|\d+\s*caps|\d+\s*pack)",
                    txt,
                    flags=re.I,
                ):
                    continue
                if txt not in options:
                    options.append(txt)

                cls = (el.get_attribute("class") or "")
                aria = (el.get_attribute("aria-selected") or "").lower()
                if ("tab-active" in cls) or (aria == "true"):
                    active = txt
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: reuse size clickables extractor (more permissive)
    if not options:
        try:
            for txt, _el in _find_size_clickables(page):
                if txt and txt not in options:
                    options.append(txt)
        except Exception:
            pass

    if not active and options:
        active = options[0]

    return options, active


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
        m2 = re.search(r"\b(\d{1,4})\s+retailers?\s+within\b", body, flags=re.I)
        if m2:
            return int(m2.group(1))
    except Exception:
        pass
    return None


def _extract_price_value(text: str) -> Optional[float]:
    m = re.search(r"\$\s*([\d,]+(?:\.\d{2})?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def _extract_row_fields(row) -> Dict[str, object]:
    """
    Extract one retailer row with best-effort:
      store_name: prefers <p class*='link-underline'> ... </p>
      address:    prefers <p class*='link-primary' class*='block'> ... </p>
      distance:   prefers data-column-id=1
      price:      prefers data-column-id=2

    Output fields:
      store_name, address_full, address_street, price, distance
    """
    store_name = ""
    address_full = ""
    address_street = ""
    distance = ""
    price_val: Optional[float] = None

    # (1) Preferred DOM nodes (your provided HTML)
    try:
        nm = row.locator("p[class*='link-underline']")
        if nm.count() > 0:
            txt = (nm.first.inner_text(timeout=1500) or "").strip()
            txt = re.sub(r"\s+", " ", txt)
            if txt:
                store_name = txt
    except Exception:
        pass

    try:
        addr = row.locator("p[class*='link-primary'][class*='block']")
        if addr.count() > 0:
            txt = (addr.first.inner_text(timeout=1500) or "").strip()
            txt = re.sub(r"\s+", " ", txt)
            if txt:
                address_full = txt
                address_street = txt.split(",", 1)[0].strip()
    except Exception:
        pass

    # (2) stable column ids (if present)
    try:
        if not store_name:
            s_txt = row.locator("[data-column-id='0']").inner_text(timeout=1500).strip()
            if s_txt:
                store_name = re.sub(r"\s+", " ", s_txt)
    except Exception:
        pass

    try:
        d_txt = row.locator("[data-column-id='1']").inner_text(timeout=1500).strip()
        if d_txt:
            distance = re.sub(r"\s+", " ", d_txt)
    except Exception:
        pass

    try:
        p_txt = row.locator("[data-column-id='2']").inner_text(timeout=1500).strip()
        pv = _extract_price_value(p_txt)
        if pv is not None:
            price_val = pv
    except Exception:
        pass

    # (3) fallback: cell scan
    if (not store_name) or (price_val is None) or (not distance):
        try:
            cells = row.locator(".rdt_TableCell")
            texts = []
            for i in range(min(cells.count(), 12)):
                t = (cells.nth(i).inner_text(timeout=800) or "").strip()
                if t:
                    texts.append(re.sub(r"\s+", " ", t))

            if price_val is None:
                for t in texts:
                    if "$" in t:
                        pv = _extract_price_value(t)
                        if pv is not None:
                            price_val = pv
                            break

            if not store_name:
                for t in texts:
                    if "$" in t:
                        continue
                    if re.search(r"\b(km|mi|miles?)\b", t, flags=re.I):
                        continue
                    store_name = t
                    break

            if not distance:
                for t in texts:
                    if re.search(r"\b(km|mi|miles?)\b", t, flags=re.I):
                        distance = t
                        break
        except Exception:
            pass

    return {
        "store_name": store_name or None,
        "address_full": address_full or None,
        "address_street": address_street or None,
        "price": price_val,
        "distance": distance or None,
    }


def _table_has_rows(page) -> bool:
    try:
        return page.locator(".rdt_TableBody .rdt_TableRow").count() > 0
    except Exception:
        return False


def _extract_all_store_prices_from_table(page, max_pages: int = 120, max_rows: int = 5000) -> List[Dict[str, object]]:
    """
    Collect all retailer rows across pagination.
    If "VIEW ALL" is set, often everything is on the first page.
    Otherwise, click Next Page until disabled.
    """
    try:
        if page.locator(".rdt_Table").count() == 0:
            return []
    except Exception:
        return []

    out: List[Dict[str, object]] = []
    seen = set()

    # Wait briefly for table population
    t0 = time.time()
    while time.time() - t0 < 10 and not _table_has_rows(page):
        _sleep(0.2, 0.4)

    for _ in range(max_pages):
        rows = page.locator(".rdt_TableBody .rdt_TableRow")
        rc = rows.count()
        if rc == 0:
            break

        for i in range(rc):
            row = rows.nth(i)
            rec = _extract_row_fields(row)

            key = (
                rec.get("store_name") or "",
                rec.get("address_street") or "",
                rec.get("price") or "",
                rec.get("distance") or "",
            )
            if key in seen:
                continue
            seen.add(key)

            out.append(rec)
            if len(out) >= max_rows:
                return out

        # Next page
        try:
            next_btn = page.locator('button[aria-label="Next Page"]')
            if next_btn.count() > 0 and next_btn.first.is_enabled():
                next_btn.first.click()
                _sleep(0.45, 0.9)
                page.wait_for_load_state("domcontentloaded", timeout=8000)
                _sleep(0.2, 0.4)
                continue
        except Exception:
            pass

        break

    return out


def _pick_best_store_record(store_rows: List[Dict[str, object]]) -> Tuple[Optional[Dict[str, object]], Optional[float]]:
    best_row: Optional[Dict[str, object]] = None
    best_price: Optional[float] = None
    for r in store_rows:
        p = r.get("price")
        if isinstance(p, (int, float)):
            fp = float(p)
            if best_price is None or fp < best_price:
                best_price = fp
                best_row = r
    return best_row, best_price



def _verify_candidate_on_product_page(
    page,
    exp: ExpectedSpec,
    cand: Candidate,
) -> Tuple[float, bool, str, str, str, List[str], Optional[int], List[Dict[str, object]]]:
    """
    Returns:
      (score, size_ok, final_url, size_selected, size_active, size_options, store_count, store_rows)
    """
    url = f"https://hibuddy.ca{cand.href}"
    _safe_goto(page, url)
    _maybe_click_age_gate(page)
    _sleep(0.6, 1.1)

    # hard gate: ensure we are actually on a product page
    if not _looks_like_product_page(page):
        return 0.0, False, page.url, "", "", [], None, []

    title = _extract_product_title(page)

    # sizes BEFORE click (sometimes useful if click fails)
    size_options, size_active = _extract_size_options_and_active(page)

    size_selected = ""
    size_ok = False
    if exp.size_primary:
        clicked = _click_size_label(page, exp.size_primary)
        if clicked:
            size_selected = clicked
            size_ok = _normalize_size_label(clicked) == _normalize_size_label(exp.size_primary)
            # after click, refresh size options/active
            size_options, size_active = _extract_size_options_and_active(page)

    # if still no active size, fall back to selected
    if not size_active and size_selected:
        size_active = size_selected

    score = cand.rough_score
    try:
        import difflib

        score += 35.0 * difflib.SequenceMatcher(None, _norm(exp.search_query), _norm(title)).ratio()
    except Exception:
        pass

    # boost if requested size matches
    if exp.size_primary and size_ok:
        score += 12.0
    elif exp.size_primary and not size_selected:
        score -= 5.0

    store_count = _extract_store_count_from_sentence(page)
    store_rows = _extract_all_store_prices_from_table(page)

    return score, size_ok, page.url, size_selected, size_active, size_options, store_count, store_rows


def _detect_input_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    sku_aliases = {"sku", "product sku", "item sku", "aglc sku", "ocs sku"}
    sku_col = None
    for c in cols:
        if _norm(c) in sku_aliases:
            sku_col = c
            break
    if not sku_col:
        sku_col = cols[0]

    query_col = None
    for c in cols:
        if _norm(c) in {"hibuddy_search_query", "search query", "hibuddy query"}:
            query_col = c
            break
    if not query_col:
        query_col = "hibuddy_search_query"

    size_col = None
    for c in cols:
        if _norm(c) in {"hibuddy_size_primary", "size", "primary size", "hibuddy size"}:
            size_col = c
            break
    if not size_col:
        size_col = "hibuddy_size_primary"

    missing = [c for c in [query_col, size_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")

    return sku_col, query_col, size_col


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Excel (.xlsx) containing SKU + hibuddy_search_query + hibuddy_size_primary")
    ap.add_argument("--output", default="hibuddy_storecounts.csv", help="Output CSV file (appends if exists)")
    ap.add_argument("--profile-dir", default="hibuddy_profile", help="Persistent browser profile directory")
    ap.add_argument("--headless", action="store_true", help="Run browser headless (recommended after setup)")
    ap.add_argument("--interactive-setup", action="store_true", help="Open browser and let you set location/radius & 'View all' once")
    ap.add_argument("--block-images", action="store_true", default=True, help="Block images/media/fonts AFTER interactive setup (default: on)")
    ap.add_argument("--no-block-images", action="store_false", dest="block_images", help="Do not block images/media/fonts during scraping")
    ap.add_argument("--resume", action="store_true", help="Skip SKUs already present in output CSV")
    ap.add_argument(
        "--verify-top-k",
        type=int,
        default=5,
        help="How many candidates to verify on product pages (default: 5). We always scan all top-K candidates (no early stop).",
    )
    ap.add_argument("--match-threshold", type=float, default=15.0, help="Accept match if score >= threshold (default: 15)")
    # NOTE: kept for backward compatibility, but we intentionally do NOT early-stop.
    ap.add_argument("--early-stop-score", type=float, default=60.0, help="(Ignored) kept for backward compatibility")
    ap.add_argument("--debug-dir", default="", help="If set, saves screenshots/html on failures into this directory")
    return ap.parse_args(argv)


def _load_done_set(out: Path) -> set:
    if not out.exists():
        return set()
    try:
        prev = pd.read_csv(out, dtype=str)
        if "SKU" in prev.columns:
            return set(prev["SKU"].astype(str).tolist())
    except Exception:
        pass
    return set()


def _write_debug(page, debug_dir: Path, sku: str, step: str) -> None:
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        page.screenshot(path=str(debug_dir / f"{ts}_{sku}_{step}.png"), full_page=True)
        (debug_dir / f"{ts}_{sku}_{step}.html").write_text(page.content(), encoding="utf-8")
    except Exception:
        pass


def run(args: argparse.Namespace) -> None:
    inp = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    profile_dir = Path(args.profile_dir).expanduser().resolve()
    debug_dir = Path(args.debug_dir).expanduser().resolve() if args.debug_dir else None

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    df = pd.read_excel(inp)
    sku_col, query_col, size_col = _detect_input_cols(df)

    done = _load_done_set(out) if args.resume else set()

    # Output is "exploded": one row per (SKU x retailer). This lets you build a proper dataset
    # without having to parse hibuddy_store_prices_json.
    header = [
        "SKU",
        "hibuddy_search_query",
        "hibuddy_size_primary",
        "hibuddy_product_url",
        "hibuddy_size_selected",
        "hibuddy_size_active",
        "hibuddy_size_options_json",
        "match_score",
        "status",
        "hibuddy_retailers_within_radius",
        "hibuddy_best_store_name",
        "hibuddy_best_store_price",
        "hibuddy_best_store_address_full",
        "hibuddy_best_store_address_street",
        "store_row_index",
        "store_name",
        "address_full",
        "address_street",
        "price",
        "distance",
        # Keep raw JSON for debugging/backward compatibility (will repeat per row)
        "hibuddy_store_prices_json",
    ]

    out_exists = out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=bool(args.headless),
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if len(context.pages) > 0 else context.new_page()

        _safe_goto(page, HIBUDDY_PRODUCTS_ALL_URL)
        _maybe_click_age_gate(page)

        if args.interactive_setup:
            print("\n=== Interactive setup ===")
            print("1) Set Location and radius.")
            print("2) Open a product page and set retailer table to 'Rows per page: VIEW ALL' (recommended).")
            print("3) Return to /products/all.")
            input("\nPress Enter to start scraping... ")
            _safe_goto(page, HIBUDDY_PRODUCTS_ALL_URL)
            _maybe_click_age_gate(page)

        if args.block_images:
            _enable_resource_blocking(context)

        f = out.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=header)
        if not out_exists:
            writer.writeheader()
            f.flush()

        try:
            total = len(df)
            for idx, row in df.iterrows():
                sku = str(row.get(sku_col, "")).strip()
                query = str(row.get(query_col, "")).strip()
                size_primary = str(row.get(size_col, "")).strip()

                if not sku:
                    continue
                if sku in done:
                    continue

                if not query:
                    writer.writerow(
                        {
                            "SKU": sku,
                            "hibuddy_search_query": "",
                            "hibuddy_size_primary": size_primary,
                            "hibuddy_product_url": "",
                            "hibuddy_size_selected": "",
                            "match_score": "",
                            "status": "missing_search_query",
                            "hibuddy_retailers_within_radius": "",
                            "hibuddy_best_store_name": "",
                            "hibuddy_best_store_price": "",
                            "hibuddy_best_store_address_full": "",
                            "hibuddy_best_store_address_street": "",
                            "store_row_index": "",
                            "store_name": "",
                            "address_full": "",
                            "address_street": "",
                            "price": "",
                            "distance": "",
                            "hibuddy_store_prices_json": "",
                        }
                    )
                    f.flush()
                    continue

                exp = ExpectedSpec(search_query=query, size_primary=size_primary)

                ok = _safe_search_to_grid(page, query, attempts=3)
                if not ok:
                    if debug_dir:
                        _write_debug(page, debug_dir, sku, "search_failed")
                    writer.writerow(
                        {
                            "SKU": sku,
                            "hibuddy_search_query": query,
                            "hibuddy_size_primary": size_primary,
                            "hibuddy_product_url": page.url,
                            "hibuddy_size_selected": "",
                            "match_score": "",
                            "status": "no_grid_results_or_no_product_links",
                            "hibuddy_retailers_within_radius": "",
                            "hibuddy_best_store_name": "",
                            "hibuddy_best_store_price": "",
                            "hibuddy_best_store_address_full": "",
                            "hibuddy_best_store_address_street": "",
                            "store_row_index": "",
                            "store_name": "",
                            "address_full": "",
                            "address_street": "",
                            "price": "",
                            "distance": "",
                            "hibuddy_store_prices_json": "",
                        }
                    )
                    f.flush()
                    continue

                candidates = _collect_candidates_from_grid(page, exp, limit=max(args.verify_top_k, 10))
                if not candidates:
                    if debug_dir:
                        _write_debug(page, debug_dir, sku, "no_candidates")
                    writer.writerow(
                        {
                            "SKU": sku,
                            "hibuddy_search_query": query,
                            "hibuddy_size_primary": size_primary,
                            "hibuddy_product_url": page.url,
                            "hibuddy_size_selected": "",
                            "match_score": "",
                            "status": "no_product_candidates_found",
                            "hibuddy_retailers_within_radius": "",
                            "hibuddy_best_store_name": "",
                            "hibuddy_best_store_price": "",
                            "hibuddy_best_store_address_full": "",
                            "hibuddy_best_store_address_street": "",
                            "store_row_index": "",
                            "store_name": "",
                            "address_full": "",
                            "address_street": "",
                            "price": "",
                            "distance": "",
                            "hibuddy_store_prices_json": "",
                        }
                    )
                    f.flush()
                    continue

                # We ALWAYS scan the first top-K candidates (default 5) and pick the best-scoring one.
                # This prevents missing the correct product that is "right next" to the first result.
                                # We always evaluate the first top-K candidates (default 5).
                # Selection priority:
                #   1) Candidate whose available/active/selected size matches hibuddy_size_primary (if provided)
                #   2) Highest match score among ties
                best_info = None  # (score, final_url, size_selected, size_active, size_options, store_count, store_rows, size_match_any)
                best_rank = None  # (size_match_flag, score)

                for j, cand in enumerate(candidates[: args.verify_top_k]):
                    try:
                        score, _size_ok, final_url, size_selected, size_active, size_options, store_count, store_rows = _verify_candidate_on_product_page(page, exp, cand)
                    except Exception:
                        if debug_dir:
                            _write_debug(page, debug_dir, sku, f"verify_err_{j}")
                        _recover_from_gateway(page, base_url=HIBUDDY_PRODUCTS_ALL_URL)
                        continue

                    size_match_any = _size_matches_requested(size_primary, size_selected, size_active, size_options)

                    # If a size was requested and this candidate doesn't even expose it, treat it as worse.
                    rank = (1 if (size_primary and size_match_any) else 0, score)

                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_info = (score, final_url, size_selected, size_active, size_options, store_count, store_rows, size_match_any)

                if best_info is None:
                    writer.writerow(
                        {
                            "SKU": sku,
                            "hibuddy_search_query": query,
                            "hibuddy_size_primary": size_primary,
                            "hibuddy_product_url": page.url,
                            "hibuddy_size_selected": "",
                            "match_score": "",
                            "status": "verify_failed",
                            "hibuddy_retailers_within_radius": "",
                            "hibuddy_best_store_name": "",
                            "hibuddy_best_store_price": "",
                            "hibuddy_best_store_address_full": "",
                            "hibuddy_best_store_address_street": "",
                            "store_row_index": "",
                            "store_name": "",
                            "address_full": "",
                            "address_street": "",
                            "price": "",
                            "distance": "",
                            "hibuddy_store_prices_json": "",
                        }
                    )
                    f.flush()
                    continue

                score, final_url, size_selected, size_active, size_options, store_count, store_rows, size_match_any = best_info
                best_row, best_price = _pick_best_store_record(store_rows)
                best_store = (best_row or {}).get("store_name") if best_row else None
                best_addr_full = (best_row or {}).get("address_full") if best_row else None
                best_addr_street = (best_row or {}).get("address_street") if best_row else None

                status = "ok" if score >= args.match_threshold else "below_threshold"
                if size_primary:
                    status = status + ("_size_match" if size_match_any else "_size_mismatch")
                else:
                    status = status + "_no_size_requested"

                raw_json = 0
#json.dumps(store_rows, ensure_ascii=False) if store_rows else ""
                # Explode to one row per retailer. 
                if store_rows:
                    for i_store, r in enumerate(store_rows):
                        price_val = r.get("price")
                        writer.writerow(
                            {
                                "SKU": sku,
                                "hibuddy_search_query": query,
                                "hibuddy_size_primary": size_primary,
                                "hibuddy_product_url": final_url,
                                "hibuddy_size_selected": size_selected,
                                "hibuddy_size_active": size_active,
                                "hibuddy_size_options_json": json.dumps(size_options, ensure_ascii=False) if size_options else "",
                                "match_score": f"{score:.2f}",
                                "status": status,
                                "hibuddy_retailers_within_radius": store_count if store_count is not None else "",
                                "hibuddy_best_store_name": best_store or "",
                                "hibuddy_best_store_price": f"{best_price:.2f}" if isinstance(best_price, (int, float)) else "",
                                "hibuddy_best_store_address_full": best_addr_full or "",
                                "hibuddy_best_store_address_street": best_addr_street or "",
                                "store_row_index": i_store,
                                "store_name": (r.get("store_name") or ""),
                                "address_full": (r.get("address_full") or ""),
                                "address_street": (r.get("address_street") or ""),
                                "price": f"{float(price_val):.2f}" if isinstance(price_val, (int, float)) else "",
                                "distance": (r.get("distance") or ""),
                                "hibuddy_store_prices_json": raw_json,
                            }
                        )
                else:
                    # Still write a single row to keep SKU visible in output.
                    writer.writerow(
                        {
                            "SKU": sku,
                            "hibuddy_search_query": query,
                            "hibuddy_size_primary": size_primary,
                            "hibuddy_product_url": final_url,
                            "hibuddy_size_selected": size_selected,
                            "hibuddy_size_active": size_active,
                            "hibuddy_size_options_json": json.dumps(size_options, ensure_ascii=False) if size_options else "",
                            "match_score": f"{score:.2f}",
                            "status": status,
                            "hibuddy_retailers_within_radius": store_count if store_count is not None else "",
                            "hibuddy_best_store_name": best_store or "",
                            "hibuddy_best_store_price": f"{best_price:.2f}" if isinstance(best_price, (int, float)) else "",
                            "hibuddy_best_store_address_full": best_addr_full or "",
                            "hibuddy_best_store_address_street": best_addr_street or "",
                            "store_row_index": "",
                            "store_name": "",
                            "address_full": "",
                            "address_street": "",
                            "price": "",
                            "distance": "",
                            "hibuddy_store_prices_json": raw_json,
                        }
                    )

                f.flush()

                # Reset to /products/all for next iteration
                try:
                    _safe_goto(page, HIBUDDY_PRODUCTS_ALL_URL, attempts=2)
                    _maybe_click_age_gate(page)
                except Exception:
                    _recover_from_gateway(page, base_url=HIBUDDY_PRODUCTS_ALL_URL)

                if (idx + 1) % 50 == 0:
                    print(f"Progress: {idx+1}/{total} rows processed...")

        finally:
            try:
                f.close()
            except Exception:
                pass
            try:
                context.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        run(parse_args(sys.argv[1:]))
        print("Done.")
    except KeyboardInterrupt:
        print("\nStopped by user. You can rerun with --resume to continue.")
