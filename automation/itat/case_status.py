"""Fetch a case status from the ITAT portal (itat.gov.in/judicial/casestatus)."""
from __future__ import annotations

import asyncio
import re

from playwright.async_api import Page

from automation.itat import captcha

CASESTATUS_URL = "https://itat.gov.in/judicial/casestatus"
_ACCORDION = "button:has-text('Search by Appeal Number')"

_REFRESH_JS = """() => {
    const img = document.getElementById('captchaImage');
    if (!img) return '';
    img.src = 'https://itat.gov.in/captcha/show?rnd=' + Date.now();
    return img.src;
}"""

_RESULT_JS = """() => {
    const panel = document.getElementById('collapseOne');
    return panel ? panel.outerHTML : '';
}"""

_WRONG_MARKERS = ("invalid captcha", "captcha is", "valid captcha",
                  "enter valid", "captcha not", "incorrect captcha")
_NO_RECORD_MARKERS = ("no record", "no data", "not found", "no such")


def classify_outcome(page_text_lower: str) -> str:
    t = (page_text_lower or "").lower()
    if any(m in t for m in _WRONG_MARKERS):
        return "wrong"
    if any(m in t for m in _NO_RECORD_MARKERS):
        return "no_record"
    if "appeal number" in t and ("status" in t or "case" in t):
        return "result"
    if "<table" in t or "result" in t:
        return "result"
    return "unknown"


def _digits(value) -> str:
    return re.sub(r"\D", "", str(value or ""))


async def _visible(page: Page, selector: str) -> bool:
    el = await page.query_selector(selector)
    try:
        return bool(el) and await el.is_visible()
    except Exception:
        return False


async def _ensure_page(page: Page, timeout_ms: int, log):
    if "judicial/casestatus" not in page.url:
        log("[ITAT] Opening ITAT case-status page...")
        await page.goto(CASESTATUS_URL, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_timeout(2500)
    if not await _visible(page, "#app_number"):
        log("[ITAT] Expanding 'Search by Appeal Number'...")
        await page.click(_ACCORDION)
        await page.wait_for_selector("#app_number", state="visible", timeout=timeout_ms)


async def _select(page: Page, selector: str, value: str):
    value = (value or "").strip()
    if not value:
        return
    try:
        await page.select_option(selector, label=value)
        return
    except Exception:
        pass
    try:
        await page.select_option(selector, value=value)
    except Exception:
        pass


async def _fill_form(page: Page, appeal_number, bench, appeal_type, appeal_year):
    await page.fill("#app_number", _digits(appeal_number))
    await _select(page, "#bench_name_1", bench)
    await _select(page, "#app_type_1", appeal_type)
    await _select(page, "#app_year_1", appeal_year)
    try:
        await page.fill("#hp", "")
    except Exception:
        pass


async def solve_captcha_field(page: Page, log_callback=None,
                              max_refresh: int = 8, pace_ms: int = 1200) -> dict:
    log = log_callback or (lambda msg: None)
    last = None
    for refreshes in range(max_refresh + 1):
        el = await page.query_selector("#captchaImage")
        if el is None:
            break
        shot = await el.screenshot()
        result = await asyncio.to_thread(captcha.solve, shot)
        last = result
        log(f"[ITAT]   captcha read {refreshes}: {result['code'] or '(unsure)'}"
            f" {result['confidence']} ({result['reason']})")
        if captcha.should_submit(result):
            return result
        if refreshes >= max_refresh:
            break
        await page.evaluate(_REFRESH_JS)
        await page.wait_for_timeout(pace_ms)
    return last or {"code": "", "confidence": "none", "reason": "no-image"}


async def _capture(page: Page):
    rows_html = ""
    try:
        rows_html = await page.evaluate(_RESULT_JS) or ""
    except Exception:
        rows_html = ""
    try:
        page_text = (await page.inner_text("body"))[:4000]
    except Exception:
        page_text = ""
    return rows_html, page_text


async def fetch_case_status(page: Page, appeal_number, bench: str = "",
                            appeal_type: str = "", appeal_year: str = "",
                            log_callback=None, max_refresh: int = 8,
                            max_wrong: int = 4, pace_ms: int = 1200,
                            timeout_ms: int = 60000) -> dict:
    log = log_callback or (lambda msg: None)
    wrong_run = attempts = reads = 0
    error_text = rows_html = page_text = ""

    try:
        await _ensure_page(page, timeout_ms, log)
        while attempts < max_wrong + 2:
            attempts += 1
            await _fill_form(page, appeal_number, bench, appeal_type, appeal_year)
            chosen = await solve_captcha_field(page, log, max_refresh, pace_ms)
            reads += 1
            if not captcha.should_submit(chosen):
                log("[ITAT] captcha unreadable after refreshes")
                if attempts >= 2:
                    rows_html, page_text = await _capture(page)
                    return _result(False, "exhausted", appeal_number, reads,
                                   attempts, error_text, rows_html, page_text)
                await _ensure_page(page, timeout_ms, log)
                continue
            await page.fill("#captcha", chosen["code"].upper())
            await page.click("#b1")
            await page.wait_for_timeout(3000)
            try:
                page_text = (await page.inner_text("body"))[:4000]
            except Exception:
                page_text = ""
            error_text = page_text
            verdict = classify_outcome(page_text)
            log(f"[ITAT] submit {chosen['code']} -> {verdict}"
                f"{(' | ' + page_text[:120]) if page_text else ''}")
            if verdict == "wrong":
                wrong_run += 1
                if wrong_run >= max_wrong:
                    rows_html, page_text = await _capture(page)
                    return _result(False, "too-many-wrong", appeal_number, reads,
                                   attempts, error_text, rows_html, page_text)
                try:
                    await _fill_form(page, appeal_number, bench, appeal_type, appeal_year)
                    await page.fill("#captcha", "")
                except Exception:
                    pass
                if await _visible(page, "#captchaImage"):
                    await page.evaluate(_REFRESH_JS)
                    await page.wait_for_timeout(pace_ms)
                continue
            if verdict in ("result", "no_record"):
                rows_html, page_text = await _capture(page)
                return _result(True, verdict, appeal_number, reads, attempts,
                               error_text, rows_html, page_text)
            if attempts >= 2:
                rows_html, page_text = await _capture(page)
                return _result(False, "unknown", appeal_number, reads, attempts,
                               error_text, rows_html, page_text)
            await page.wait_for_timeout(2000)
        rows_html, page_text = await _capture(page)
        return _result(False, "exhausted", appeal_number, reads, attempts,
                       error_text, rows_html, page_text)
    except Exception as exc:
        log(f"[ITAT] fetch error: {exc}")
        return _result(False, "error", appeal_number, reads, attempts,
                       str(exc)[:200], rows_html, page_text)


def _result(ok, reason, appeal_number, reads, attempts, error_text,
            rows_html="", page_text=""):
    return {
        "ok": bool(ok), "reason": reason,
        "appeal_number": _digits(appeal_number),
        "rows_html": rows_html or "", "page_text": page_text or "",
        "captcha_reads": reads, "attempts": attempts,
        "error_text": (error_text or "")[:300],
    }
