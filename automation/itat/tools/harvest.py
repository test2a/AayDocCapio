"""Harvest anonymous ITAT captcha images for training.

Development tool only. Pulls the captcha shown on
https://itat.gov.in/judicial/casestatus, saves each image under
``<out_dir>/raw``, and writes ``<out_dir>/labels.csv`` with an auto-proposed
label (blank unless the solver is confident). No client data is touched: the
captcha is served anonymously before any case is looked up.

    python harvest.py [count] [out_dir]

The portal rate-limits captcha rotation — after roughly twenty quick grabs it
repeats the same image and then serves a broken one. This tool throttles each
grab, detects duplicate frames by hash, and skips tiny or blank frames so the
saved set stays clean. A real harvest needs PyTorch-free deps only: PIL,
playwright and the captcha module.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PIL import Image  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402

from automation.itat import captcha  # noqa: E402
from charset import CAPTCHA_LEN, is_valid_code  # noqa: E402

PAGE_URL = "https://itat.gov.in/judicial/casestatus"
CAPTCHA_SEL = "#captchaImage"

REFRESH_JS = """() => {
    const img = document.querySelector('#captchaImage');
    if (!img) return '';
    img.src = 'https://itat.gov.in/captcha/show?rnd=' + Date.now();
    return img.src;
}"""

MIN_BYTES = 400


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _usable(data: bytes, prev_hash: str | None) -> tuple[bool, str]:
    """Reject tiny, duplicate, or glyph-less frames without touching the disk."""
    if not data or len(data) < MIN_BYTES:
        return False, "tiny"
    if prev_hash and _md5(data) == prev_hash:
        return False, "duplicate"
    try:
        im = Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        return False, "undecodable"
    w, h = im.size
    if w < 80 or h < 25:
        return False, "small"
    dark = sum(1 for v in im.getdata() if v < 90)
    if dark < 40:
        return False, "no-glyphs"
    return True, ""


def _propose_label(data: bytes) -> str:
    try:
        result = captcha.solve(data)
    except Exception:
        return ""
    if captcha.should_submit(result):
        code = result.get("code") or ""
        return code if is_valid_code(code) else ""
    return ""


async def _grab(page) -> bytes:
    el = await page.query_selector(CAPTCHA_SEL)
    if el is None:
        return b""
    try:
        return await el.screenshot()
    except Exception:
        return b""


async def harvest(count: int, out_dir: str) -> int:
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    labels_path = os.path.join(out_dir, "labels.csv")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            try:
                await page.goto(PAGE_URL, wait_until="networkidle", timeout=60000)
            except Exception as exc:
                print(f"ITAT portal unreachable — check network/VPN ({exc})",
                      file=sys.stderr)
                return 1
            await page.wait_for_selector(CAPTCHA_SEL, timeout=30000)
            await page.wait_for_timeout(2500)

            rows: list[tuple[str, str]] = []
            labelled = 0
            prev_hash: str | None = None
            attempts = 0
            max_attempts = count * 4 + 40

            while len(rows) < count and attempts < max_attempts:
                attempts += 1
                await asyncio.sleep(random.uniform(2.0, 3.0))

                shot = await _grab(page)
                ok, why = _usable(shot, prev_hash)
                if not ok:
                    await page.evaluate(REFRESH_JS)
                    await asyncio.sleep(random.uniform(2.8, 3.6))
                    shot = await _grab(page)
                    ok, why = _usable(shot, prev_hash)
                if not ok:
                    print(f"  skip frame ({why})")
                    await page.evaluate(REFRESH_JS)
                    continue

                prev_hash = _md5(shot)
                name = f"cap_{len(rows) + 1:06d}.png"
                with open(os.path.join(raw_dir, name), "wb") as fh:
                    fh.write(shot)

                label = await asyncio.to_thread(_propose_label, shot)
                if label:
                    labelled += 1
                rel = f"raw/{name}"
                rows.append((rel, label))
                print(f"[{len(rows):>4}/{count}] {rel} label={label or '(blank)'}")

                with open(labels_path, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["filename", "label"])
                    writer.writerows(rows)

                if len(rows) % 15 == 0:
                    await page.evaluate(REFRESH_JS)
        finally:
            await browser.close()

    print(f"captured {len(rows)} images, {labelled} auto-labelled -> {out_dir}")
    if not rows:
        print("no usable captcha frames captured", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    count = int(argv[1]) if len(argv) > 1 else 200
    out_dir = argv[2] if len(argv) > 2 else "harvest"
    return asyncio.run(harvest(count, out_dir))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
