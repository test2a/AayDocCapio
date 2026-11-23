"""Inspect a harvested ITAT captcha set to finalise the alphabet and check the solver.

Development tool only. Reads ``<dir>/labels.csv`` (``filename,label``), reports how
many images there are, how many carry a label, and the sorted set of characters
observed (with counts) so ``tools/charset.py`` can be tuned to what the portal
actually renders. It then re-runs :func:`automation.itat.captcha.solve` on each
labelled image and prints filename/label/solver-code/confidence/match, plus the
aggregate exact-match accuracy. Finally it writes ``<dir>/montage.png`` — a
captioned grid of up to 60 images — for quick visual review.

    python inspect_sample.py [dir]
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PIL import Image, ImageDraw  # noqa: E402

from automation.itat import captcha  # noqa: E402
from charset import CAPTCHA_LEN, CHARSET  # noqa: E402

MONTAGE_MAX = 60
MONTAGE_COLS = 10
CELL_W, CELL_H = 190, 78
CAPTION_H = 14


def _tidy(label: str) -> str:
    """Upper-case, drop out-of-alphabet chars, keep partial/full labels as-is."""
    return "".join(c for c in (label or "").strip().upper() if c in CHARSET)


def _read_labels(csv_path: str) -> list[tuple[str, str]]:
    base = os.path.dirname(os.path.abspath(csv_path))
    rows: list[tuple[str, str]] = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            name = (r.get("filename") or "").strip()
            if not name:
                continue
            path = name if os.path.isabs(name) else os.path.join(base, name)
            rows.append((path, _tidy(r.get("label") or "")))
    return rows


def _solve(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return captcha.solve(data)
    except Exception as exc:
        return {"code": "", "confidence": "error", "reason": str(exc)[:40]}


def _montage(rows: list[tuple[str, str]], out_path: str) -> None:
    chosen = rows[:MONTAGE_MAX]
    if not chosen:
        return
    n = len(chosen)
    cols = min(MONTAGE_COLS, n)
    rows_n = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL_W, rows_n * (CELL_H + CAPTION_H)),
                      "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(chosen):
        cx = (i % cols) * CELL_W
        cy = (i // cols) * (CELL_H + CAPTION_H)
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((CELL_W - 6, CELL_H - 6))
            sheet.paste(im, (cx + 3, cy + 3))
        except Exception:
            pass
        name = os.path.basename(path)
        draw.text((cx + 3, cy + CELL_H), f"{name} [{label or '-'}]", fill="black")
    sheet.save(out_path)


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else "harvest"
    csv_path = os.path.join(target, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"no labels.csv in {target!r} — run tools/harvest.py first")
        return 1

    rows = _read_labels(csv_path)
    labelled = [(p, lab) for p, lab in rows if lab]
    full = [(p, lab) for p, lab in labelled if len(lab) == CAPTCHA_LEN]
    print(f"dir: {os.path.abspath(target)}")
    print(f"total images: {len(rows)}")
    print(f"labelled: {len(labelled)} ({len(full)} full {CAPTCHA_LEN}-char codes)")

    char_counts = Counter()
    for _, lab in labelled:
        char_counts.update(lab)
    observed = "".join(sorted(char_counts))
    print(f"observed charset ({len(char_counts)}): {observed}")
    for ch in observed:
        print(f"  {ch}: {char_counts[ch]}")

    if not labelled:
        print("nothing to score (no labels)")
        _montage(rows, os.path.join(target, "montage.png"))
        print(f"montage -> {os.path.join(target, 'montage.png')}")
        return 0

    print("\nfilename                          label   solver  conf      match")
    hits = submitted = submitted_ok = 0
    for path, label in labelled:
        result = _solve(path)
        code = result.get("code") or ""
        conf = result.get("confidence") or "?"
        match = bool(code) and code == label
        hits += int(match)
        if captcha.should_submit(result):
            submitted += 1
            submitted_ok += int(match)
        name = os.path.basename(path)
        print(f"{name:<32}  {label:<6}  {code or '------':<6}  {conf:<8}  "
              f"{'OK' if match else ('--' if not code else 'X')}")

    total = len(labelled)
    print(f"\nexact-match accuracy: {hits}/{total} "
          f"({100.0 * hits / total:.1f}%)")
    if len(full) != total:
        print(f"note: {total - len(full)} label(s) are partial, not full "
              f"{CAPTCHA_LEN}-char codes (auto-proposed or hand-cut)")
    print(f"submitted (should_submit): {submitted}, "
          f"correct among submitted: {submitted_ok}")

    out_montage = os.path.join(target, "montage.png")
    _montage(rows, out_montage)
    print(f"montage -> {out_montage} ({min(len(rows), MONTAGE_MAX)} tiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
