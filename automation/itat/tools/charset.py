"""Single source of truth for the ITAT captcha alphabet and model input size.

Everything that renders, trains, or reads an ITAT captcha imports from here:

* ``model.py``    — number of classification heads' classes
* ``synth.py``    — which characters to draw
* ``preprocess.py``/``../captcha.py`` — the resize target (W, H)
* ``train.py``    — label <-> index mapping

To change the alphabet or input size, edit only this file and retrain. The
values below are the initial guess (6-character uppercase alphanumeric); they
are refined from live samples via ``tools/harvest.py`` + ``tools/inspect_sample.py``
— see ``PlansofThisProject/F-27_itat_captcha_solver.md``.
"""
from __future__ import annotations

CAPTCHA_LEN = 6

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

W, H = 182, 50

CHAR_TO_INDEX = {c: i for i, c in enumerate(CHARSET)}
N_CLASSES = len(CHARSET)


def encode(label: str) -> list[int]:
    """Map a label string to character indices."""
    if len(label) != CAPTCHA_LEN:
        raise ValueError(f"label must be {CAPTCHA_LEN} chars, got {label!r}")
    return [CHAR_TO_INDEX[c.upper()] for c in label]


def decode(indices: list[int]) -> str:
    """Map character indices back to a label string."""
    return "".join(CHARSET[i] for i in indices)


def normalise(text: str) -> str:
    """Uppercase and keep only in-alphabet characters, trimmed to CAPTCHA_LEN."""
    out = []
    for ch in (text or "").upper():
        if ch in CHAR_TO_INDEX:
            out.append(ch)
        if len(out) == CAPTCHA_LEN:
            break
    return "".join(out)


def is_valid_code(text: str) -> bool:
    """True if *text* is exactly CAPTCHA_LEN in-alphabet characters."""
    return bool(text) and len(text) == CAPTCHA_LEN and all(c in CHAR_TO_INDEX for c in text)
