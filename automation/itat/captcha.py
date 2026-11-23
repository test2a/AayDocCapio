"""Read the 6-character captcha on itat.gov.in/judicial/casestatus."""
from __future__ import annotations

import io
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.join(_HERE, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

try:
    from charset import (CAPTCHA_LEN, CHARSET, W, H,  # noqa: F401
                         decode, is_valid_code, normalise)
except Exception:  # pragma: no cover
    CAPTCHA_LEN, CHARSET, W, H = 6, "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", 182, 50

    def decode(indices):
        return "".join(CHARSET[i] for i in indices)

    def normalise(text):
        return "".join(c for c in (text or "").upper() if c in CHARSET)[:CAPTCHA_LEN]

    def is_valid_code(text):
        return bool(text) and len(text) == CAPTCHA_LEN and all(c in CHARSET for c in text)

_MODEL_PATH = os.path.join(_HERE, "model.onnx")

_VOTE_VARIANTS = (
    (True, 30, 4), (True, 45, 4), (True, 55, 3),
    (False, 30, 4), (False, 45, 3), (False, 60, 3),
)

_ocr = None
_session = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        import ddddocr
        _ocr = ddddocr.DdddOcr(show_ad=False, beta=True)
    return _ocr


def _get_session():
    global _session
    if _session is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(_MODEL_PATH)
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(
            _MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


def _open(png):
    if isinstance(png, (bytes, bytearray)):
        return Image.open(io.BytesIO(png))
    return png


def _clean(text) -> str:
    code = normalise(text)
    return code if len(code) == CAPTCHA_LEN else ""


def render4(png) -> np.ndarray:
    im = _open(png).convert("RGB").resize((W, H), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    gray = a.mean(axis=2)
    inv = 1.0 - gray / 255.0
    dark_abs = (gray < 45).astype(np.float32)
    dark_rel = (gray < gray.mean() * 0.72).astype(np.float32)
    red = ((a[..., 0] > 150) & (a[..., 1] < 100) & (a[..., 2] < 100)).astype(np.float32)
    return np.stack([inv, dark_abs, dark_rel, red], axis=0)


def render_thr(png, thr: int = 45, scale: int = 4, crop: bool = True) -> bytes:
    im = _open(png).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    if crop and a.shape[0] > 46 and a.shape[1] > 120:
        a = a[1:-1, 1:-1]
    gray = a.mean(axis=2)
    red = (a[..., 0] > 150) & (a[..., 1] < 100) & (a[..., 2] < 100)
    mask = (gray < thr) & ~red
    out = np.where(mask, 0, 255).astype(np.uint8)
    img = Image.fromarray(out).resize(
        (out.shape[1] * scale, out.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def cnn_read(png):
    try:
        sess = _get_session()
        x = render4(png)[None, ...]
        outs = sess.run(None, {sess.get_inputs()[0].name: x})
        probs = []
        for logits in outs:
            row = np.asarray(logits[0], dtype=np.float64)
            e = np.exp(row - row.max())
            probs.append(e / e.sum())
        chars = [CHARSET[int(p.argmax())] for p in probs]
        return "".join(chars), float(min(p.max() for p in probs))
    except Exception:
        return "", 0.0


def _ocr_read(data: bytes) -> str:
    try:
        return _clean(_get_ocr().classification(data))
    except Exception:
        return ""


def vote(png):
    votes = [_ocr_read(render_thr(png, thr, scale, crop))
             for crop, thr, scale in _VOTE_VARIANTS]
    got = [v for v in votes if v]
    if not got:
        return "", 0.0, votes
    code, n = Counter(got).most_common(1)[0]
    return code, n / len(_VOTE_VARIANTS), votes


def solve(png) -> dict:
    cnn, cnn_min = cnn_read(png)
    t45 = _ocr_read(render_thr(png, 45))
    t60 = _ocr_read(render_thr(png, 60))
    vote_code, agree, votes = vote(png)
    n_votes = sum(1 for v in votes if v)

    if agree >= 0.6:
        code, conf, reason = vote_code, "high", "vote-strong"
    elif t45 and t45 == t60 and n_votes >= 4:
        code, conf, reason = t45, "high", "t45=t60+nv4"
    elif cnn and cnn_min >= 0.90 and cnn in (vote_code, t45, t60):
        code, conf, reason = cnn, "high", "cnn-agree"
    elif t45 and t45 == t60 and cnn == t45:
        code, conf, reason = t45, "high", "t45=t60+cnn"
    elif (agree >= 0.5 and vote_code and vote_code in (t45, t60)
          and cnn and cnn_min >= 0.70):
        code, conf, reason = vote_code, "medium", "vote-half+cnn"
    elif cnn and cnn_min >= 0.85 and vote_code and vote_code == cnn:
        code, conf, reason = cnn, "medium", "cnn-vote"
    elif t45 and t60 and t45 != t60:
        code, conf, reason = "", "conflict", "t45-t60-conflict"
    elif cnn and cnn_min >= 0.70:
        code, conf, reason = "", "low", "cnn-weak"
    else:
        code, conf, reason = "", "none", "unreadable"

    return {
        "code": code, "confidence": conf, "reason": reason,
        "cnn": cnn, "cnn_min": round(cnn_min, 4),
        "t45": t45, "t60": t60, "vote": vote_code,
        "agreement": round(agree, 2), "n_votes": n_votes,
    }


def should_submit(result: dict) -> bool:
    return bool(result.get("code")) and result.get("confidence") in ("high", "medium")
