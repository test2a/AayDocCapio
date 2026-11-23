"""Unit tests for the ITAT captcha solver and case-status helper (no network)."""
import io
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automation.itat import captcha
from automation.itat.tools import charset
from automation.itat.case_status import classify_outcome, _result


def _png(w=182, h=50):
    img = Image.new("RGB", (w, h), (200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_charset_roundtrip_and_validation():
    assert len(charset.CHARSET) == charset.N_CLASSES
    code = "A1B2C3"
    assert charset.decode(charset.encode(code)) == code
    assert charset.is_valid_code(code)
    assert not charset.is_valid_code("A1B2C")
    assert not charset.is_valid_code("A1B2C!")
    assert charset.normalise(" ab12-cd99 ") == "AB12CD"


def test_render4_shape_and_range():
    x = captcha.render4(_png())
    assert x.shape == (4, charset.H, charset.W)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_render_thr_is_png():
    data = captcha.render_thr(_png(), thr=45)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    im = Image.open(io.BytesIO(data))
    assert im.size == (180 * 4, 48 * 4)


def _result_ok(**kw):
    base = {"code": "", "confidence": "none", "cnn": "", "cnn_min": 0.0,
            "t45": "", "t60": "", "vote": "", "agreement": 0.0, "n_votes": 0}
    base.update(kw)
    return base


def test_should_submit_requires_code_and_confidence():
    assert not captcha.should_submit(_result_ok())
    assert not captcha.should_submit(_result_ok(code="A1B2C3", confidence="low"))
    assert not captcha.should_submit(_result_ok(confidence="high"))
    assert captcha.should_submit(_result_ok(code="A1B2C3", confidence="high"))
    assert captcha.should_submit(_result_ok(code="A1B2C3", confidence="medium"))


def test_solve_prefers_consensus_vote(monkeypatch):
    monkeypatch.setattr(captcha, "cnn_read", lambda png: ("111111", 0.99))
    monkeypatch.setattr(captcha, "_ocr_read", lambda data: "")
    monkeypatch.setattr(captcha, "vote",
                        lambda png: ("Z2B4D6", 1.0, ["Z2B4D6"] * 6))
    r = captcha.solve(_png())
    assert r["code"] == "Z2B4D6"
    assert r["reason"] == "vote-strong"
    assert captcha.should_submit(r)


def test_solve_accepts_alphanumeric_cnn(monkeypatch):
    monkeypatch.setattr(captcha, "cnn_read", lambda png: ("AB12CD", 0.95))
    monkeypatch.setattr(captcha, "_ocr_read", lambda data: "AB12CD")
    monkeypatch.setattr(captcha, "vote", lambda png: ("AB12CD", 0.5, ["AB12CD"] * 3))
    r = captcha.solve(_png())
    assert r["code"] == "AB12CD"
    assert captcha.should_submit(r)


def test_solve_refuses_on_conflict(monkeypatch):
    monkeypatch.setattr(captcha, "cnn_read", lambda png: ("AAA111", 0.2))
    values = iter(["AB12CD", "ZZ99YY"])
    monkeypatch.setattr(captcha, "_ocr_read", lambda data: next(values))
    monkeypatch.setattr(captcha, "vote", lambda png: ("", 0.0, [""] * 6))
    r = captcha.solve(_png())
    assert r["code"] == ""
    assert r["reason"] == "t45-t60-conflict"
    assert not captcha.should_submit(r)


def test_solve_unreadable(monkeypatch):
    monkeypatch.setattr(captcha, "cnn_read", lambda png: ("", 0.0))
    monkeypatch.setattr(captcha, "_ocr_read", lambda data: "")
    monkeypatch.setattr(captcha, "vote", lambda png: ("", 0.0, [""] * 6))
    r = captcha.solve(_png())
    assert r["code"] == ""
    assert r["reason"] == "unreadable"


def test_cnn_read_without_model_is_empty():
    assert captcha.cnn_read(_png()) == ("", 0.0)


@pytest.mark.parametrize("text,verdict", [
    ("invalid captcha please retry", "wrong"),
    ("enter valid captcha", "wrong"),
    ("no record found", "no_record"),
    ("appeal number 1234 status pending", "result"),
    ("something unexpected", "unknown"),
    ("", "unknown"),
])
def test_classify_outcome(text, verdict):
    assert classify_outcome(text) == verdict


def test_case_result_shape():
    r = _result(True, "result", "1234", 2, 1, "ok", "<table>", "text")
    assert r["ok"] is True and r["reason"] == "result"
    assert r["appeal_number"] == "1234"
    assert r["rows_html"] == "<table>"
    assert set(r) >= {"ok", "reason", "appeal_number", "rows_html", "page_text",
                      "captcha_reads", "attempts", "error_text"}


def test_import_does_not_pull_in_torch():
    code = ("import sys; import automation.itat.captcha; "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
