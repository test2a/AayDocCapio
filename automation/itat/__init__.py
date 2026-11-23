"""ITAT portal automation (captcha reading + case-status fetch)."""

from automation.itat.captcha import solve, should_submit  # noqa: F401

__all__ = ["solve", "should_submit", "fetch_case_status"]


def fetch_case_status(*args, **kwargs):
    from automation.itat.case_status import fetch_case_status as _f
    return _f(*args, **kwargs)
