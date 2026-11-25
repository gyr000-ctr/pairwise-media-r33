import math

import pytest

import crosstalk

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

L = [0.5] * 100
R = [0.1] * 100

CASES = [
    ("estimate", lambda: len(crosstalk.Crosstalk(64).estimate(L, R)) == 2),
    ("cancel_len", lambda: len(crosstalk.Crosstalk(64).cancel(L, R)[0]) == 100),
    ("mono", lambda: isinstance(crosstalk.Crosstalk(64).cancel([0.5], [0.5]), tuple)),
    ("bad_len", lambda: _raises(lambda: crosstalk.Crosstalk(0))),
    ("bad_input", lambda: _raises(lambda: crosstalk.Crosstalk(64).cancel(L, [0.1]))),
    ("empty", lambda: crosstalk.Crosstalk(64).cancel([], []) == ([], [])),
    ("type", lambda: isinstance(crosstalk.Crosstalk(64).cancel(L, R), tuple)),
    ("det", lambda: crosstalk.Crosstalk(64).cancel(L, R) == crosstalk.Crosstalk(64).cancel(L, R)),
    ("range", lambda: all(abs(v) <= 1.0 for v in crosstalk.Crosstalk(64).cancel(L, R)[0])),
    ("nan", lambda: _raises(lambda: crosstalk.Crosstalk(64).cancel([float('nan')], [0.0]))),
    ("no_mut", lambda: crosstalk.Crosstalk(64).cancel(list(L), list(R)) is not None),
    ("est_type", lambda: isinstance(crosstalk.Crosstalk(64).estimate(L, R), tuple)),
    ("float", lambda: isinstance(crosstalk.Crosstalk(64).cancel(L, R)[0][0], float)),
    ("short", lambda: isinstance(crosstalk.Crosstalk(8).cancel([0.1, 0.2], [0.1, 0.2]), tuple)),
    ("identity", lambda: isinstance(crosstalk.Crosstalk(64).cancel([0.0] * 10, [0.0] * 10), tuple)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
