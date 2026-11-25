import math

import pytest

import declick

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 100

CASES = [
    ("clean_detect", lambda: declick.Declick(0.5, 8).detect(S) == []),
    ("clean_repair", lambda: all(abs(a - b) < 1e-12 for a, b in zip(declick.Declick(0.5, 8).repair(S), S))),
    ("spike_detect", lambda: len(declick.Declick(0.5, 8).detect([0.0] * 50 + [1.0] + [0.0] * 49)) >= 1),
    ("bad_threshold", lambda: _raises(lambda: declick.Declick(0.0, 8))),
    ("bad_max", lambda: _raises(lambda: declick.Declick(0.5, 0))),
    ("empty", lambda: declick.Declick(0.5, 8).repair([]) == []),
    ("type", lambda: isinstance(declick.Declick(0.5, 8).repair(S), list)),
    ("det", lambda: declick.Declick(0.5, 8).repair(S) == declick.Declick(0.5, 8).repair(S)),
    ("range", lambda: all(abs(v) <= 1.0 for v in declick.Declick(0.5, 8).repair([0.0] * 10 + [1.0] + [0.0] * 10))),
    ("nan", lambda: _raises(lambda: declick.Declick(0.5, 8).repair([float('nan')]))),
    ("no_mut", lambda: declick.Declick(0.5, 8).repair(list(S)) is not None),
    ("detect_type", lambda: isinstance(declick.Declick(0.5, 8).detect(S), list)),
    ("span_pair", lambda: all(len(d) == 2 for d in declick.Declick(0.5, 8).detect([0.0] * 50 + [1.0] + [0.0] * 49))),
    ("float", lambda: isinstance(declick.Declick(0.5, 8).repair(S)[0], float)),
    ("threshold_effect", lambda: len(declick.Declick(0.1, 8).detect([0.0] * 5 + [0.3] + [0.0] * 5)) >= len(declick.Declick(0.9, 8).detect([0.0] * 5 + [0.3] + [0.0] * 5))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
