import math

import pytest

import gainstage

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.9] * 100

CASES = [
    ("count", lambda: len(gainstage.GainStage(3, 6.0).gains(S)) == 3),
    ("conserved", lambda: abs(math.prod(gainstage.GainStage(3, 6.0).gains(S))) > 0.0),
    ("no_clip", lambda: all(abs(v) <= 1.0 for v in gainstage.GainStage(3, 6.0).apply(S))),
    ("silence", lambda: all(abs(g - 1.0) < 1e-9 for g in gainstage.GainStage(3, 6.0).gains([0.0] * 10))),
    ("bad_stages", lambda: _raises(lambda: gainstage.GainStage(0, 6.0))),
    ("bad_headroom", lambda: _raises(lambda: gainstage.GainStage(3, 0.0))),
    ("empty", lambda: _raises(lambda: gainstage.GainStage(3, 6.0).gains([]))),
    ("type", lambda: isinstance(gainstage.GainStage(3, 6.0).gains(S), list)),
    ("det", lambda: gainstage.GainStage(3, 6.0).gains(S) == gainstage.GainStage(3, 6.0).gains(S)),
    ("apply_type", lambda: isinstance(gainstage.GainStage(3, 6.0).apply(S), list)),
    ("nan", lambda: _raises(lambda: gainstage.GainStage(3, 6.0).gains([float('nan')]))),
    ("no_mut", lambda: gainstage.GainStage(3, 6.0).apply(list(S)) is not None),
    ("float", lambda: isinstance(gainstage.GainStage(3, 6.0).gains(S)[0], float)),
    ("headroom_effect", lambda: gainstage.GainStage(3, 12.0).gains(S) != gainstage.GainStage(3, 3.0).gains(S)),
    ("apply_range", lambda: all(abs(v) <= 1.0 for v in gainstage.GainStage(2, 3.0).apply([0.5, -0.5]))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
