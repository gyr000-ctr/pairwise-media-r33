import math

import pytest

import chord

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 4096

CASES = [
    ("detect", lambda: chord.Chord(['maj', 'min']).detect(S) is not None),
    ("probs_sum", lambda: abs(sum(chord.Chord(['maj', 'min']).probs(S)) - 1.0) < 1e-12),
    ("bad_templates", lambda: _raises(lambda: chord.Chord([]))),
    ("bad_template_name", lambda: _raises(lambda: chord.Chord(['bogus']))),
    ("empty", lambda: _raises(lambda: chord.Chord(['maj']).detect([]))),
    ("type", lambda: isinstance(chord.Chord(['maj']).probs(S), list)),
    ("det", lambda: chord.Chord(['maj', 'min']).detect(S) == chord.Chord(['maj', 'min']).detect(S)),
    ("nan", lambda: _raises(lambda: chord.Chord(['maj']).detect([float('nan')] * 100))),
    ("no_mut", lambda: chord.Chord(['maj']).detect(list(S)) is not None),
    ("detect_type", lambda: isinstance(chord.Chord(['maj']).detect(S), tuple)),
    ("probs_len", lambda: len(chord.Chord(['maj', 'min']).probs(S)) == 2),
    ("probs_range", lambda: all(0.0 <= v <= 1.0 for v in chord.Chord(['maj', 'min']).probs(S))),
    ("silence", lambda: chord.Chord(['maj']).detect([0.0] * 1000) is None),
    ("float", lambda: isinstance(chord.Chord(['maj', 'min']).probs(S)[0], float)),
    ("name_in_set", lambda: chord.Chord(['maj', 'min']).detect(S)[0] in ('maj', 'min')),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
