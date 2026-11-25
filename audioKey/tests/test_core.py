import math

import pytest

import audiokey

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.1] * 4096

CASES = [
    ("chroma_len", lambda: len(audiokey.KeyDetect(48000).chroma(S)) == 12),
    ("detect", lambda: audiokey.KeyDetect(48000).detect(S) is not None),
    ("silence", lambda: audiokey.KeyDetect(48000).detect([0.0] * 1000) is None),
    ("bad_rate", lambda: _raises(lambda: audiokey.KeyDetect(0))),
    ("empty", lambda: _raises(lambda: audiokey.KeyDetect(48000).chroma([]))),
    ("type", lambda: isinstance(audiokey.KeyDetect(48000).chroma(S), list)),
    ("det", lambda: audiokey.KeyDetect(48000).detect(S) == audiokey.KeyDetect(48000).detect(S)),
    ("nan", lambda: _raises(lambda: audiokey.KeyDetect(48000).chroma([float('nan')] * 100))),
    ("no_mut", lambda: audiokey.KeyDetect(48000).chroma(list(S)) is not None),
    ("detect_type", lambda: isinstance(audiokey.KeyDetect(48000).detect(S), tuple)),
    ("chroma_range", lambda: all(v >= 0.0 for v in audiokey.KeyDetect(48000).chroma(S))),
    ("float", lambda: isinstance(audiokey.KeyDetect(48000).chroma(S)[0], float)),
    ("score", lambda: isinstance(audiokey.KeyDetect(48000).score(S, 0), float)),
    ("bad_score", lambda: _raises(lambda: audiokey.KeyDetect(48000).score(S, 99))),
    ("key_pair", lambda: len(audiokey.KeyDetect(48000).detect(S)) == 2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
