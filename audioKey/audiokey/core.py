"""audiokey 核心 API：基于 Krumhansl 模板的调性检测。

实现要点
========

* **色度谱（chroma）**：对输入做 Hann 加窗后取实数 FFT，按
  ``A4 = 440 Hz`` 的十二平均律把每个频点折叠到 12 个音级：

      ``p = (12 * log2(f / 440) + 9) mod 12``

  其中 ``9`` 是音级 ``A`` 在 ``C=0, C#=1, ... B=11`` 编号下的下标。
  每个频点的幅度按四舍五入累加到对应音级，最后归一化为总和 1 的
  分布（静音时返回 12 个 0.0）。

* **模板**：Krumhansl-Kessler（K-K）大调 / 小调音级轮廓。以 C 为
  基准的轮廓通过循环移位生成 12 个大调与 12 个小调，共 24 个模板，
  顺序为先 12 个大调（C..B）后 12 个小调（C..B）。

* **相关**：色度谱与模板之间的 Pearson 相关系数。检测结果就是
  24 个相关系数中最大者对应的调性；``score`` 与 ``detect`` 使用同一
  套（已中心化、单位化）模板，因此与逐元素暴力计算在浮点误差范围内
  一致。

* **错误分类**：所有面向用户的参数错误都抛出 ``ValueError`` 的子类，
  并且互可区分（非法采样率、非法输入、非有限输入、非法模板）。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "AudioKeyError",
    "InvalidRateError",
    "InvalidInputError",
    "NonFiniteInputError",
    "InvalidTemplateError",
    "KeyDetect",
    "REFERENCE_FREQUENCY",
    "TONIC_NAMES",
    "MAJOR_PROFILE",
    "MINOR_PROFILE",
    "TEMPLATES",
    "KEY_TEMPLATES",
    "TEMPLATE_NAMES",
    "KEY_NAMES",
    "frequency_to_pitch_class",
    "correlate",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

REFERENCE_FREQUENCY = 440.0

#: 音级名称，下标即色度谱下标（C = 0 ... B = 11）。
TONIC_NAMES = (
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
)

#: Krumhansl-Kessler 大调音级轮廓（以 C 为主音）。
MAJOR_PROFILE = (
    6.35, 2.73, 2.98, 2.62, 2.38, 2.51,
    2.23, 2.00, 2.20, 2.04, 1.89, 2.06,
)

#: Krumhansl-Kessler 小调音级轮廓（以 C 为主音）。
MINOR_PROFILE = (
    6.33, 2.68, 2.60, 2.53, 2.49, 2.39,
    2.21, 2.38, 2.23, 2.12, 2.44, 2.29,
)

#: 音级名（小写）到色度谱下标的映射，用于解析调名。
_TONIC_LOOKUP = {name.lower(): index for index, name in enumerate(TONIC_NAMES)}


def _rotate(profile, tonic_index):
    """把以 C 为主音的轮廓移位到 ``tonic_index`` 对应的主音。

    色度谱下标 ``j`` 相对主音的音级为 ``(j - tonic_index) mod 12``。
    """
    return tuple(profile[(j - tonic_index) % 12] for j in range(12))


def _build_templates():
    """生成 24 个模板及其名称。"""
    templates = []
    names = []
    for tonic_index, tonic in enumerate(TONIC_NAMES):
        templates.append(_rotate(MAJOR_PROFILE, tonic_index))
        names.append((tonic, "major"))
    for tonic_index, tonic in enumerate(TONIC_NAMES):
        templates.append(_rotate(MINOR_PROFILE, tonic_index))
        names.append((tonic, "minor"))
    return tuple(templates), tuple(names)


#: 24 个模板（顺序：12 大调在前，12 小调在后），每个模板 12 个权重。
TEMPLATES, TEMPLATE_NAMES = _build_templates()

#: ``TEMPLATES`` 的别名。
KEY_TEMPLATES = TEMPLATES

#: 调性名称的字符串形式，例如 ``"C major"``。
KEY_NAMES = tuple("%s %s" % name for name in TEMPLATE_NAMES)

# 预计算：中心化 + 单位化的模板矩阵（24 x 12）。只读，线程安全。
_TEMPLATE_MATRIX = np.asarray(TEMPLATES, dtype=np.float64)
_TEMPLATE_CENTERED = _TEMPLATE_MATRIX - _TEMPLATE_MATRIX.mean(axis=1, keepdims=True)
_TEMPLATE_NORMS = np.linalg.norm(_TEMPLATE_CENTERED, axis=1)
_UNIT_TEMPLATES = _TEMPLATE_CENTERED / _TEMPLATE_NORMS[:, np.newaxis]

_TEMPLATE_COUNT = len(TEMPLATES)


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------

class AudioKeyError(ValueError):
    """audiokey 参数错误的基类（继承 ``ValueError``，便于通用捕获）。"""


class InvalidRateError(AudioKeyError):
    """采样率非法（非正、非有限或类型错误）。"""


class InvalidInputError(AudioKeyError):
    """采样序列非法（类型错误、非一维或为空）。"""


class NonFiniteInputError(InvalidInputError):
    """采样序列包含 ``nan`` / ``inf`` 等非有限值。"""


class InvalidTemplateError(AudioKeyError):
    """模板非法（下标越界、长度不为 12、含非有限值或零方差）。"""


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def frequency_to_pitch_class(frequency, reference=REFERENCE_FREQUENCY):
    """把频率按 ``A4 = 440 Hz`` 的十二平均律折叠为 0..11 的音级下标。

    ``C = 0, C# = 1, ..., A = 9, B = 11``。
    """
    try:
        frequency = float(frequency)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError("frequency must be a number") from exc
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise InvalidInputError("frequency must be positive and finite")
    pitch = 12.0 * math.log2(frequency / float(reference)) + 9.0
    return int(round(pitch)) % 12


def correlate(chroma, template):
    """计算两个 12 维向量之间的 Pearson 相关系数（暴力/参考实现）。

    该函数与 :meth:`KeyDetect.score` 使用同一数学定义，可用来交叉校验。
    """
    left = np.asarray(chroma, dtype=np.float64)
    right = np.asarray(template, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        raise InvalidTemplateError("chroma and template must be 1-D vectors of equal length")
    if left.size == 0:
        raise InvalidTemplateError("vectors must not be empty")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise InvalidTemplateError("vectors must contain only finite values")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    norms = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if norms == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / norms)


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class KeyDetect:
    """Krumhansl 模板调性检测器。

    参数
    ----
    rate:
        采样率（Hz），必须是正有限实数。

    属性
    ----
    rate: float
        采样率。
    templates: tuple
        24 个模板（大调 12 个在前、小调 12 个在后）。
    template_names: tuple
        与模板一一对应的 ``(主音, 调式)`` 名称。
    """

    REFERENCE_FREQUENCY = REFERENCE_FREQUENCY
    TEMPLATES = TEMPLATES
    TEMPLATE_NAMES = TEMPLATE_NAMES
    KEY_NAMES = KEY_NAMES

    def __init__(self, rate):
        if isinstance(rate, bool) or not isinstance(
            rate, (int, float, np.integer, np.floating)
        ):
            raise InvalidRateError(
                "rate must be a positive finite number, got %r" % (rate,)
            )
        rate = float(rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise InvalidRateError(
                "rate must be a positive finite number, got %r" % (rate,)
            )
        self.rate = rate
        # 只读视图：实例不持有可变状态，支持多线程并发调用。
        self.templates = TEMPLATES
        self.template_names = TEMPLATE_NAMES
        self.keys = TEMPLATE_NAMES

    # -- 输入处理 ---------------------------------------------------------

    @staticmethod
    def _as_samples(samples):
        if samples is None:
            raise InvalidInputError(
                "samples must be a one-dimensional sequence of numbers"
            )
        try:
            array = np.asarray(samples, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError(
                "samples must be a one-dimensional numeric sequence: %s" % (exc,)
            ) from exc
        if array.ndim != 1:
            raise InvalidInputError("samples must be one-dimensional")
        if array.size == 0:
            raise InvalidInputError("samples must not be empty")
        if not np.all(np.isfinite(array)):
            raise NonFiniteInputError(
                "samples must contain only finite values (nan/inf rejected)"
            )
        return array

    @staticmethod
    def _is_silent(array):
        return float(np.max(np.abs(array))) == 0.0

    # -- 色度谱 -----------------------------------------------------------

    def _chroma_array(self, array):
        n = int(array.size)
        if n >= 3:
            window = np.hanning(n)
        else:  # 极短输入：Hann 窗会退化，改用矩形窗
            window = np.ones(n)

        spectrum = np.fft.rfft(array * window)
        magnitude = np.abs(spectrum)
        if magnitude.size <= 1:  # 只有直流分量，没有可折叠的频点
            return np.zeros(12, dtype=np.float64)

        frequencies = np.fft.rfftfreq(n, d=1.0 / self.rate)[1:]
        magnitudes = magnitude[1:]

        pitch = 12.0 * np.log2(frequencies / REFERENCE_FREQUENCY) + 9.0
        classes = np.mod(np.floor(np.mod(pitch, 12.0) + 0.5), 12.0)
        classes = classes.astype(np.int64)

        chroma = np.bincount(
            classes, weights=magnitudes, minlength=12
        ).astype(np.float64)

        total = float(chroma.sum())
        if total > 0.0:
            chroma = chroma / total
        return chroma

    def chroma(self, samples):
        """返回 12 维色度谱（总和为 1 的 ``list``，静音时为全 0）。"""
        array = self._as_samples(samples)
        return self._chroma_array(array).tolist()

    # -- 相关 / 检测 -----------------------------------------------------

    @staticmethod
    def _unit_chroma(chroma):
        centered = chroma - chroma.mean()
        norm = float(np.linalg.norm(centered))
        if not math.isfinite(norm) or norm == 0.0:
            return None
        return centered / norm

    def _resolve_template(self, template):
        if isinstance(template, bool):
            raise TypeError("template index must be an integer, not bool")

        if isinstance(template, (int, np.integer)):
            index = int(template)
            if index < 0 or index >= _TEMPLATE_COUNT:
                raise InvalidTemplateError(
                    "template index must be in [0, %d), got %d"
                    % (_TEMPLATE_COUNT, index)
                )
            return _UNIT_TEMPLATES[index]

        if isinstance(template, (float, np.floating)):
            value = float(template)
            if not math.isfinite(value) or not value.is_integer():
                raise InvalidTemplateError(
                    "template index must be an integer in [0, %d), got %r"
                    % (_TEMPLATE_COUNT, template)
                )
            index = int(value)
            if 0 <= index < _TEMPLATE_COUNT:
                return _UNIT_TEMPLATES[index]
            raise InvalidTemplateError(
                "template index must be in [0, %d), got %r"
                % (_TEMPLATE_COUNT, template)
            )

        if isinstance(template, str):
            return self._template_from_name(template)

        if isinstance(template, np.ndarray):
            if template.ndim != 1:
                raise InvalidTemplateError(
                    "template array must be one-dimensional, got shape %r"
                    % (template.shape,)
                )
            return self._template_from_sequence(template.tolist())

        if isinstance(template, (tuple, list)):
            if len(template) == 2 and all(isinstance(part, str) for part in template):
                return self._template_from_name(" ".join(template))
            return self._template_from_sequence(template)

        raise TypeError(
            "template must be an index, a key name, or a 12-element numeric sequence"
        )

    @staticmethod
    def _template_from_name(name):
        text = str(name).strip().lower()
        text = text.replace(":", "").replace("_", "").replace(" ", "")
        if not text:
            raise InvalidTemplateError("template name must not be empty")

        suffixes = (
            ("major", "major"),
            ("maj", "major"),
            ("minor", "minor"),
            ("min", "minor"),
            ("m", "minor"),
        )
        for suffix, mode in suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                tonic_text = text[: -len(suffix)]
                tonic_index = _TONIC_LOOKUP.get(tonic_text)
                if tonic_index is not None:
                    return _UNIT_TEMPLATES[
                        tonic_index if mode == "major" else tonic_index + 12
                    ]

        # 只有主音名时默认按大调处理，例如 ``"C"``。
        tonic_index = _TONIC_LOOKUP.get(text)
        if tonic_index is not None:
            return _UNIT_TEMPLATES[tonic_index]

        raise InvalidTemplateError("unknown key/template name: %r" % (name,))

    @staticmethod
    def _template_from_sequence(sequence):
        try:
            array = np.asarray(sequence, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InvalidTemplateError(
                "template must be a 12-element numeric sequence: %s" % (exc,)
            ) from exc
        if array.ndim != 1 or array.size != 12:
            raise InvalidTemplateError(
                "template must contain exactly 12 weights, got shape %r"
                % (array.shape,)
            )
        if not np.all(np.isfinite(array)):
            raise InvalidTemplateError(
                "template must contain only finite values (nan/inf rejected)"
            )
        centered = array - array.mean()
        norm = float(np.linalg.norm(centered))
        if norm == 0.0:
            raise InvalidTemplateError(
                "template must not be constant (zero variance)"
            )
        return centered / norm

    def score(self, samples, template):
        """色度谱与指定模板的 Pearson 相关系数（``float``）。

        ``template`` 可以是：

        * 0..23 的模板下标（大调在前、小调在后）；
        * 调名，如 ``"C major"`` / ``"Am"``；
        * ``(主音, 调式)`` 形式的二元组，如 ``("C", "major")``；
        * 长度 12 的数值模板。

        非法模板抛出 :class:`InvalidTemplateError`；类型不受支持时抛出
        ``TypeError``，二者可区分。
        """
        array = self._as_samples(samples)
        template_vector = self._resolve_template(template)
        chroma = self._chroma_array(array)
        unit = self._unit_chroma(chroma)
        if unit is None:
            return 0.0
        return float(np.dot(template_vector, unit))

    def detect(self, samples):
        """检测调性，返回 ``(主音, 调式)``；静音输入返回 ``None``。

        调式取 ``"major"`` 或 ``"minor"``。结果等于 24 个模板
        :meth:`score` 的最大者（下标最小者优先，保证确定性）。
        """
        array = self._as_samples(samples)
        if self._is_silent(array):
            return None

        chroma = self._chroma_array(array)
        unit = self._unit_chroma(chroma)
        if unit is None:
            # 退化情况（例如恒定信号经加窗后能量集中在最低频点之外）
            # 仍返回确定的调性，避免调用方拿到 None 误判为静音。
            return TEMPLATE_NAMES[0]

        scores = _UNIT_TEMPLATES @ unit
        index = int(np.argmax(scores))
        return TEMPLATE_NAMES[index]
