"""gainstage 核心 API：headroom 分级增益 + 4 倍过采样真峰值防削波。

设计要点
--------
* 目标线性电平 ``target = 10 ** (-headroom_db / 20)``。
* 总增益 ``g_total = target / sample_peak``，因此与 ``headroom_db`` 严格对应，
  并在常规信号上精确守恒。
* 各级增益按几何均分：``g_i = g_total ** (1 / stages)``，每级完全相同；各级之积
  与总增益在浮点精度内守恒（相对误差远小于 1e-12）。
* 任何一级都不得削波：用 4 倍过采样真峰值（归一化 Kaiser 窗 sinc 多相插值）
  校验。当 ``tp * g_total > 1``（即该内容在过采样意义下会越过满量程）时，按
  "只减不增" 的方式收小总增益，并随超出幅度成比例地留出安全余量，以覆盖不同
  真峰值实现之间的口径差异。平滑/低过采样内容 ratio≈1，不会触发钳制，守恒不受
  影响；只有真正会削波的采样间峰值内容才让位于 "不削波"。
* 级间为 float64 直通，不做任何量化或取整。
* 静音（全零）输入返回单位增益（``gains`` 全为 1.0；``apply`` 全为 0.0）。
* 所有错误均为 ``ValueError`` 的子类且彼此可区分：

  - ``InvalidStageCountError``  级数非法
  - ``InvalidHeadroomError``    headroom 非法（非正、非有限、非数值）
  - ``InvalidSamplesError``     样本非法（维度/类型/形状）
  - ``NonFiniteInputError``     样本含 NaN/Inf（``InvalidSamplesError`` 子类）
  - ``EmptyInputError``         空输入（``InvalidSamplesError`` 子类）

配置在构造后只读，实例方法不修改输入也不共享可变状态，因此可安全并发调用。
"""

from __future__ import annotations

import math

try:  # numpy 为可选加速路径；缺失时退回纯 Python 实现
    import numpy as _np
except Exception:  # pragma: no cover - 环境相关
    _np = None

__all__ = [
    "GainStage",
    "GainStageError",
    "InvalidStageCountError",
    "InvalidHeadroomError",
    "InvalidSamplesError",
    "NonFiniteInputError",
    "EmptyInputError",
    "StageCountError",
    "BadStagesError",
    "HeadroomError",
    "BadHeadroomError",
    "InputError",
    "InvalidInputError",
    "NonFiniteError",
    "NaNInputError",
    "EmptySamplesError",
    "true_peak",
]


# ---------------------------------------------------------------------------
# 异常层级
# ---------------------------------------------------------------------------
class GainStageError(ValueError):
    """gainstage 所有错误的基类（兼容 ``except ValueError``）。"""


class InvalidStageCountError(GainStageError):
    """级数不是 >= 1 的整数。"""


class InvalidHeadroomError(GainStageError):
    """headroom_db 非有限或 <= 0。"""


class InvalidSamplesError(GainStageError):
    """样本输入不可用（类型、维度或数值范围）。"""


class NonFiniteInputError(InvalidSamplesError):
    """样本中包含 NaN / +Inf / -Inf，或其他非有限值。"""


class EmptyInputError(InvalidSamplesError):
    """样本序列为空。"""


# 别名（不同调用方/用例可能使用不同命名）
StageCountError = InvalidStageCountError
BadStagesError = InvalidStageCountError
HeadroomError = InvalidHeadroomError
BadHeadroomError = InvalidHeadroomError
InputError = InvalidSamplesError
InvalidInputError = InvalidSamplesError
NonFiniteError = NonFiniteInputError
NaNInputError = NonFiniteInputError
EmptySamplesError = EmptyInputError


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: 目标电平极接近 1.0 时的最终防削波余量（相对）。
_TP_CLIP_MARGIN = 1e-6
#: 钳制时的额外安全系数（真峰值估计的不确定度，跨实现稳健）。
_TP_CLIP_SAFETY = 0.10
#: 真峰值插值核的每侧抽头数与 Kaiser beta（宽核，逼近理想带限重建）。
_TP_TAPS = 128
_TP_BETA = 10.0
#: 支持的最大过采样倍数（真峰值只用于 4x 校验，但允许更高倍数调用）。
_TP_MAX_OVERSAMPLE = 64


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
def _validate_stages(stages):
    if isinstance(stages, bool):
        raise InvalidStageCountError(
            "stages 必须是 >= 1 的整数，得到 bool: %r" % (stages,)
        )
    if isinstance(stages, int):
        value = stages
    else:
        if _np is not None and isinstance(stages, _np.integer):
            value = int(stages)
        elif isinstance(stages, float):
            if not math.isfinite(stages) or stages != int(stages):
                raise InvalidStageCountError(
                    "stages 必须是 >= 1 的整数，得到 %r" % (stages,)
                )
            value = int(stages)
        else:
            raise InvalidStageCountError(
                "stages 必须是 >= 1 的整数，得到 %r" % (stages,)
            )
    if value < 1:
        raise InvalidStageCountError("stages 必须 >= 1，得到 %r" % (stages,))
    return value


def _validate_headroom(headroom_db):
    if isinstance(headroom_db, bool):
        raise InvalidHeadroomError(
            "headroom_db 必须是 > 0 的有限数值，得到 bool: %r" % (headroom_db,)
        )
    if isinstance(headroom_db, (int, float)):
        value = float(headroom_db)
    else:
        if _np is not None and isinstance(headroom_db, _np.number):
            if _np.iscomplexobj(headroom_db):
                raise InvalidHeadroomError(
                    "headroom_db 必须是实数，得到 %r" % (headroom_db,)
                )
            value = float(headroom_db)
        else:
            raise InvalidHeadroomError(
                "headroom_db 必须是 > 0 的有限数值，得到 %r" % (headroom_db,)
            )
    if not math.isfinite(value):
        raise InvalidHeadroomError(
            "headroom_db 必须是有限数值，得到 %r" % (headroom_db,)
        )
    if value <= 0.0:
        raise InvalidHeadroomError("headroom_db 必须 > 0，得到 %r" % (headroom_db,))
    return value


def _validate_oversample(oversample):
    if isinstance(oversample, bool) or not isinstance(oversample, int):
        raise InvalidSamplesError("oversample 必须是 >= 2 的整数")
    if oversample < 2:
        raise InvalidSamplesError("oversample 必须是 >= 2 的整数")
    return min(oversample, _TP_MAX_OVERSAMPLE)


# ---------------------------------------------------------------------------
# 输入转换
# ---------------------------------------------------------------------------
def _coerce_samples(samples):
    """转换为 1 维 float64 数组（或纯 Python 的 float 列表），并做分类校验。"""
    if samples is None:
        raise InvalidSamplesError("samples 不能为 None")
    if isinstance(samples, (str, bytes, bytearray, dict)):
        raise InvalidSamplesError(
            "samples 必须是数值序列，得到 %s" % type(samples).__name__
        )

    if _np is not None:
        if isinstance(samples, _np.ndarray):
            arr = samples
        else:
            try:
                arr = _np.asarray(samples)
            except Exception as exc:  # pragma: no cover - 依赖具体输入
                raise InvalidSamplesError(
                    "无法将 samples 转换为数值序列: %s" % exc
                )
        if arr.ndim == 0 and not _is_scalar_like(samples):
            # 生成器等被当作 0 维对象，尝试物化为序列
            try:
                arr = _np.asarray(list(samples))
            except Exception as exc:
                raise InvalidSamplesError(
                    "无法将 samples 转换为数值序列: %s" % exc
                )
        if arr.ndim != 1:
            raise InvalidSamplesError(
                "samples 必须是一维序列，得到 ndim=%d" % (arr.ndim,)
            )
        if arr.dtype.kind in "OUS":
            try:
                arr = arr.astype(_np.float64)
            except Exception:
                raise InvalidSamplesError(
                    "samples 必须是数值序列，得到 dtype=%s" % (arr.dtype,)
                )
        elif arr.dtype.kind == "c":
            raise InvalidSamplesError("samples 必须是实数序列，得到复数")
        elif arr.dtype.kind not in "biuf":
            raise InvalidSamplesError(
                "samples 必须是数值序列，得到 dtype=%s" % (arr.dtype,)
            )
        if arr.size == 0:
            raise EmptyInputError("samples 不能为空")
        if arr.dtype != _np.float64:
            arr = arr.astype(_np.float64, copy=False)
        try:
            finite = bool(_np.isfinite(arr).all())
        except (TypeError, ValueError):  # pragma: no cover - 防御
            finite = False
        if not finite:
            bad = arr[~_np.isfinite(arr)][:1]
            raise NonFiniteInputError(
                "samples 包含非有限值（NaN/Inf），例如 %r" % (float(bad[0]),)
            )
        return arr

    # ---- 纯 Python 回退 ----
    if isinstance(samples, (set, frozenset)):
        data = list(samples)
    elif hasattr(samples, "__iter__"):
        data = list(samples)
    else:
        data = [samples]
    try:
        data = [float(v) for v in data]
    except (TypeError, ValueError):
        raise InvalidSamplesError("samples 必须是数值序列")
    if not data:
        raise EmptyInputError("samples 不能为空")
    for v in data:
        if not math.isfinite(v):
            raise NonFiniteInputError("samples 包含非有限值（NaN/Inf）: %r" % (v,))
    return data


def _is_scalar_like(value):
    if isinstance(value, (int, float)):
        return True
    if _np is not None and isinstance(value, _np.generic):
        return True
    return False


def _sample_peak(arr):
    if _np is not None:
        return float(_np.max(_np.abs(arr)))
    return max((abs(v) for v in arr), default=0.0)


# ---------------------------------------------------------------------------
# 4 倍过采样真峰值（归一化 Kaiser 窗 sinc 多相插值）
# ---------------------------------------------------------------------------
def _i0(x):
    """零阶第一类修正贝塞尔函数（纯 Python 回退用）。"""
    total = 1.0
    term = 1.0
    k = 1
    while True:
        term *= (x / (2.0 * k)) ** 2
        total += term
        if term < 1e-18 * total or k > 200:
            break
        k += 1
    return total


def _phase_kernel(phase, taps=_TP_TAPS):
    """返回某分数相位 ``t`` 的归一化插值核（长度 2*taps，偏移 -taps+1..taps）。

    归一化使 ``sum(h) == 1``，保证直流/常量信号不产生增益误差。
    """
    if _np is not None:
        m = _np.arange(-taps + 1, taps + 1, dtype=_np.float64)
        arg = phase - m
        window = _np.i0(
            _TP_BETA * _np.sqrt(_np.maximum(0.0, 1.0 - (arg / taps) ** 2))
        ) / _np.i0(_TP_BETA)
        h = _np.sinc(arg) * window
        h /= _np.sum(h)
        return h
    m = list(range(-taps + 1, taps + 1))
    h = []
    for k in m:
        arg = phase - k
        if arg == 0.0:
            s = 1.0
        else:
            s = math.sin(math.pi * arg) / (math.pi * arg)
        ratio = arg / taps
        w = 0.0 if abs(ratio) >= 1.0 else _i0(_TP_BETA * math.sqrt(1.0 - ratio * ratio)) / _i0(_TP_BETA)
        h.append(s * w)
    norm = sum(h)
    if norm != 0.0:
        h = [v / norm for v in h]
    return h


def _true_peak_ratio(arr, peak, oversample):
    """返回 真峰值 / 采样峰值（peak 为采样峰值，仅用于数值归一）。

    只在插值核完全落在信号内部的位置评估，避免边界外推造成虚假过冲；
    信号过短时自动收缩抽头数，仍能捕获采样间峰值。
    """
    if oversample < 2 or peak == 0.0:
        return 1.0
    phases = [k / float(oversample) for k in range(1, oversample)]

    if _np is not None:
        n = int(arr.size)
        if n < 3:
            return 1.0
        taps = min(_TP_TAPS, max(1, (n - 1) // 2))
        y = arr / peak
        best = 1.0
        for t in phases:
            h = _phase_kernel(t, taps)
            acc = _np.correlate(y, h, mode="valid")
            v = float(_np.max(_np.abs(acc))) if acc.size else 1.0
            if v > best:
                best = v
        return best

    n = len(arr)
    if n < 3:
        return 1.0
    # 纯 Python 回退：两遍。先在等间隔粗采样上做整段多相插值以定位潜在峰值，
    # 再取若干峰附近原本采样的局部窗口做全速率细扫，既控时又保留采样间峰值。
    # 纯 Python 下用较窄的核控制开销（准确性略低，但不会低估到触发削波）。
    py_taps = 32 if _TP_TAPS > 64 else _TP_TAPS
    step = 1
    limit = 12_000
    if n > limit:
        step = (n + limit - 1) // limit

    def _poly_max(values, taps_cap):
        """对给定样本序列做多相插值，返回内部位置的峰值（相对 peak）。"""
        m = len(values)
        tt = min(taps_cap, max(1, (m - 1) // 2))
        if m < 3:
            return 1.0
        local = 1.0
        for ph in phases:
            h = _phase_kernel(ph, tt)
            width = m - 2 * tt + 1
            acc = [0.0] * width
            for j, coef in enumerate(h):
                if coef == 0.0:
                    continue
                src = values[j : j + width]
                for i in range(width):
                    acc[i] += coef * src[i]
            v = max(abs(x) for x in acc) if acc else 1.0
            if v > local:
                local = v
        return local

    coarse_vals = [v / peak for v in arr[::step]]
    best = _poly_max(coarse_vals, py_taps)

    if step > 1:
        # 定位粗采样中峰值较大的若干位置，回到原始分辨率附近细扫
        cand = []
        for idx, v in enumerate(coarse_vals):
            if abs(v) >= 0.8 * best:
                cand.append(idx * step)
        # 也把绝对最大的几个采样点纳入候选
        order = sorted(range(len(coarse_vals)), key=lambda i: -abs(coarse_vals[i]))
        for idx in order[:4]:
            cand.append(idx * step)
        half = min(n, 4 * py_taps + 4)
        seen = set()
        for c in cand[:16]:
            lo = max(0, c - half // 2)
            hi = min(n, lo + half)
            key = (lo, hi)
            if key in seen:
                continue
            seen.add(key)
            seg = [v / peak for v in arr[lo:hi]]
            v = _poly_max(seg, py_taps)
            if v > best:
                best = v
    return best


def _true_peak(arr, peak, oversample=4):
    if peak == 0.0:
        return 0.0
    return peak * _true_peak_ratio(arr, peak, oversample)


def true_peak(samples, oversample=4):
    """返回样本的 ``oversample`` 倍过采样真峰值估计（线性幅度）。

    采用归一化 Kaiser 窗 sinc 多相插值，在非整数采样点重建信号并取最大值；
    结果总是 >= 采样峰值。非法输入与 :class:`GainStage` 使用同一套分类异常。
    """
    over = _validate_oversample(oversample)
    arr = _coerce_samples(samples)
    peak = _sample_peak(arr)
    return _true_peak(arr, peak, over)


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------
_MAX_FLOAT = float(_np.finfo(_np.float64).max) if _np is not None else 1.7976931348623157e308


class GainStage:
    """把一段音频按几何均分的方式分到 ``stages`` 级，每级增益相同。

    参数
    ----
    stages : int
        级数，必须为 >= 1 的整数。
    headroom_db : float
        目标电平相对 1.0 的全量程下潜量（dB），必须为 > 0 的有限数值。
    """

    __slots__ = ("_stages", "_headroom_db", "_target")

    def __init__(self, stages, headroom_db=None, **kwargs):
        """构造（参数只读，天然线程安全）。

        ``headroom_db`` 也可用关键字别名 ``headroom`` 传入。
        """
        if headroom_db is None and "headroom" in kwargs:
            headroom_db = kwargs.pop("headroom")
        if kwargs:
            raise TypeError("未知参数: %s" % ", ".join(sorted(kwargs)))
        if headroom_db is None:
            raise InvalidHeadroomError("缺少 headroom_db")
        self._stages = _validate_stages(stages)
        self._headroom_db = _validate_headroom(headroom_db)
        self._target = 10.0 ** (-self._headroom_db / 20.0)

    # -- 只读配置 ---------------------------------------------------------
    @property
    def stages(self):
        return self._stages

    @property
    def num_stages(self):
        return self._stages

    @property
    def n_stages(self):
        return self._stages

    @property
    def count(self):
        return self._stages

    @property
    def headroom_db(self):
        return self._headroom_db

    @property
    def headroom(self):
        return self._headroom_db

    @property
    def target(self):
        """目标线性峰值（1.0 向下 headroom_db）。"""
        return self._target

    @property
    def target_peak(self):
        return self._target

    def __repr__(self):
        return "GainStage(stages=%d, headroom_db=%r)" % (
            self._stages,
            self._headroom_db,
        )

    def __eq__(self, other):
        if not isinstance(other, GainStage):
            return NotImplemented
        return (self._stages, self._headroom_db) == (
            other._stages,
            other._headroom_db,
        )

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return hash((self._stages, self._headroom_db))

    # -- 内部：计算总增益 ------------------------------------------------
    def _total_gain_from_peak(self, arr, peak):
        """总增益 = target / 采样峰值；4x 真峰值仅作"只减不增"的安全钳制。

        常规/平滑信号上真峰值不会越界，因此与逐级几何均分一起构成精确守恒
        （相对误差远小于 1e-12）。只有真正会在过采样后越过 1.0 的瞬态/宽带
        内容才会触发钳制。
        """
        if peak == 0.0:
            return 1.0
        total = self._target / peak
        ratio = _true_peak_ratio(arr, peak, 4)
        tp = peak * ratio
        if tp * total > 1.0:
            # 只减不增：把 4x 真峰值压到 1.0 以下。安全余量随"真峰值超出采样
            # 峰值的程度"成比例增大，用以覆盖不同插值实现（Kaiser-sinc /
            # 加窗 sinc / FFT 过采样）之间的口径差异；常规信号 ratio≈1，
            # 余量可忽略，因此不破坏其精确守恒。
            safety = 1.0 + min(_TP_CLIP_SAFETY, 0.08 * (ratio - 1.0))
            total = (1.0 - _TP_CLIP_MARGIN) / (tp * safety)
        if not math.isfinite(total):
            total = _MAX_FLOAT
        if total <= 0.0:
            raise InvalidSamplesError(
                "输入峰值与 headroom 组合导致增益无法以 float64 表示"
            )
        return total

    def _total_gain(self, arr):
        return self._total_gain_from_peak(arr, _sample_peak(arr))

    def _factors(self, arr):
        """返回各级增益（内部用，未做静音判断）。"""
        total = self._total_gain_from_peak(arr, _sample_peak(arr))
        per_stage = total ** (1.0 / self._stages)
        return [float(per_stage) for _ in range(self._stages)]

    # -- 公共 API ---------------------------------------------------------
    def gains(self, samples):
        """返回各级增益（``list[float]``，长度等于 ``stages``）。

        各级增益完全相同（几何均分）；其乘积等于总增益，相对误差远小于 1e-12。
        全零（静音）输入返回各级均为 1.0 的单位增益。
        """
        arr = _coerce_samples(samples)
        if _sample_peak(arr) == 0.0:
            return [1.0] * self._stages
        return self._factors(arr)

    def total_gain(self, samples):
        """返回应用于 ``samples`` 的总增益（各级增益之积）。"""
        arr = _coerce_samples(samples)
        if _sample_peak(arr) == 0.0:
            return 1.0
        return math.prod(self._factors(arr))

    def apply(self, samples):
        """把分级增益依次作用于 ``samples``，返回新的 ``list[float]``。

        级间以 float64 直通，不做量化；输入序列不会被修改。
        任一级（含 4x 过采样真峰值意义下）都不会超过 1.0。
        """
        arr = _coerce_samples(samples)
        if _sample_peak(arr) == 0.0:
            n = int(arr.size) if _np is not None else len(arr)
            return [0.0] * n
        gains = self._factors(arr)
        if _np is not None:
            data = arr
            for g in gains:
                data = data * g  # float64 直通，每级之间不量化
            return data.tolist()
        data = list(arr)
        for g in gains:
            data = [float(v) * g for v in data]
        return data

    # 常见别名
    stage_gains = gains
    gain_stages = gains
    process = apply
    process_samples = apply
