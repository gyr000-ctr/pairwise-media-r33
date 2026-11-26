"""declick 核心 API：基于二阶差分的咔嗒声检测与修复。

检测
----
对信号 x 计算二阶差分::

    d2[i] = x[i-1] - 2*x[i] + x[i+1],   1 <= i <= n-2

当 ``|d2[i]| > threshold`` 时把 i 视为突变点。连续的突变点合并成一个
窗口，窗口两侧各扩展一个样本（保证位于数组首尾的咔嗒声也被覆盖），
再按 ``max_len`` 切分后作为检测结果返回。整个检测过程不修改输入。

修复
----
只在检测到的窗口内部写入新样本，窗口外的样本逐样本原样复制。窗口内部
使用三次 Hermite 插值（Catmull-Rom 式单侧切线，并对端点区间做钳制以
避免过冲）；窗口贴近数组边界、缺少一侧锚点时使用可用的那一侧锚点做
常数延拓；若整个数组都被判为突变则没有可靠锚点，此时保持原样。
"""

import math

__all__ = ["Declick"]


class Declick:
    """咔嗒声检测 / 修复器。

    参数
    ----
    threshold : float
        二阶差分检测阈值，必须是有限正数。
    max_len : int
        单个检测窗口允许的最大样本数，必须是正整数。更长的突变区会被
        切分为多个不超过 ``max_len`` 的窗口。
    """

    __slots__ = ("_threshold", "_max_len")

    def __init__(self, threshold, max_len):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a real number (int or float)")
        threshold = float(threshold)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")

        if isinstance(max_len, bool) or not isinstance(max_len, int):
            raise TypeError("max_len must be an integer")
        if max_len <= 0:
            raise ValueError("max_len must be a positive integer")

        object.__setattr__(self, "_threshold", threshold)
        object.__setattr__(self, "_max_len", max_len)

    # 只读属性：实例构造后不再变化，方法天然并发安全。
    @property
    def threshold(self):
        return self._threshold

    @property
    def max_len(self):
        return self._max_len

    def __setattr__(self, name, value):
        raise AttributeError("Declick instances are immutable")

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce(samples):
        """把输入转换为 float 列表，并对非法输入分类报错。"""
        if samples is None or isinstance(samples, (str, bytes)):
            raise TypeError("samples must be an iterable of real numbers")
        try:
            iterator = iter(samples)
        except TypeError:
            raise TypeError("samples must be an iterable of real numbers") from None

        try:
            values = [float(v) for v in iterator]
        except (TypeError, ValueError):
            raise TypeError("samples must contain only real numbers") from None

        if not all(map(math.isfinite, values)):
            raise ValueError("samples must contain only finite values")
        return values

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
    def _find_windows(self, xs):
        """返回合并、扩展后的窗口列表（闭区间，元素为 int）。"""
        n = len(xs)
        if n < 3:
            return []

        threshold = self._threshold
        runs = []
        start = -1

        prev = xs[0]
        cur = xs[1]
        # i 为二阶差分中心，i in [1, n-2]
        for i in range(1, n - 1):
            nxt = xs[i + 1]
            d = prev - 2.0 * cur + nxt
            if d > threshold or d < -threshold:
                if start < 0:
                    start = i
            elif start >= 0:
                runs.append((start, i - 1))
                start = -1
            prev = cur
            cur = nxt

        if start >= 0:
            runs.append((start, n - 2))

        if not runs:
            return []

        last = n - 1
        windows = []
        for s, e in runs:
            # 左右各扩展一个样本，覆盖位于数组边界的咔嗒声。
            s -= 1
            e += 1
            if s < 0:
                s = 0
            if e > last:
                e = last
            if windows and s <= windows[-1][1] + 1:
                # 扩展后相邻或重叠的窗口合并为一个连续修复区。
                if e > windows[-1][1]:
                    windows[-1][1] = e
            else:
                windows.append([s, e])
        return windows

    def detect(self, samples):
        """检出突变窗口，返回 ``[[start, end], ...]``（闭区间、升序）。"""
        xs = self._coerce(samples)
        windows = self._find_windows(xs)

        max_len = self._max_len
        spans = []
        for s, e in windows:
            length = e - s + 1
            if length <= max_len:
                spans.append([s, e])
                continue
            # 均分为若干段，每段长度都不超过 max_len。
            parts = -(-length // max_len)
            base, extra = divmod(length, parts)
            cursor = s
            for k in range(parts):
                size = base + (1 if k < extra else 0)
                spans.append([cursor, cursor + size - 1])
                cursor += size
        return spans

    # ------------------------------------------------------------------
    # 修复
    # ------------------------------------------------------------------
    @staticmethod
    def _repair_window(out, a, b):
        """用三次 Hermite 插值修复 out[a:b+1]（闭区间）。"""
        n = len(out)
        if a == 0 and b == n - 1:
            # 没有任何干净锚点，无法可靠插值：保持原样。
            return
        if a == 0:
            right = out[b + 1]
            for i in range(a, b + 1):
                out[i] = right
            return
        if b == n - 1:
            left = out[a - 1]
            for i in range(a, b + 1):
                out[i] = left
            return

        left = out[a - 1]
        right = out[b + 1]
        # Catmull-Rom 式单侧切线（每个样本的斜率）。
        slope0 = left - out[a - 2] if a >= 2 else 0.0
        slope1 = out[b + 2] - right if b <= n - 3 else 0.0

        h = float(b - a + 2)  # 锚点之间的样本间隔数
        lo = left if left < right else right
        hi = right if left < right else left

        for j in range(1, b - a + 2):
            t = j / h
            t2 = t * t
            t3 = t2 * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            value = h00 * left + h10 * (h * slope0) + h01 * right + h11 * (h * slope1)
            # 钳制到两端锚点之间，避免插值过冲。
            if value < lo:
                value = lo
            elif value > hi:
                value = hi
            out[a - 1 + j] = value

    def repair(self, samples):
        """返回修复后的新列表，未检出区域逐样本保持不变。"""
        xs = self._coerce(samples)
        out = list(xs)
        for a, b in self._find_windows(xs):
            self._repair_window(out, a, b)
        return out
