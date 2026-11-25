"""crosstalk 核心 API（初始未实现）。"""


class Crosstalk:
    def __init__(self, est_len):
        """构造"""
        raise NotImplementedError

    def estimate(self, left, right):
        """估计系数"""
        raise NotImplementedError

    def cancel(self, left, right):
        """消串"""
        raise NotImplementedError
