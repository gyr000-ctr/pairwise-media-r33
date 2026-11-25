"""declick 核心 API（初始未实现）。"""


class Declick:
    def __init__(self, threshold, max_len):
        """构造"""
        raise NotImplementedError

    def detect(self, samples):
        """检出"""
        raise NotImplementedError

    def repair(self, samples):
        """修复"""
        raise NotImplementedError
