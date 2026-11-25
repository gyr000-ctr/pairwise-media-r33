"""chord 核心 API（初始未实现）。"""


class Chord:
    def __init__(self, templates):
        """构造"""
        raise NotImplementedError

    def detect(self, samples):
        """识别"""
        raise NotImplementedError

    def probs(self, samples):
        """概率"""
        raise NotImplementedError
