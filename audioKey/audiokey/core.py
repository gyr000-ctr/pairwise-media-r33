"""audiokey 核心 API（初始未实现）。"""


class KeyDetect:
    def __init__(self, rate):
        """构造"""
        raise NotImplementedError

    def chroma(self, samples):
        """色度谱"""
        raise NotImplementedError

    def detect(self, samples):
        """检测调性"""
        raise NotImplementedError
