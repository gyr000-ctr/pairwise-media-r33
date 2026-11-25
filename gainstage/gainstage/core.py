"""gainstage 核心 API（初始未实现）。"""


class GainStage:
    def __init__(self, stages, headroom_db):
        """构造"""
        raise NotImplementedError

    def gains(self, samples):
        """各级增益"""
        raise NotImplementedError

    def apply(self, samples):
        """应用"""
        raise NotImplementedError
