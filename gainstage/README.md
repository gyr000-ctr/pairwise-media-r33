# gainstage

## 题面（可直接粘贴）

子目录：gainstage。增益分级（headroom）。约束：按峰值与目标 headroom 计算各级增益且总增益守恒（≤1e-12）；任何一级不得削波（峰值 ≤1）；级数可配且每级增益按几何均分；静音输入返回单位增益；100 万样本不超过 400 ms；非法 headroom 可区分报错。验收：cd gainstage && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gainstage && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
