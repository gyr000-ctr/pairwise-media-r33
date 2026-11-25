# audioKey

## 题面（可直接粘贴）

子目录：audioKey。调性检测（Krumhansl 模板）。约束：必须把色度谱与 24 个模板（12 大调/12 小调）做相关，取最大者；与暴力相关计算一致（≤1e-9）；色度谱按 A4=440 折叠；静音返回 None；100 万样本不超过 900 ms；非法模板可区分报错。验收：cd audiokey && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd audioKey && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
