# chord

## 题面（可直接粘贴）

子目录：chord。和弦识别。约束：按模板（大三/小三/减/增/七）匹配色度谱并给出置信度；与逐模板相关对照（≤1e-9）；置信度为 Softmax 且和为 1（≤1e-12）；无和弦返回 None；100 万样本不超过 900 ms；非法模板集可区分报错。验收：cd chord && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd chord && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
