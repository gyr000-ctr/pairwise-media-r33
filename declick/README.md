# declick

## 题面（可直接粘贴）

子目录：declick。去咔嗒声。约束：用二阶差分检测突变并只在检出窗口内做插值修复；未检出区域逐样本不变（≤1e-12）；插值用 AR 或三次 Hermite 且修复后二阶差分有界；检测阈值可配且误检 ≤2%；100 万样本不超过 700 ms；非法阈值可区分报错。验收：cd declick && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd declick && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
