# crosstalk

## 题面（可直接粘贴）

子目录：crosstalk。声道串音消除。约束：用最小二乘估计串音系数（与真值差 ≤1e-3）；消串后源信号 SNR 提升 ≥10 dB；增益矩阵必须可逆（接近奇异时报错）；立体声对单声道输入不做处理；100 万样本不超过 700 ms；非法参数可区分报错。验收：cd crosstalk && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd crosstalk && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
