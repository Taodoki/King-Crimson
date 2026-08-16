# quantbt — A股量化策略回测平台

quantbt 是一个面向 A 股的量化策略回测与信号输出框架，实现经典策略 → 回测引擎 → 绩效归因 → 信号输出的完整闭环。

## 快速开始

```python
from quantbt.api import Backtest
from quantbt.strategy import TimeSeriesMomentum

result = Backtest(
    strategy=TimeSeriesMomentum(lookback=60),
    symbols=['000001', '000002', '600000'],
    start='2023-01-01',
    end='2024-12-31',
).run()

result.summary()           # 终端打印绩效总表
result.trades              # 交易明细
result.plot_equity()       # 保存净值曲线 HTML
```

## 安装

```bash
pip install -r requirements.txt
```

## 功能

### 策略动量 / 均值回归

| 策略 | 说明 |
|---|---|
| `TimeSeriesMomentum` | 时间序列动量：过去 N 日收益 > 阈值则做多 |
| `CrossSectionalMomentum` | 截面动量：做多过去收益最高的前 20% |
| `ZScoreMeanReversion` | Z-score 均值回归：超卖时买入，回归后退出 |
| `CompositeStrategy` | 多策略组合 |

### 回测引擎

- 逐日撮合 + 逐日估值（信号计算向量化），A 股日频约 250 次迭代/年
- A 股规则原生支持：印花税卖出征收（自动按 2023-08-28 降税分段）、T+1、
  1 手整数、零股一次性卖出、成交量限制、涨跌停（主板 ±10%、创业板/科创板
  ±20%、北交所 ±30%，按板块与生效日期区分）
- 停牌股沿用最后收盘价估值（不会把持仓记成 0）
- 信号权重按多头总和归一化到 ≤1：引擎模拟现金账户而非融资账户，
  vol_target 缩放产生的 >1 权重会被等比压缩，不会出现负现金
- shift(1) 在 Strategy 基类约定并测试锁定，杜绝未来函数
- 状态机策略（`ZScoreMeanReversion`）声明 `event_driven=True`，止损/退出在
  信号变化当日成交，不等待下一次调仓日

### 绩效归因

- 收益分析：累计/年化收益、月度热力图、滚动收益
- 风险指标：Sharpe、Sortino（标准下偏矩口径）、Calmar、VaR(95%)、CVaR、
  最大回撤及恢复期
- 因子暴露：自定义因子 OLS / 滚动回归（`FactorAnalyzer`；注意传入的必须是
  **日收益**序列，`build_market_factor` 返回的是累计净值曲线，仅用于展示，
  不可直接喂给 `fit()`）
- 基准对比：超额收益、beta、跟踪误差（benchmark 参数）

### 实盘信号输出（人工调仓清单）

实盘为模拟信号输出，不接入真实券商。策略信号经 `run_live.py` 生成一份人工可执行的调仓清单，由使用者自行在券商客户端手动下单：

```bash
# 生成今日调仓清单
python run_live.py

# 显式模拟模式（与默认等价）
python run_live.py --dry-run
```

## 项目结构

```
quantbt/
├── quantbt/
│   ├── api.py              # Backtest 入口
│   ├── core/               # 引擎、组合、Broker、日历
│   ├── data/               # 数据源（baostock / akshare）
│   ├── strategy/           # 策略实现
│   ├── attribution/        # 绩效归因
│   ├── trader/             # 信号输出（人工调仓清单）
│   ├── plot/               # Plotly 图表
│   └── utils/              # Rich 终端主题
├── tests/                  # 24+ 测试
└── run_live.py             # 信号输出入口
```

## 数据源

默认使用 **baostock**（免费、免注册、免 token）。也支持 AKShare（需网络连接东方财富）。
两个数据源的成交量单位已归一化为股（AKShare 原始单位为手，内部 ×100）。

## 建模约束（使用前必读）

- **不建模融资融券**：多头权重归一化到 ≤1，回测不含杠杆成本与强平。
- **ST 股 ±5% 涨跌幅未建模**（需要 ST 标记数据），北交所代码在 baostock 数据源
  中取数受限。
- **前复权（qfq）价格**用于收益与调仓计算；除权日当天的涨跌停判断存在近似误差。
  需要精确模拟涨跌停成交时，请提供不复权价格。
- **印花税**：显式传 `stamp_duty` 会锁定固定税率；不传则按历史税率分段
  （2023-08-28 前 0.1%、之后 0.05%）。
- **成交量限制**：单笔订单上限为当日成交量的 `volume_limit`（默认 1%），
  超过上限的部分不会补单，会在下一次调仓时再处理。
- **实盘信号输出**：`run_live.py` 只生成人工调仓清单，不接真实券商；
  `--month` 模式为每日独立模拟（每日按初始资金重算），不代表连续持仓演化。

## License

MIT