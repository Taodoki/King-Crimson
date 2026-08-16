#!/usr/bin/env python3
"""quantbt 信号输出 — 一键运行入口。

本程序只生成人工可执行的调仓清单，不接入真实券商、不真实下单。

用法:
    # 生成今日调仓清单:
    python run_live.py

    # 显式模拟模式 (与默认等价):
    python run_live.py --dry-run

    # 指定配置文件:
    python run_live.py --env my_config.env

    # 跑一个月 (每天检查并输出调仓清单):
    python run_live.py --month
"""

import sys
import argparse
from pathlib import Path

# 确保包在路径上
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="quantbt — A股量化信号输出（人工调仓清单）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--env",
        default=None,
        help="配置文件路径 (默认: .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟运行，不真实下单",
    )
    parser.add_argument(
        "--month",
        action="store_true",
        help="持续运行一个月 (默认只跑当天)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    args = parser.parse_args()

    # 配置
    from quantbt.trader.live_config import load_config

    config = load_config(args.env)

    # 日志
    import logging

    level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(
                log_dir / f"live_{__import__('datetime').date.today().isoformat()}.log",
                encoding="utf-8",
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("run_live")

    # 打印配置预览
    print("=" * 60)
    print("quantbt A股量化信号输出")
    print("=" * 60)
    print(f"  策略: {config['strategy_name']}")
    print(f"  参数: {config['strategy_params']}")
    print(f"  股票池: {', '.join(config['stock_pool'])}")
    print(f"  最大持仓: {config['max_positions']}")
    print(f"  初始资金: {config['initial_capital']:,.0f}")
    print(f"  模式: 信号输出（人工调仓清单，不接入真实券商）")
    print(f"  运行: {'一个月 (每天)' if args.month else '仅今天'}")
    print("=" * 60)

    # 构建策略
    from quantbt.strategy.momentum import TimeSeriesMomentum
    from quantbt.strategy.mean_reversion import ZScoreMeanReversion

    strategy_map = {
        "momentum": TimeSeriesMomentum,
        "mean_reversion": ZScoreMeanReversion,
    }
    strategy_cls = strategy_map.get(config["strategy_name"])
    if strategy_cls is None:
        logger.error("未知策略: %s (可选: momentum, mean_reversion)", config["strategy_name"])
        sys.exit(1)

    strategy = strategy_cls(**config["strategy_params"])
    logger.info("策略初始化完成: %s", strategy)

    # 实盘已移除，恒为信号输出
    from quantbt.trader.tonghuashun import TonghuashunTrader

    trader = TonghuashunTrader(exe_path=config["ths_exe_path"])

    if not args.dry_run:
        logger.warning("实盘已移除：本程序仅输出人工调仓清单，不真实下单。")
        args.dry_run = True
    logger.info("信号输出模式：生成人工调仓清单，不接入真实券商")

    # 执行器
    from quantbt.trader.executor import LiveExecutor

    executor = LiveExecutor(
        trader=trader,
        max_positions=config["max_positions"],
    )

    # 调度器
    from quantbt.trader.scheduler import DailyScheduler

    scheduler = DailyScheduler(
        strategy=strategy,
        stock_pool=config["stock_pool"],
        trader=trader,
        executor=executor,
        initial_capital=config["initial_capital"],
        rebalance_freq=config["rebalance_freq"],
    )

    # 运行
    if args.month:
        results = scheduler.run_month(dry_run=args.dry_run)
        success = sum(1 for r in results if "error" not in r)
        failed = sum(1 for r in results if "error" in r)
        print(f"\n{'='*60}")
        print(f"月度运行完成: {success} 天成功, {failed} 天失败")
    else:
        from datetime import date
        result = scheduler.run_once(dry_run=args.dry_run)
        if "error" in result:
            print(f"\n运行失败: {result['error']}")
            sys.exit(1)
        # 单日模式同样输出错误明细，避免 errors 计数被静默吞掉
        exec_result = result.get("execution", {})
        errors = exec_result.get("errors", [])
        if errors:
            print(f"\n执行错误明细 ({len(errors)}):")
            for e in errors:
                print(f"  - {e.get('symbol', '?')} ({e.get('side', '?')}): {e.get('reason', 'unknown')}")
        scheduler._print_daily_report(result)

    # 断开同花顺
    if trader.connected:
        trader.disconnect()

    print("\n运行完成。")


if __name__ == "__main__":
    main()