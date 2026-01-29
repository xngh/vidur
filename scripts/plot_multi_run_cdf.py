from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vidur.metrics.constants import WorkflowMetrics

# =========================
# 用户可修改区域
# =========================
# 运行方式（项目根目录下）：
#   python -m scripts.plot_multi_run_cdf

# 选择要绘制的指标（一次只画一个 CDF）
# 可选值参考 WorkflowMetrics（定义在 vidur/metrics/constants.py）
METRIC_NAME = WorkflowMetrics.WORKFLOW_E2E_TIME.value

# 要对比的实验结果列表：name 用于图例，dir 指向单次运行的输出目录
# 例如：/home/linchx/vidur/simulator_output/2026-01-29_21-01-02-820028
RUNS = [
    {"name": "parrot", "dir": "/home/linchx/vidur/simulator_output/2026-01-29_21-01-02-820028"},
    {"name": "sharp", "dir": "/home/linchx/vidur/simulator_output/2026-01-29_21-03-52-262066"},
]

# 输出目录（会自动创建）
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "simulator_output" / "comparison_plots"

# =========================
# 一般无需修改区域
# =========================

X_AXIS_LABELS = {
    WorkflowMetrics.WORKFLOW_SLO_ATTAINMENT.value: "Count",
}


def _load_cdf_csv(run_dir: Path, metric_name: str) -> pd.DataFrame:
    csv_path = run_dir / "plots" / f"{metric_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到指标 CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "cdf" not in df.columns:
        if metric_name not in df.columns:
            raise ValueError(
                f"{csv_path} 中缺少列: {metric_name} 或 cdf，无法绘制 CDF"
            )
        df["cdf"] = df[metric_name].rank(method="first", pct=True)

    if metric_name not in df.columns:
        raise ValueError(f"{csv_path} 中缺少列: {metric_name}")

    return df.sort_values("cdf")


def _get_x_label(metric_name: str) -> str:
    return X_AXIS_LABELS.get(metric_name, "Time (sec)")


def plot_multi_run_cdf(metric_name: str, runs: list[dict[str, str]]) -> Path:
    if not runs:
        raise ValueError("RUNS 为空，请在脚本顶部添加需要对比的实验目录。")

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{metric_name}_cdf_compare.png"

    plt.figure(figsize=(7, 5))
    for run in runs:
        run_name = run["name"]
        run_dir = Path(run["dir"])
        df = _load_cdf_csv(run_dir, metric_name)
        plt.plot(df[metric_name], df["cdf"], label=run_name)

    plt.xlabel(_get_x_label(metric_name))
    plt.ylabel("CDF")
    plt.title(f"{metric_name} CDF")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main() -> None:
    output_path = plot_multi_run_cdf(METRIC_NAME, RUNS)
    print(f"已保存对比图: {output_path}")


if __name__ == "__main__":
    main()
