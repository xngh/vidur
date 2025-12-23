
from typing import List, Dict, Optional
from vidur.entities.unified_request import UnifiedRequest
from vidur.entities.full_request import FullRequest
from vidur.request_generator.sharegpt_request_generator import ShareGPTRequestGenerator
from vidur.config.config import ShareGPTRequestGeneratorConfig
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")


def plot_distribution(requests: List[FullRequest]):
    if not requests:
        print("请求列表为空，无法绘制CDF图")
        return

        # 收集输入和输出token长度
    input_lengths = []
    output_lengths = []

    for request in requests:
        # 获取输入token_ids的长度
        if hasattr(request, 'input_token_ids') and request.input_token_ids is not None:
            input_lengths.append(len(request.input_token_ids))
        else:
            input_lengths.append(0)

        # 获取输出token_ids的长度
        if hasattr(request, 'output_token_ids') and request.output_token_ids is not None:
            output_lengths.append(len(request.output_token_ids))
        else:
            output_lengths.append(0)

    # 计算总长度
    total_lengths = [i + o for i, o in zip(input_lengths, output_lengths)]

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 计算CDF
    def calculate_cdf(data):
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, cdf

    # 计算输入、输出和总长度的CDF
    sorted_input, input_cdf = calculate_cdf(input_lengths)
    sorted_output, output_cdf = calculate_cdf(output_lengths)
    sorted_total, total_cdf = calculate_cdf(total_lengths)

    # 绘制CDF曲线
    plt.plot(sorted_input, input_cdf,
             label=f'Input Tokens (n={len(input_lengths)})',
             linewidth=3, color='#1f77b4', linestyle='-', marker='', markersize=0)

    plt.plot(sorted_output, output_cdf,
             label=f'Output Tokens (n={len(output_lengths)})',
             linewidth=3, color='#ff7f0e', linestyle='-', marker='', markersize=0)

    plt.plot(sorted_total, total_cdf,
             label=f'Total Tokens (Input+Output)',
             linewidth=3, color='#2ca02c', linestyle='-', marker='', markersize=0)

    # 添加垂直线标注关键分位数
    percentiles = [50, 75, 90, 95, 99]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    datasets = [(sorted_input, input_cdf, 'Input'),
                (sorted_output, output_cdf, 'Output'),
                (sorted_total, total_cdf, 'Total')]

    for percentile in percentiles:
        plt.axvline(x=np.percentile(total_lengths, percentile),
                    color='red', linestyle='--', alpha=0.3, linewidth=1)
        plt.text(np.percentile(total_lengths, percentile), 0.02,
                 f'{percentile}%', rotation=90, verticalalignment='bottom',
                 horizontalalignment='right', fontsize=9, color='red')

    # 设置图形属性
    plt.xlabel('Token Length', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    plt.title('CDF of Request Token Lengths', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='lower right', fontsize=12, framealpha=0.9)

    # 设置坐标轴范围
    plt.xlim(left=0)
    plt.ylim([0, 1.05])

    # 添加刻度线
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2, linestyle=':')

    # 在图上添加统计表格
    stats_text = "Statistics Summary:\n\n"

    # 输入长度统计
    stats_text += "Input Tokens:\n"
    stats_text += f"  Mean: {np.mean(input_lengths):.1f}\n"
    stats_text += f"  Median: {np.median(input_lengths):.1f}\n"
    stats_text += f"  90th %ile: {np.percentile(input_lengths, 90):.1f}\n"
    stats_text += f"  99th %ile: {np.percentile(input_lengths, 99):.1f}\n\n"

    # 输出长度统计
    stats_text += "Output Tokens:\n"
    stats_text += f"  Mean: {np.mean(output_lengths):.1f}\n"
    stats_text += f"  Median: {np.median(output_lengths):.1f}\n"
    stats_text += f"  90th %ile: {np.percentile(output_lengths, 90):.1f}\n"
    stats_text += f"  99th %ile: {np.percentile(output_lengths, 99):.1f}\n\n"

    # 总长度统计
    stats_text += "Total Tokens:\n"
    stats_text += f"  Mean: {np.mean(total_lengths):.1f}\n"
    stats_text += f"  Median: {np.median(total_lengths):.1f}\n"
    stats_text += f"  90th %ile: {np.percentile(total_lengths, 90):.1f}\n"
    stats_text += f"  99th %ile: {np.percentile(total_lengths, 99):.1f}"

    # 将统计信息放在图的右上角
    plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # 调整布局
    plt.tight_layout()

    # 打印详细统计信息到控制台
    print("\n" + "=" * 80)
    print("DETAILED TOKEN LENGTH STATISTICS (CDF Analysis)")
    print("=" * 80)

    print(f"\nTotal number of requests: {len(requests)}")

    print("\nInput Token Length Statistics:")
    print("-" * 40)
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    for p in percentiles:
        value = np.percentile(input_lengths, p)
        print(f"{p:5.1f}th percentile: {value:8.1f} tokens")

    print("\nOutput Token Length Statistics:")
    print("-" * 40)
    for p in percentiles:
        value = np.percentile(output_lengths, p)
        print(f"{p:5.1f}th percentile: {value:8.1f} tokens")

    print("\nTotal Token Length Statistics:")
    print("-" * 40)
    for p in percentiles:
        value = np.percentile(total_lengths, p)
        print(f"{p:5.1f}th percentile: {value:8.1f} tokens")

    print("\nKey Insights:")
    print("-" * 40)
    print(f"1. 50% of requests have ≤ {np.median(total_lengths):.0f} total tokens")
    print(f"2. 90% of requests have ≤ {np.percentile(total_lengths, 90):.0f} total tokens")
    print(f"3. 99% of requests have ≤ {np.percentile(total_lengths, 99):.0f} total tokens")
    print(f"4. Max total tokens: {np.max(total_lengths):.0f}")
    print(f"5. Average total tokens: {np.mean(total_lengths):.1f}")

    # 显示图形
    plt.show()

    # 返回统计数据以便进一步分析
    return {
        'input_lengths': input_lengths,
        'output_lengths': output_lengths,
        'total_lengths': total_lengths,
        'input_cdf': (sorted_input, input_cdf),
        'output_cdf': (sorted_output, output_cdf),
        'total_cdf': (sorted_total, total_cdf)
    }


def prepare_data(requests: List[UnifiedRequest]) -> List[FullRequest]:
    requests = []

    for unified_request in unified_requests:
        content = ""

        while (full_requests := unified_request.get_next_requests(0, content)) and len(full_requests) > 0:
            for full_request in full_requests:
                requests.append(full_request)
                content += full_request.input_str
                content += full_request.output_str
                unified_request.update_on_request_finish(full_request, 0)
    return requests

# 这里暂时没考虑map_reduce
def prepare_data_v2(unified_requests: List[UnifiedRequest]) -> List[Dict]:
    requests = []
    for request in unified_requests:
        content = 0
        for config in request.workflow_config:
            input_tokens = GLOBAL_TOKENIZER.encode(config["input_str"])
            output_tokens = GLOBAL_TOKENIZER.encode(config["output_str"])
            req = {"input": len(input_tokens) + content, "output": len(output_tokens)}

            requests.append(req)
            content = req["input"] + req["output"]

    return requests


def plot_cdf_distribution_v2(requests: List[Dict]):
    """
    绘制请求的输入和输出token长度的CDF图
    适应新的数据结构：每个请求是一个字典，包含"input"和"output"键

    参数:
    requests: 字典列表，每个字典包含"input"和"output"键，表示token长度
    """
    if not requests:
        print("请求列表为空，无法绘制CDF图")
        return

    # 从字典中提取输入和输出长度
    input_lengths = [req["input"] for req in requests if "input" in req]
    output_lengths = [req["output"] for req in requests if "output" in req]

    # 计算总长度（输入+输出）
    total_lengths = []
    for req in requests:
        if "input" in req and "output" in req:
            total_lengths.append(req["input"] + req["output"])

    print(f"分析 {len(requests)} 个请求")
    print(f"有输入长度的请求数: {len(input_lengths)}")
    print(f"有输出长度的请求数: {len(output_lengths)}")
    print(f"有完整输入输出长度的请求数: {len(total_lengths)}")

    # 创建图形
    plt.figure(figsize=(14, 10))

    # 计算CDF
    def calculate_cdf(data):
        if len(data) == 0:
            return np.array([]), np.array([])
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, cdf

    # 计算输入、输出和总长度的CDF
    sorted_input, input_cdf = calculate_cdf(input_lengths)
    sorted_output, output_cdf = calculate_cdf(output_lengths)
    sorted_total, total_cdf = calculate_cdf(total_lengths)

    # 创建子图：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 输入长度CDF
    if len(input_lengths) > 0:
        axes[0].plot(sorted_input, input_cdf,
                     label=f'Input Tokens (n={len(input_lengths)})',
                     linewidth=2.5, color='#1f77b4')

        # 添加关键分位数线
        percentiles = [50, 75, 90, 95, 99]
        for percentile in percentiles:
            if percentile <= 100:
                value = np.percentile(input_lengths, percentile)
                axes[0].axvline(x=value, color='red', linestyle='--', alpha=0.3, linewidth=1)
                axes[0].text(value, 0.02, f'{percentile}%', rotation=90,
                             verticalalignment='bottom', horizontalalignment='right',
                             fontsize=8, color='red')

        axes[0].set_xlabel('Input Token Length', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        axes[0].set_title('CDF of Input Token Lengths', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].set_xlim(left=0)
        axes[0].set_ylim([0, 1.05])
    else:
        axes[0].text(0.5, 0.5, 'No input data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('No Input Data', fontsize=14, fontweight='bold')

    # 2. 输出长度CDF
    if len(output_lengths) > 0:
        axes[1].plot(sorted_output, output_cdf,
                     label=f'Output Tokens (n={len(output_lengths)})',
                     linewidth=2.5, color='#ff7f0e')

        # 添加关键分位数线
        for percentile in percentiles:
            if percentile <= 100:
                value = np.percentile(output_lengths, percentile)
                axes[1].axvline(x=value, color='red', linestyle='--', alpha=0.3, linewidth=1)
                axes[1].text(value, 0.02, f'{percentile}%', rotation=90,
                             verticalalignment='bottom', horizontalalignment='right',
                             fontsize=8, color='red')

        axes[1].set_xlabel('Output Token Length', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        axes[1].set_title('CDF of Output Token Lengths', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(loc='lower right', fontsize=10)
        axes[1].set_xlim(left=0)
        axes[1].set_ylim([0, 1.05])
    else:
        axes[1].text(0.5, 0.5, 'No output data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('No Output Data', fontsize=14, fontweight='bold')

    # 3. 总长度CDF
    if len(total_lengths) > 0:
        axes[2].plot(sorted_total, total_cdf,
                     label=f'Total Tokens (Input+Output)',
                     linewidth=2.5, color='#2ca02c')

        # 添加关键分位数线
        for percentile in percentiles:
            if percentile <= 100:
                value = np.percentile(total_lengths, percentile)
                axes[2].axvline(x=value, color='red', linestyle='--', alpha=0.3, linewidth=1)
                axes[2].text(value, 0.02, f'{percentile}%', rotation=90,
                             verticalalignment='bottom', horizontalalignment='right',
                             fontsize=8, color='red')

        axes[2].set_xlabel('Total Token Length', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        axes[2].set_title('CDF of Total Token Lengths', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].legend(loc='lower right', fontsize=10)
        axes[2].set_xlim(left=0)
        axes[2].set_ylim([0, 1.05])
    else:
        axes[2].text(0.5, 0.5, 'No total data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('No Total Data', fontsize=14, fontweight='bold')

    # 添加整体标题
    fig.suptitle('Token Length Distribution (CDF)', fontsize=16, fontweight='bold', y=1.02)

    # 调整布局
    plt.tight_layout()

    # 打印详细统计信息
    print("\n" + "=" * 80)
    print("DETAILED TOKEN LENGTH STATISTICS")
    print("=" * 80)

    if len(input_lengths) > 0:
        print("\nInput Token Length Statistics:")
        print("-" * 40)
        print(f"  Min: {np.min(input_lengths):.0f}")
        print(f"  Max: {np.max(input_lengths):.0f}")
        print(f"  Mean: {np.mean(input_lengths):.2f}")
        print(f"  Std: {np.std(input_lengths):.2f}")
        print(f"  Median: {np.median(input_lengths):.2f}")
        for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
            if p <= 100:
                print(f"  {p}th percentile: {np.percentile(input_lengths, p):.2f}")

    if len(output_lengths) > 0:
        print("\nOutput Token Length Statistics:")
        print("-" * 40)
        print(f"  Min: {np.min(output_lengths):.0f}")
        print(f"  Max: {np.max(output_lengths):.0f}")
        print(f"  Mean: {np.mean(output_lengths):.2f}")
        print(f"  Std: {np.std(output_lengths):.2f}")
        print(f"  Median: {np.median(output_lengths):.2f}")
        for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
            if p <= 100:
                print(f"  {p}th percentile: {np.percentile(output_lengths, p):.2f}")

    if len(total_lengths) > 0:
        print("\nTotal Token Length (Input + Output) Statistics:")
        print("-" * 40)
        print(f"  Min: {np.min(total_lengths):.0f}")
        print(f"  Max: {np.max(total_lengths):.0f}")
        print(f"  Mean: {np.mean(total_lengths):.2f}")
        print(f"  Std: {np.std(total_lengths):.2f}")
        print(f"  Median: {np.median(total_lengths):.2f}")
        for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
            if p <= 100:
                print(f"  {p}th percentile: {np.percentile(total_lengths, p):.2f}")

    # 显示图形
    plt.show()

    # 返回统计数据以便进一步分析
    return {
        'input_lengths': input_lengths,
        'output_lengths': output_lengths,
        'total_lengths': total_lengths,
        'requests': requests
    }


def plot_histogram_distribution(requests: List[Dict], use_log_scale: bool = False):
    """
    绘制请求的输入和输出token长度的柱状分布图（直方图）

    参数:
    requests: 字典列表，每个字典包含"input"和"output"键，表示token长度
    use_log_scale: 是否使用对数坐标轴（对于长尾分布很有用）
    """
    if not requests:
        print("请求列表为空，无法绘制分布图")
        return

    # 从字典中提取输入和输出长度
    input_lengths = [req["input"] for req in requests if "input" in req]
    output_lengths = [req["output"] for req in requests if "output" in req]

    # 计算总长度（输入+输出）
    total_lengths = []
    for req in requests:
        if "input" in req and "output" in req:
            total_lengths.append(req["input"] + req["output"])

    print(f"分析 {len(requests)} 个请求")
    print(f"有输入长度的请求数: {len(input_lengths)}")
    print(f"有输出长度的请求数: {len(output_lengths)}")
    print(f"有完整输入输出长度的请求数: {len(total_lengths)}")

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. 输入长度分布
    if len(input_lengths) > 0:
        ax1 = axes[0]

        # 确定合适的bins数量
        max_input = max(input_lengths)
        min_input = min(input_lengths)

        # 如果数据范围很大，使用对数bins
        if max_input / (min_input + 1) > 100:  # 范围很大
            bins = np.logspace(np.log10(min_input + 1), np.log10(max_input + 1), 30)
            ax1.set_xscale('log')
        else:
            bins = 30

        # 绘制直方图
        n, bins, patches = ax1.hist(input_lengths, bins=bins,
                                    alpha=0.7, color=colors[0],
                                    edgecolor='black', linewidth=0.5)

        ax1.set_xlabel('Input Token Length', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Distribution of Input Token Lengths', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 添加统计信息
        stats_text = f'Count: {len(input_lengths)}\n'
        stats_text += f'Min: {min_input}\n'
        stats_text += f'Max: {max_input}\n'
        stats_text += f'Mean: {np.mean(input_lengths):.1f}\n'
        stats_text += f'Median: {np.median(input_lengths):.1f}\n'
        stats_text += f'Std: {np.std(input_lengths):.1f}'

        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                 fontsize=9)

        # 如果使用对数坐标
        if use_log_scale:
            ax1.set_yscale('log')
            ax1.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, 'No input data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('No Input Data', fontsize=14, fontweight='bold')

    # 2. 输出长度分布
    if len(output_lengths) > 0:
        ax2 = axes[1]

        # 确定合适的bins数量
        max_output = max(output_lengths)
        min_output = min(output_lengths)

        # 如果数据范围很大，使用对数bins
        if max_output / (min_output + 1) > 100:  # 范围很大
            bins = np.logspace(np.log10(min_output + 1), np.log10(max_output + 1), 50)
            ax2.set_xscale('log')
        else:
            bins = 30

        # 绘制直方图
        n, bins, patches = ax2.hist(output_lengths, bins=bins,
                                    alpha=0.7, color=colors[1],
                                    edgecolor='black', linewidth=0.5)

        ax2.set_xlabel('Output Token Length', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution of Output Token Lengths', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 添加统计信息
        stats_text = f'Count: {len(output_lengths)}\n'
        stats_text += f'Min: {min_output}\n'
        stats_text += f'Max: {max_output}\n'
        stats_text += f'Mean: {np.mean(output_lengths):.1f}\n'
        stats_text += f'Median: {np.median(output_lengths):.1f}\n'
        stats_text += f'Std: {np.std(output_lengths):.1f}'

        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                 fontsize=9)

        # 如果使用对数坐标
        if use_log_scale:
            ax2.set_yscale('log')
            ax2.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No output data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('No Output Data', fontsize=14, fontweight='bold')

    # 3. 总长度分布
    if len(total_lengths) > 0:
        ax3 = axes[2]

        # 确定合适的bins数量
        max_total = max(total_lengths)
        min_total = min(total_lengths)

        # 如果数据范围很大，使用对数bins
        if max_total / (min_total + 1) > 100:  # 范围很大
            bins = np.logspace(np.log10(min_total + 1), np.log10(max_total + 1), 50)
            ax3.set_xscale('log')
        else:
            bins = 30

        # 绘制直方图
        n, bins, patches = ax3.hist(total_lengths, bins=bins,
                                    alpha=0.7, color=colors[2],
                                    edgecolor='black', linewidth=0.5)

        ax3.set_xlabel('Total Token Length', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Distribution of Total Token Lengths', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 添加统计信息
        stats_text = f'Count: {len(total_lengths)}\n'
        stats_text += f'Min: {min_total}\n'
        stats_text += f'Max: {max_total}\n'
        stats_text += f'Mean: {np.mean(total_lengths):.1f}\n'
        stats_text += f'Median: {np.median(total_lengths):.1f}\n'
        stats_text += f'Std: {np.std(total_lengths):.1f}'

        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                 fontsize=9)

        # 如果使用对数坐标
        if use_log_scale:
            ax3.set_yscale('log')
            ax3.set_ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'No total data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('No Total Data', fontsize=14, fontweight='bold')

    # 添加整体标题
    log_text = " (Log Scale)" if use_log_scale else ""
    fig.suptitle(f'Token Length Distribution{log_text}', fontsize=16, fontweight='bold', y=1.02)

    # 调整布局
    plt.tight_layout()

    # 打印详细统计信息
    print("\n" + "=" * 80)
    print("DETAILED TOKEN LENGTH STATISTICS")
    print("=" * 80)

    if len(input_lengths) > 0:
        print("\nInput Token Length Statistics:")
        print("-" * 40)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"  {p:2}th percentile: {np.percentile(input_lengths, p):8.1f}")

    if len(output_lengths) > 0:
        print("\nOutput Token Length Statistics:")
        print("-" * 40)
        for p in percentiles:
            print(f"  {p:2}th percentile: {np.percentile(output_lengths, p):8.1f}")

    if len(total_lengths) > 0:
        print("\nTotal Token Length Statistics:")
        print("-" * 40)
        for p in percentiles:
            print(f"  {p:2}th percentile: {np.percentile(total_lengths, p):8.1f}")

    # 显示图形
    plt.show()


def plot_fixed_bin_histogram(
        requests: List[Dict],
        bin_width: int = 300,
        max_bin: Optional[int] = None,
        show_cumulative: bool = True,
        normalize: bool = False
):
    """
    使用固定宽度的桶绘制柱状图，如每100个token为一个区域

    参数:
    requests: 字典列表，每个字典包含"input"和"output"键，表示token长度
    bin_width: 每个桶的宽度（以token为单位），例如100表示每100个token为一个桶
    max_bin: 最大桶的边界（可选），如果不指定则自动计算
    show_cumulative: 是否显示累积分布线
    normalize: 是否将频率标准化为百分比
    """
    if not requests:
        print("请求列表为空，无法绘制分布图")
        return

    # 从字典中提取输入和输出长度
    input_lengths = [req["input"] for req in requests if "input" in req]
    output_lengths = [req["output"] for req in requests if "output" in req]

    # 计算总长度（输入+输出）
    total_lengths = []
    for req in requests:
        if "input" in req and "output" in req:
            total_lengths.append(req["input"] + req["output"])

    print(f"分析 {len(requests)} 个请求")
    print(f"有输入长度的请求数: {len(input_lengths)}")
    print(f"有输出长度的请求数: {len(output_lengths)}")
    print(f"有完整输入输出长度的请求数: {len(total_lengths)}")
    print(f"桶宽度: {bin_width} tokens")

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. 输入长度分布
    if len(input_lengths) > 0:
        _plot_single_fixed_bin(
            ax=axes[0, 0],
            data=input_lengths,
            bin_width=bin_width,
            max_bin=max_bin,
            show_cumulative=show_cumulative,
            normalize=normalize,
            title="Input Token Length Distribution",
            color='#1f77b4',
            data_name="Input"
        )
    else:
        axes[0, 0].text(0.5, 0.5, 'No input data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('No Input Data', fontsize=14, fontweight='bold')

    # 2. 输出长度分布
    if len(output_lengths) > 0:
        _plot_single_fixed_bin(
            ax=axes[0, 1],
            data=output_lengths,
            bin_width=bin_width,
            max_bin=max_bin,
            show_cumulative=show_cumulative,
            normalize=normalize,
            title="Output Token Length Distribution",
            color='#ff7f0e',
            data_name="Output"
        )
    else:
        axes[0, 1].text(0.5, 0.5, 'No output data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('No Output Data', fontsize=14, fontweight='bold')

    # 3. 总长度分布
    if len(total_lengths) > 0:
        _plot_single_fixed_bin(
            ax=axes[1, 0],
            data=total_lengths,
            bin_width=bin_width,
            max_bin=max_bin,
            show_cumulative=show_cumulative,
            normalize=normalize,
            title="Total Token Length Distribution",
            color='#2ca02c',
            data_name="Total"
        )
    else:
        axes[1, 0].text(0.5, 0.5, 'No total data available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('No Total Data', fontsize=14, fontweight='bold')

    # 4. 汇总统计信息
    ax_stats = axes[1, 1]
    ax_stats.axis('off')

    # 准备汇总统计信息
    stats_text = "Dataset Summary Statistics\n"
    stats_text += "=" * 40 + "\n\n"

    if len(input_lengths) > 0:
        stats_text += "Input Tokens:\n"
        stats_text += f"  Total: {sum(input_lengths):,}\n"
        stats_text += f"  Avg per request: {np.mean(input_lengths):.1f}\n"
        stats_text += f"  Median: {np.median(input_lengths):.1f}\n"
        stats_text += f"  90th %ile: {np.percentile(input_lengths, 90):.1f}\n"
        stats_text += f"  99th %ile: {np.percentile(input_lengths, 99):.1f}\n\n"

    if len(output_lengths) > 0:
        stats_text += "Output Tokens:\n"
        stats_text += f"  Total: {sum(output_lengths):,}\n"
        stats_text += f"  Avg per request: {np.mean(output_lengths):.1f}\n"
        stats_text += f"  Median: {np.median(output_lengths):.1f}\n"
        stats_text += f"  90th %ile: {np.percentile(output_lengths, 90):.1f}\n"
        stats_text += f"  99th %ile: {np.percentile(output_lengths, 99):.1f}\n\n"

    if len(total_lengths) > 0:
        stats_text += "Total Tokens (Input+Output):\n"
        stats_text += f"  Total: {sum(total_lengths):,}\n"
        stats_text += f"  Avg per request: {np.mean(total_lengths):.1f}\n"
        stats_text += f"  Median: {np.median(total_lengths):.1f}\n"
        stats_text += f"  90th %ile: {np.percentile(total_lengths, 90):.1f}\n"
        stats_text += f"  99th %ile: {np.percentile(total_lengths, 99):.1f}\n\n"

    stats_text += f"Bin Configuration:\n"
    stats_text += f"  Bin width: {bin_width} tokens\n"
    stats_text += f"  Total bins shown in each plot"

    ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                  verticalalignment='top', horizontalalignment='left',
                  fontsize=10, family='monospace',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # 添加整体标题
    normalize_text = " (Normalized)" if normalize else ""
    fig.suptitle(f'Token Length Distribution - Fixed {bin_width} Token Bins{normalize_text}',
                 fontsize=16, fontweight='bold', y=0.98)

    # 调整布局
    plt.tight_layout()

    # 打印详细分布信息
    print("\n" + "=" * 80)
    print(f"DETAILED DISTRIBUTION BY {bin_width}-TOKEN BINS")
    print("=" * 80)

    # 为每个数据集打印分布详情
    datasets = [
        ("Input", input_lengths),
        ("Output", output_lengths),
        ("Total", total_lengths)
    ]

    for name, data in datasets:
        if len(data) > 0:
            print(f"\n{name} Token Length Distribution:")
            print("-" * 60)

            # 计算固定宽度的桶
            max_val = max(data)
            if max_bin is not None:
                bin_edges = np.arange(0, max_bin + bin_width, bin_width)
            else:
                # 确保覆盖最大值
                last_bin = ((max_val // bin_width) + 1) * bin_width
                bin_edges = np.arange(0, last_bin + bin_width, bin_width)

            # 计算每个桶的统计
            hist, bin_edges = np.histogram(data, bins=bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # 打印每个桶的统计信息
            for i, (center, count) in enumerate(zip(bin_centers, hist)):
                if count > 0:
                    lower = bin_edges[i]
                    upper = bin_edges[i + 1]
                    percentage = (count / len(data)) * 100
                    print(f"  [{lower:5d}-{upper:5d}): {count:5d} requests ({percentage:5.1f}%)")

            # 打印超出最大桶的请求（如果有）
            if max_bin is not None and max(data) > max_bin:
                overflow_count = sum(1 for x in data if x >= max_bin)
                overflow_percentage = (overflow_count / len(data)) * 100
                print(f"  [{max_bin:5d}+     ): {overflow_count:5d} requests ({overflow_percentage:5.1f}%)")

    # 显示图形
    plt.show()


def _plot_single_fixed_bin(
        ax,
        data: List[int],
        bin_width: int,
        max_bin: Optional[int],
        show_cumulative: bool,
        normalize: bool,
        title: str,
        color: str,
        data_name: str
):
    """
    辅助函数：在单个轴上绘制固定宽度桶的直方图
    """
    # 创建固定宽度的桶
    max_val = max(data)

    if max_bin is not None:
        # 使用指定的最大桶
        bin_edges = np.arange(0, max_bin + bin_width, bin_width)
    else:
        # 自动计算，确保覆盖所有数据
        last_bin = ((max_val // bin_width) + 1) * bin_width
        bin_edges = np.arange(0, last_bin + bin_width, bin_width)

    # 计算直方图
    if normalize:
        weights = np.ones_like(data) / len(data) * 100  # 转换为百分比
        hist, bin_edges = np.histogram(data, bins=bin_edges, weights=weights)
        ylabel = "Percentage (%)"
    else:
        hist, bin_edges = np.histogram(data, bins=bin_edges)
        ylabel = "Number of Requests"

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 绘制柱状图
    bars = ax.bar(bin_centers, hist, width=bin_width * 0.8,
                  alpha=0.7, color=color, edgecolor='black', linewidth=0.5)

    # 在柱子上方显示数值
    for bar, value in zip(bars, hist):
        if value > 0:  # 只显示非零值
            height = bar.get_height()
            if normalize:
                text = f"{value:.1f}%"
            else:
                text = f"{int(value)}"
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    text, ha='center', va='bottom', fontsize=8, rotation=0)

    # 设置x轴标签
    ax.set_xlabel('Token Length', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # 设置标题
    bins_count = len(bin_edges) - 1
    if max_bin is not None:
        ax.set_title(f'{title}\n{bins_count} bins of {bin_width} tokens, up to {max_bin}',
                     fontsize=13, fontweight='bold')
    else:
        ax.set_title(f'{title}\n{bins_count} bins of {bin_width} tokens',
                     fontsize=13, fontweight='bold')

    # 设置x轴刻度
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([str(int(x)) for x in bin_edges], rotation=45, fontsize=9)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 添加累积分布线（在次坐标轴上）
    if show_cumulative:
        # 计算累积分布
        cumulative = np.cumsum(hist)
        if normalize:
            cumulative = cumulative / cumulative[-1] * 100
        else:
            cumulative = cumulative / len(data) * 100

        ax2 = ax.twinx()
        ax2.plot(bin_centers, cumulative, color='red', linewidth=2,
                 linestyle='--', marker='o', markersize=3, label='Cumulative %')
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=10, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim([0, 105])

        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # 添加统计信息
    stats_text = f"{data_name} Tokens\n"
    stats_text += f"Total: {len(data)} requests\n"
    stats_text += f"Min: {min(data):.0f}\n"
    stats_text += f"Max: {max(data):.0f}\n"
    stats_text += f"Mean: {np.mean(data):.1f}\n"
    stats_text += f"Median: {np.median(data):.1f}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
            fontsize=9)

if __name__ == '__main__':
    # python -m data.analysis.length_distribution
    config = ShareGPTRequestGeneratorConfig({
        "trace_file": "data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        "qps": 0.5,
    })

    unified_requests = ShareGPTRequestGenerator(config).generate_requests()[:1000]

    requests = prepare_data_v2(unified_requests)

    #stats_data = plot_cdf_distribution_v2(requests)
    states_data = plot_fixed_bin_histogram(requests)
