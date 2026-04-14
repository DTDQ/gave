#!/usr/bin/env python3
"""
plot_loss.py - 从训练日志中解析 loss 并绘制下降曲线
用法: python plot_loss.py --log <log_file> [--output <output_path>]
"""

import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path):
    """从日志文件中解析训练 loss 和验证 loss"""
    train_pattern = re.compile(
        r'\[Step\s+(\d+)/\d+\]\s+(?:训练|初始)损失:\s+([\d.]+)\s+'
        r'\(L1:\s+([\d.]+),\s+L2:\s+([\d.]+),\s+L3:\s+([\d.]+),\s+L4_cpa:\s+([\d.]+)\)'
    )
    val_pattern = re.compile(
        r'\[Step\s+(\d+)\]\s+验证损失:\s+([\d.]+)'
    )

    train_steps, train_total, train_l1, train_l2, train_l3, train_l4 = [], [], [], [], [], []
    val_steps, val_loss = [], []

    with open(log_path, 'r') as f:
        for line in f:
            m = train_pattern.search(line)
            if m:
                train_steps.append(int(m.group(1)))
                train_total.append(float(m.group(2)))
                train_l1.append(float(m.group(3)))
                train_l2.append(float(m.group(4)))
                train_l3.append(float(m.group(5)))
                train_l4.append(float(m.group(6)))
                continue
            m = val_pattern.search(line)
            if m:
                val_steps.append(int(m.group(1)))
                val_loss.append(float(m.group(2)))

    return {
        'train_steps': train_steps, 'train_total': train_total,
        'train_l1': train_l1, 'train_l2': train_l2,
        'train_l3': train_l3, 'train_l4': train_l4,
        'val_steps': val_steps, 'val_loss': val_loss,
    }


def plot_loss(data, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')

    # 1. Total loss + val loss
    ax = axes[0, 0]
    ax.plot(data['train_steps'], data['train_total'], label='Train Total', alpha=0.7, linewidth=0.8)
    if data['val_steps']:
        ax.plot(data['val_steps'], data['val_loss'], 'ro-', label='Val Loss', markersize=3, linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. L1 (action loss)
    ax = axes[0, 1]
    ax.plot(data['train_steps'], data['train_l1'], color='tab:orange', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('L1 (Action Loss)')
    ax.grid(True, alpha=0.3)

    # 3. L4 (CPA penalty)
    ax = axes[1, 0]
    ax.plot(data['train_steps'], data['train_l4'], color='tab:red', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('L4 (CPA Penalty)')
    ax.grid(True, alpha=0.3)

    # 4. L2 + L3
    ax = axes[1, 1]
    ax.plot(data['train_steps'], data['train_l2'], label='L2 (Score)', alpha=0.7, linewidth=0.8)
    ax.plot(data['train_steps'], data['train_l3'], label='L3 (Weight)', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('L2 (Score) & L3 (Weight)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Loss 曲线已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制训练 loss 下降曲线')
    parser.add_argument('--log', type=str, required=True, help='训练日志文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出图片路径（默认与日志同目录）')
    args = parser.parse_args()

    if args.output is None:
        import os
        log_dir = os.path.dirname(args.log) or '.'
        args.output = os.path.join(log_dir, 'loss_curve.png')

    data = parse_log(args.log)
    print(f"解析到 {len(data['train_steps'])} 个训练点, {len(data['val_steps'])} 个验证点")

    if not data['train_steps']:
        print("错误：日志中未找到任何 loss 数据")
        return

    plot_loss(data, args.output)


if __name__ == '__main__':
    main()
