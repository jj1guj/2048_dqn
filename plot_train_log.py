import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOG_PATTERN = re.compile(
    r"Episode:\s*(\d+),\s*"
    r"Total Reward:\s*([-+]?\d*\.?\d+),\s*"
    r"Max Tile:\s*(\d+),\s*"
    r"Steps:\s*(\d+),\s*"
    r"LR:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_log(log_path: Path):
    episodes, rewards, max_tiles, steps, lrs = [], [], [], [], []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            episodes.append(int(m.group(1)))
            rewards.append(float(m.group(2)))
            max_tiles.append(int(m.group(3)))
            steps.append(int(m.group(4)))
            lrs.append(float(m.group(5)))

    if not episodes:
        raise ValueError("No valid training log lines were found.")

    return (
        np.array(episodes, dtype=np.int32),
        np.array(rewards, dtype=np.float32),
        np.array(max_tiles, dtype=np.int32),
        np.array(steps, dtype=np.int32),
        np.array(lrs, dtype=np.float32),
    )


def rolling_mean(values: np.ndarray, window: int):
    if window <= 1 or len(values) < window:
        return np.arange(len(values)), values
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    x = np.arange(window - 1, len(values))
    return x, smoothed


def rolling_rate(mask: np.ndarray, window: int):
    return rolling_mean(mask.astype(np.float32), window)


def binned_stats(episodes, rewards, max_tiles, bin_size: int):
    x_centers, avg_rewards = [], []
    p256, p512, p1024 = [], [], []

    n = len(episodes)
    for start in range(0, n, bin_size):
        end = min(start + bin_size, n)
        if end - start <= 0:
            continue
        r = rewards[start:end]
        t = max_tiles[start:end]
        e = episodes[start:end]

        x_centers.append(float(e.mean()))
        avg_rewards.append(float(r.mean()))
        p256.append(float((t >= 256).mean() * 100.0))
        p512.append(float((t >= 512).mean() * 100.0))
        p1024.append(float((t >= 1024).mean() * 100.0))

    return {
        "x": np.array(x_centers, dtype=np.float32),
        "avg_reward": np.array(avg_rewards, dtype=np.float32),
        "p256": np.array(p256, dtype=np.float32),
        "p512": np.array(p512, dtype=np.float32),
        "p1024": np.array(p1024, dtype=np.float32),
    }


def save_reward_plot(out_dir, episodes, rewards, ma_window):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, rewards, alpha=0.25, linewidth=1.0, label="Reward (raw)")

    mx, my = rolling_mean(rewards, ma_window)
    ax.plot(episodes[mx], my, linewidth=2.2, label=f"Reward MA({ma_window})")

    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reward_trend.png", dpi=150)
    plt.close(fig)


def save_max_tile_plot(out_dir, episodes, max_tiles, ma_window):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(episodes, max_tiles, alpha=0.35, linewidth=1.0, label="Max Tile (raw)")

    mx, my = rolling_mean(max_tiles.astype(np.float32), ma_window)
    ax.plot(episodes[mx], my, linewidth=2.2, label=f"Max Tile MA({ma_window})")

    ax.set_title("Max Tile per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Max Tile")
    ax.set_yticks([32, 64, 128, 256, 512, 1024, 2048])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "max_tile_trend.png", dpi=150)
    plt.close(fig)


def save_steps_lr_plot(out_dir, episodes, steps, lrs, ma_window):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(episodes, steps, alpha=0.25, linewidth=1.0, label="Steps (raw)")
    sx, sy = rolling_mean(steps.astype(np.float32), ma_window)
    ax1.plot(episodes[sx], sy, linewidth=2.0, label=f"Steps MA({ma_window})")
    ax1.set_title("Episode Length")
    ax1.set_ylabel("Steps")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(episodes, lrs, linewidth=2.0, color="tab:orange", label="Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("LR")
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "steps_and_lr.png", dpi=150)
    plt.close(fig)


def save_reach_rate_plot(out_dir, episodes, max_tiles, ma_window):
    fig, ax = plt.subplots(figsize=(12, 5))

    for threshold, color in [(256, "tab:green"), (512, "tab:blue"), (1024, "tab:red")]:
        mask = max_tiles >= threshold
        x, y = rolling_rate(mask, ma_window)
        ax.plot(episodes[x], y * 100.0, linewidth=2.0, color=color, label=f">= {threshold} (rolling %)")

    ax.set_title(f"Tile Reach Rates (Rolling Window = {ma_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tile_reach_rates.png", dpi=150)
    plt.close(fig)


def save_binned_summary_plot(out_dir, binned):
    x = binned["x"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(x, binned["avg_reward"], marker="o", linewidth=2.0)
    ax1.set_title("Binned Average Reward")
    ax1.set_ylabel("Avg Reward")
    ax1.grid(alpha=0.25)

    ax2.plot(x, binned["p256"], marker="o", linewidth=2.0, label=">=256")
    ax2.plot(x, binned["p512"], marker="o", linewidth=2.0, label=">=512")
    ax2.plot(x, binned["p1024"], marker="o", linewidth=2.0, label=">=1024")
    ax2.set_title("Binned Tile Reach Rates")
    ax2.set_xlabel("Episode (bin center)")
    ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "binned_summary.png", dpi=150)
    plt.close(fig)


def save_summary_json(out_dir, episodes, rewards, max_tiles, steps, lrs, binned):
    summary = {
        "episodes": int(len(episodes)),
        "episode_min": int(episodes.min()),
        "episode_max": int(episodes.max()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "max_tile_counts": {
            "32_or_less": int(np.sum(max_tiles <= 32)),
            "64": int(np.sum(max_tiles == 64)),
            "128": int(np.sum(max_tiles == 128)),
            "256": int(np.sum(max_tiles == 256)),
            "512": int(np.sum(max_tiles == 512)),
            "1024_or_more": int(np.sum(max_tiles >= 1024)),
        },
        "steps_mean": float(steps.mean()),
        "lr_start": float(lrs[0]),
        "lr_end": float(lrs[-1]),
        "last_bin": {
            "avg_reward": float(binned["avg_reward"][-1]) if len(binned["avg_reward"]) else None,
            "p256": float(binned["p256"][-1]) if len(binned["p256"]) else None,
            "p512": float(binned["p512"][-1]) if len(binned["p512"]) else None,
            "p1024": float(binned["p1024"][-1]) if len(binned["p1024"]) else None,
        },
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Plot training analysis graphs from train_log.txt")
    parser.add_argument("--log-path", type=Path, default=Path("train_log.txt"), help="Path to train log")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_plots"), help="Output directory for images")
    parser.add_argument("--ma-window", type=int, default=200, help="Moving average window")
    parser.add_argument("--bin-size", type=int, default=500, help="Episode bin size for summary plots")
    args = parser.parse_args()

    episodes, rewards, max_tiles, steps, lrs = parse_log(args.log_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    save_reward_plot(args.out_dir, episodes, rewards, args.ma_window)
    save_max_tile_plot(args.out_dir, episodes, max_tiles, args.ma_window)
    save_steps_lr_plot(args.out_dir, episodes, steps, lrs, args.ma_window)
    save_reach_rate_plot(args.out_dir, episodes, max_tiles, args.ma_window)

    binned = binned_stats(episodes, rewards, max_tiles, args.bin_size)
    save_binned_summary_plot(args.out_dir, binned)
    save_summary_json(args.out_dir, episodes, rewards, max_tiles, steps, lrs, binned)

    print(f"Saved plots to: {args.out_dir.resolve()}")
    print("- reward_trend.png")
    print("- max_tile_trend.png")
    print("- steps_and_lr.png")
    print("- tile_reach_rates.png")
    print("- binned_summary.png")
    print("- summary.json")


if __name__ == "__main__":
    main()
