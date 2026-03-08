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


def save_diagnostic_plot(out_dir, episodes, rewards, max_tiles, steps, bin_size, ma_window):
    """All-in-one 6-panel diagnostic: reward trend, ≥512 rate, reward σ/CV,
    avg episode length, cumulative ≥1024 count, early-vs-recent tile dist."""
    import matplotlib.gridspec as gridspec

    eps = episodes
    rew = rewards
    tile = max_tiles

    WIN = ma_window
    BIN = bin_size

    def ma(arr, w):
        cs = np.cumsum(arr.astype(float))
        cs[w:] = cs[w:] - cs[:-w]
        return cs[w - 1:] / w

    # binned stats
    bin_edges = np.arange(0, eps.max() + BIN, BIN)
    bc, p512, p1024, rew_mean, rew_std, steps_mean = [], [], [], [], [], []
    for lo in bin_edges[:-1]:
        hi = lo + BIN
        mask = (eps >= lo) & (eps < hi)
        if not mask.any():
            continue
        n = mask.sum()
        bc.append((lo + hi) / 2)
        p512.append((tile[mask] >= 512).sum() / n * 100)
        p1024.append((tile[mask] >= 1024).sum() / n * 100)
        rew_mean.append(rew[mask].mean())
        rew_std.append(rew[mask].std())
        steps_mean.append(steps[mask].mean())
    bc = np.array(bc)
    p512 = np.array(p512)
    p1024 = np.array(p1024)
    rew_mean = np.array(rew_mean)
    rew_std = np.array(rew_std)
    steps_mean = np.array(steps_mean)

    ep_count = eps[-1]
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"2048 DQN  —  ep 0-{ep_count}  ·  All-in-one diagnostic",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1) Total Reward MA
    eps_ma = eps[WIN - 1:]
    ax = fig.add_subplot(gs[0, :2])
    ax.scatter(eps, rew, s=0.2, alpha=0.15, color="steelblue", rasterized=True)
    ax.plot(eps_ma, ma(rew, WIN), color="crimson", lw=2, label=f"MA-{WIN}")
    z = np.polyfit(bc, rew_mean, 1)
    ax.plot(bc, np.polyval(z, bc), "k--", lw=1.2,
            label=f"trend ({z[0] * 1000:+.2f}/1kep)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("Total Reward"); ax.legend(fontsize=9)

    # 2) ≥512 tile rate bar
    ax2 = fig.add_subplot(gs[0, 2])
    z2 = np.polyfit(bc, p512, 1)
    ax2.bar(bc, p512, width=BIN * 0.8, color="royalblue", alpha=0.7, label="≥512%")
    ax2.plot(bc, np.polyval(z2, bc), "r--", lw=1.5,
             label=f"trend ({z2[0] * 1000:+.3f}%/1kep)")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("%")
    ax2.set_title(f"≥512 Tile Rate  ({BIN}-ep bins)")
    ax2.legend(fontsize=8)

    # 3) Reward mean ± σ  (proxy for NoisyNet exploration)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(bc, rew_mean, color="darkgreen", lw=2, label="mean reward")
    ax3.fill_between(bc, rew_mean - rew_std, rew_mean + rew_std,
                     alpha=0.25, color="green", label="±1σ")
    ax3_r = ax3.twinx()
    cv = rew_std / np.where(rew_mean == 0, 1, rew_mean) * 100
    ax3_r.plot(bc, cv, color="darkorange", lw=1.5, ls="--", label="CV%")
    ax3_r.set_ylabel("CV (%)", color="darkorange")
    ax3_r.tick_params(axis="y", labelcolor="darkorange")
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Reward")
    ax3.set_title("Reward Mean ± σ  (CV = σ/mean, proxy for NoisyNet exploration)")
    lines0, labels0 = ax3.get_legend_handles_labels()
    lines1, labels1 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines0 + lines1, labels0 + labels1, fontsize=8)

    # 4) Avg episode length
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(bc, steps_mean, color="purple", lw=2, label="avg steps")
    z4 = np.polyfit(bc, steps_mean, 1)
    ax4.plot(bc, np.polyval(z4, bc), "r--", lw=1.5,
             label=f"trend ({z4[0] * 1000:+.1f}/1kep)")
    ax4.set_xlabel("Episode"); ax4.set_ylabel("Steps")
    ax4.set_title("Avg Episode Length  (declining = worse policy)")
    ax4.legend(fontsize=8)

    # 5) Cumulative ≥1024 count
    ax5 = fig.add_subplot(gs[2, :2])
    cum1024 = np.cumsum(tile >= 1024)
    ax5.plot(eps, cum1024, color="firebrick", lw=2)
    total_1024 = int(cum1024[-1])
    ax5.set_xlabel("Episode"); ax5.set_ylabel("Cumulative ≥1024 tiles")
    ax5.set_title(f"Cumulative ≥1024 Tile Appearances  ({total_1024} total)")
    ax5.axhline(total_1024, color="gray", ls=":", lw=1)

    # 6) Max tile dist: early vs recent
    ax6 = fig.add_subplot(gs[2, 2])
    tile_vals = [32, 64, 128, 256, 512, 1024]
    colors6 = ["#d9f0a3", "#addd8e", "#78c679", "#31a354", "#006837", "#00441b"]
    early_cutoff = min(2500, int(ep_count * 0.1) + 1)
    recent_cutoff = max(ep_count - 2500, int(ep_count * 0.9))
    for label, mask, alpha in [
        (f"ep 0-{early_cutoff}", eps < early_cutoff, 1.0),
        (f"ep {recent_cutoff}-{ep_count}", eps >= recent_cutoff, 0.55),
    ]:
        n = mask.sum()
        if n == 0:
            continue
        pcts = [(tile[mask] == t).sum() / n * 100 for t in tile_vals]
        ax6.barh([str(t) for t in tile_vals], pcts, height=0.35,
                 alpha=alpha, label=label, color=colors6)
    ax6.set_xlabel("%"); ax6.set_title("Max Tile Dist: early vs recent")
    ax6.legend(fontsize=8)

    out = out_dir / "diagnostic_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
    save_diagnostic_plot(args.out_dir, episodes, rewards, max_tiles, steps, args.bin_size, args.ma_window)
    save_summary_json(args.out_dir, episodes, rewards, max_tiles, steps, lrs, binned)

    print(f"Saved plots to: {args.out_dir.resolve()}")
    print("- reward_trend.png")
    print("- max_tile_trend.png")
    print("- steps_and_lr.png")
    print("- tile_reach_rates.png")
    print("- binned_summary.png")
    print("- diagnostic_full.png")
    print("- summary.json")


if __name__ == "__main__":
    main()
