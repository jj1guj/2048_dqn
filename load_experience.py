"""
保存されたexperience(プレイデータ)を読み込むユーティリティ

使用例:
    from load_experience import load_all_experiences, load_experience
    # 1つのファイルを読み込む
    data = load_experience("experiences/episode_0001_20260227_120000_score1234_max256.pkl")
    # 全エピソードを読み込んでDQN学習用のバッチに変換
    states, actions, rewards, next_states, dones = load_all_experiences()
"""

import glob
import os
import pickle

import numpy as np

EXPERIENCE_DIR = "experiences"


def load_experience(filepath: str) -> dict:
    """1つのエピソードファイルを読み込む

    Returns:
        dict with keys:
            - episode: エピソード番号
            - timestamp: 保存時刻
            - total_score: 合計スコア
            - max_tile: 最大タイル値
            - num_steps: ステップ数
            - transitions: list of dict (state, action, reward, next_state, terminated, info)
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_all_experiences(experience_dir: str = EXPERIENCE_DIR) -> tuple:
    """全エピソードのexperienceを読み込み、DQN学習用のnumpy配列として返す

    Returns:
        (states, actions, rewards, next_states, dones) のタプル
        - states: np.ndarray shape (N, 4, 4, 16)
        - actions: np.ndarray shape (N,)
        - rewards: np.ndarray shape (N,)
        - next_states: np.ndarray shape (N, 4, 4, 16)
        - dones: np.ndarray shape (N,)
    """
    files = sorted(glob.glob(os.path.join(experience_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"{experience_dir}/ にexperienceファイルが見つかりません")

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []

    for filepath in files:
        data = load_experience(filepath)
        for t in data["transitions"]:
            all_states.append(t["state"])
            all_actions.append(t["action"])
            all_rewards.append(t["reward"])
            all_next_states.append(t["next_state"])
            all_dones.append(t["terminated"])

    print(f"読み込み: {len(files)}エピソード, {len(all_states)}トランジション")

    return (
        np.array(all_states),
        np.array(all_actions),
        np.array(all_rewards, dtype=np.float32),
        np.array(all_next_states),
        np.array(all_dones, dtype=np.bool_),
    )


if __name__ == "__main__":
    # 保存済みexperienceの一覧と統計を表示
    files = sorted(glob.glob(os.path.join(EXPERIENCE_DIR, "*.pkl")))
    if not files:
        print(f"{EXPERIENCE_DIR}/ にファイルがありません。先にplay_browser.pyでプレイしてください。")
    else:
        total_transitions = 0
        for f in files:
            data = load_experience(f)
            total_transitions += data["num_steps"]
            print(
                f"  {os.path.basename(f)}: "
                f"{data['num_steps']}ステップ, "
                f"スコア:{data['total_score']}, "
                f"最大タイル:{data['max_tile']}"
            )
        print(f"\n合計: {len(files)}エピソード, {total_transitions}トランジション")
