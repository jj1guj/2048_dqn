import gymnasium as gym
import imageio
import logging
import numpy as np
import random
import time
import torch

from n_network import N_Network
from replay_buffer import Experience
from priorized_replay_buffer import PrioritizedReplayBuffer
from n_step_buffer import NStepBuffer

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("train_log.txt", mode="w")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

env = __import__("fast2048_env").Fast2048Env(size=4, max_pow=16)

device = torch.device("cuda")

q_net = N_Network().to(device)
t_net = N_Network().to(device)
t_net.load_state_dict(q_net.state_dict())
t_net.eval()
best_weight = q_net.state_dict()

episodes = 50000

buffer_size = 1000000
batch_size = 128
replay_buffer = PrioritizedReplayBuffer(buffer_size, batch_size, 
                                        alpha=0.6, beta_start=0.4, 
                                        beta_end=1.0, beta_frames=episodes)

lr = 1e-4
optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3000, min_lr=1e-5)

gamma = 0.99
tau = 0.005  # ソフトターゲット更新率

n_step = 5
n_step_buffer = NStepBuffer(n_step, gamma)

def get_legal_actions(obs):
    """観測から合法手のリストを返す (4x4x16 one-hot → 4x4 board)"""
    board = obs.argmax(axis=-1)  # one-hot → log2値
    actions = []
    for action in range(4):
        if is_move_legal(board, action):
            actions.append(action)
    return actions if actions else [0]  # フォールバック

def is_move_legal(board, action):
    """指定方向に動かせるかチェック"""
    size = 4
    for i in range(size):
        line = []
        for j in range(size):
            if action == 0:   r, c = j, i        # UP
            elif action == 1: r, c = i, size-1-j  # RIGHT
            elif action == 2: r, c = size-1-j, i  # DOWN
            else:             r, c = i, j          # LEFT
            line.append(board[r][c])
        # 空きマスに向かって動ける or 隣接同値でマージ可能
        non_zero = [v for v in line if v != 0]
        if len(non_zero) < sum(1 for v in line if v != 0):
            return True  # 到達不能（ロジック的にここは来ない）
        # 空きがあって詰められる
        if non_zero != [line[j] for j in range(len(non_zero))]:
            return True
        # 隣接マージ可能
        for k in range(len(non_zero) - 1):
            if non_zero[k] == non_zero[k + 1]:
                return True
    return False

def board_potential(obs):
    """ボード状態のポテンシャル関数 Φ(s)。
    ポテンシャルベースシェーピング F = γ*Φ(s') - Φ(s) に使用。
    最適方策を変えずに配置改善を誘導できる（Ng et al., 1999）。
    """
    board = obs.argmax(axis=-1)  # (4,4) log2値
    max_val = board.max()

    # コーナーと対応するflip設定: (row, col) -> (flip_row, flip_col)
    corner_flip = {
        (0, 0): (False, False),  # 左上基点
        (0, 3): (False, True),   # 右上基点
        (3, 0): (True,  False),  # 左下基点
        (3, 3): (True,  True),   # 右下基点
    }

    # 最大タイルがいるコーナーを特定
    max_corner_flip = None
    for (r, c), flip in corner_flip.items():
        if board[r][c] == max_val:
            max_corner_flip = flip
            break

    # 最大タイルがコーナーにいない場合はボーナスなし
    if max_corner_flip is None:
        return 0.0

    # 1. コーナーボーナス
    corner_bonus = float(max_val) * 0.5

    # 2. 単調性: 最大タイルが実際にいるコーナーの方向だけで評価
    #    他コーナー基点のスコアは使わない → 無関係な行列の単調性維持を抑制
    def monotone_score(b, flip_row, flip_col):
        """flip_row/colでボードを反転して左上基点の単調性を計算"""
        g = b[::-1, :] if flip_row else b[:, :]
        g = g[:, ::-1] if flip_col else g[:, :]
        score = 0.0
        for i in range(4):
            for j in range(3):
                if g[i][j] >= g[i][j+1]:   # 行: 左→右が降順
                    score += 1
                if g[j][i] >= g[j+1][i]:   # 列: 上→下が降順
                    score += 1
        return score

    mono = monotone_score(board, *max_corner_flip)
    mono_bonus = mono * 0.1  # 最大 24*0.1 = 2.4

    return corner_bonus + mono_bonus


def now_policy(state):
    legal = get_legal_actions(state)
    s = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = q_net(s).squeeze()
        mask = torch.full((4,), float('-inf'), device=device)
        for a in legal:
            mask[a] = 0
        return (q + mask).argmax().item()
        

def tderror(states, actions, next_states, rewards, terminateds, weights):
    states, actions, next_states, rewards, terminateds, weights = (
        states.to(device), actions.to(device), next_states.to(device),
        rewards.to(device), terminateds.to(device), weights.to(device)
    )
    actions = actions.long().unsqueeze(1)
    q_values = q_net(states.float())
    q_value = q_values.gather(1, actions).squeeze()

    with torch.no_grad():
        # Double DQN: q_netでアクション選択、t_netで評価
        next_actions = q_net(next_states.float()).argmax(1, keepdim=True)
        next_q_value = t_net(next_states.float()).gather(1, next_actions).squeeze()
        q_value_target = rewards + (1 - terminateds) * (gamma ** n_step) * next_q_value

    td_errors = (q_value - q_value_target).detach().cpu().numpy()

    # IS重みで補正したHuber Loss
    element_loss = torch.nn.SmoothL1Loss(reduction='none')(q_value, q_value_target)
    loss = (weights * element_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()

    return td_errors

def soft_update_target():
    """ソフトターゲット更新 (Polyak averaging)"""
    for t_param, q_param in zip(t_net.parameters(), q_net.parameters()):
        t_param.data.copy_(tau * q_param.data + (1.0 - tau) * t_param.data)


def train():
    global best_weight
    max_reward = 0
    total_steps = 0
    has_trained = False
    for episode in range(episodes):
        # ノイズをリセット
        q_net.reset_noise()
        t_net.reset_noise()

        state, _ = env.reset()
        time_step = 0
        episode_over = False
        total_reward = 0
        max_tile = 0
        n_step_buffer.reset()

        while not episode_over:
            action = now_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            reward = np.log2(float(reward) + 1)
            episode_over = terminated or truncated

            if info["max"] > 0:
                current_tile = int(2 ** info["max"])
                if current_tile > max_tile:
                    max_tile = current_tile
                    reward += float(info["max"])

            # ポテンシャルベース報酬シェーピング: F = γ*Φ(s') - Φ(s)
            # 最適方策を変えずに配置改善を誘導する（Ng et al., 1999）
            reward += gamma * board_potential(next_state) - board_potential(state)

            # 違法手はバッファに入れない＆次のアクションを試す
            if not info["is_legal"]:
                continue

            # 1-step experienceをn_step_bufferに追加
            one_step = (state, action, reward, next_state, episode_over)
            n_step_buffer.add(Experience(*one_step))

            # n-stepバッファが溜まったらn-step遷移をReplayBufferに追加
            if n_step_buffer.is_ready():
                nstep_exp = n_step_buffer.get()
                replay_buffer.add(nstep_exp)

            state = next_state
            total_steps += 1
            if total_steps % 4 == 0 and len(replay_buffer) >= batch_size * 10:
                # 学習ステップでノイズをリセット
                q_net.reset_noise()
                states, actions, rewards, next_states, terminateds, indices, weights = replay_buffer.sampling()
                td_errors = tderror(states, actions, next_states, rewards, terminateds, weights)
                replay_buffer.update_priorities(indices, td_errors)
                # ソフトターゲット更新（学習ごとに少しずつ更新）
                soft_update_target()
                has_trained = True

            total_reward += float(reward)
            time_step += 1

        # エピソード終了時に残りすべてをflush
        for exp in n_step_buffer.flush():
            replay_buffer.add(exp)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Episode: {episode}, Total Reward: {total_reward:.1f}, Max Tile: {max_tile}, Steps: {time_step}, LR: {current_lr:.2e}')
        if has_trained:
            scheduler.step(total_reward)

        if total_reward > max_reward:
            max_reward = total_reward
            best_weight = q_net.state_dict()
            torch.save(best_weight, 'model.pth')


def main():
    train()
    torch.save(best_weight, 'model.pth')



if __name__ == "__main__":
    main()
