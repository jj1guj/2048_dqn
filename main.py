import gymnasium as gym
import imageio
import numpy as np
import random
import time
import torch

from n_network import N_Network
from replay_buffer import ReplayBuffer

env = gym.make("gymnasium_2048:gymnasium_2048/TwentyFortyEight-v0", size=4, max_pow=16)

device = torch.device("cuda")

q_net = N_Network().to(device)
t_net = N_Network().to(device)
t_net.load_state_dict(q_net.state_dict())
t_net.eval()
best_weight = q_net.state_dict()

buffer_size = 100000
batch_size = 128
replay_buffer = ReplayBuffer(buffer_size, batch_size)

lr = 1e-4
optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

start_epsilon = 1.0
change_epsilon = start_epsilon
epsilon_decay = 0.995
epsilon_min = 0.05
epsilon_reset_cycle = 3000

gamma = 0.99
tau = 0.005  # ソフトターゲット更新率

episodes = 10000

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

def now_policy_train(state):
    legal = get_legal_actions(state)
    if np.random.rand() <= change_epsilon:
        return random.choice(legal)
    else:
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q = q_net(s).squeeze()
            # 違法手を-infでマスク
            mask = torch.full((4,), float('-inf'), device=device)
            for a in legal:
                mask[a] = 0
            return (q + mask).argmax().item()

def now_policy(state):
    legal = get_legal_actions(state)
    s = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = q_net(s).squeeze()
        mask = torch.full((4,), float('-inf'), device=device)
        for a in legal:
            mask[a] = 0
        return (q + mask).argmax().item()
        

def tderror(states, actions, next_states, rewards, terminateds):
    states, actions, next_states, rewards, terminateds = states.to(device), actions.to(device), next_states.to(device), rewards.to(device), terminateds.to(device)
    actions = actions.long().unsqueeze(1)
    q_values = q_net(states.float())
    q_value = q_values.gather(1, actions).squeeze()

    with torch.no_grad():
        # Double DQN: q_netでアクション選択、t_netで評価
        next_actions = q_net(next_states.float()).argmax(1, keepdim=True)
        next_q_value = t_net(next_states.float()).gather(1, next_actions).squeeze()
        q_value_target = rewards + (1 - terminateds) * gamma * next_q_value

    # Huber Loss (SmoothL1) は外れ値に対してMSEより安定
    loss = torch.nn.SmoothL1Loss()(q_value, q_value_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()

def soft_update_target():
    """ソフトターゲット更新 (Polyak averaging)"""
    for t_param, q_param in zip(t_net.parameters(), q_net.parameters()):
        t_param.data.copy_(tau * q_param.data + (1.0 - tau) * t_param.data)


def train():
    global best_weight
    global change_epsilon
    global start_epsilon
    max_reward = 0
    total_steps = 0
    for episode in range(episodes):
        state, _ = env.reset()
        time_step = 0
        episode_over = False
        total_reward = 0
        max_tile = 0

        while not episode_over:
            action = now_policy_train(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            reward = np.log2(float(reward) + 1)
            episode_over = terminated or truncated

            if info["max"] > 0:
                current_tile = int(2 ** info["max"])
                if current_tile > max_tile:
                    max_tile = current_tile
                    reward += float(info["max"])

            experience = (state, action, reward, next_state, episode_over)

            # 違法手はバッファに入れない＆次のアクションを試す
            if not info["is_legal"]:
                continue

            replay_buffer.add(experience)

            state = next_state
            total_steps += 1
            if total_steps % 4 == 0 and len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, terminateds = replay_buffer.sampling()
                tderror(states, actions, next_states, rewards, terminateds)
                # ソフトターゲット更新（学習ごとに少しずつ更新）
                soft_update_target()

            total_reward += float(reward)
            time_step += 1

        print(f'Episode: {episode}, Total Reward: {total_reward:.1f}, Max Tile: {max_tile}, Steps: {time_step}, Epsilon: {change_epsilon:.3f}')
        change_epsilon = max(epsilon_min, change_epsilon * epsilon_decay)
        if (episode + 1) % epsilon_reset_cycle == 0:
            start_epsilon /= 2
            change_epsilon = start_epsilon

        if total_reward > max_reward:
            max_reward = total_reward
            best_weight = q_net.state_dict()


def main():
    train()
    torch.save(best_weight, 'model.pth')



if __name__ == "__main__":
    main()
