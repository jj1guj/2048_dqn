import gymnasium as gym
import imageio
import numpy as np
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

buffer_size = 10000
batch_size = 512
replay_buffer = ReplayBuffer(buffer_size, batch_size)

lr = 1e-3
optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

change_epsilon = 0.5

gamma = 0.99

episodes = 10000

td_interval = 30

def now_policy_train(state):
    global change_epsilon
    if np.random.rand() <= change_epsilon:
        change_epsilon = max(0.01, change_epsilon * 0.995)
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return q_net(state).argmax().item()
        
def now_policy(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return q_net(state).argmax().item()
        

def tderror(states, actions, next_states, rewards, terminateds):
    states, actions, next_states, rewards, terminateds = states.to(device), actions.to(device), next_states.to(device), rewards.to(device), terminateds.to(device)
    actions = actions.long().unsqueeze(1)
    q_values = q_net(states.float())
    q_value = q_values.gather(1, actions).squeeze()

    with torch.no_grad():
        next_q_max = t_net(next_states.float()).max(1)[0]
        # Q値のターゲット値をベルマン方程式に従って計算
        q_value_target = rewards + (1 - terminateds) * gamma * next_q_max

    # Q値から価値観数とのTD誤差を計算
    loss = torch.nn.MSELoss()(q_value, q_value_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    global best_weight
    for episode in range(episodes):
        state, _ = env.reset()
        time_step = 0
        episode_over = False
        noise = 0
        total_reward = 0
        max_reward = 0

        while not episode_over:
            action = now_policy_train(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_over = terminated or truncated
            experience = (state, action, reward, next_state, episode_over)

            replay_buffer.add(experience)

            state = next_state
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, terminateds = replay_buffer.sampling()
                tderror(states, actions, next_states, rewards, terminateds)

            # 一定間隔でターゲットネットワークの重みを更新
            if time_step % td_interval == 0:
                t_net.load_state_dict(q_net.state_dict())

            total_reward += float(reward)
            time_step += 1
            w = q_net.state_dict()

        print(f'Episode: {episode}, Total Reward: {total_reward}, Time Step: {time_step}')
        if total_reward > max_reward:
            max_reward = total_reward
            best_weight = q_net.state_dict()


def main():
    train()
    torch.save(best_weight, 'model.pth')



if __name__ == "__main__":
    main()
