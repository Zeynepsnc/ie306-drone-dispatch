import gymnasium as gym
import drone_dispatch_env
import torch

from ddpg_model import Actor, Critic


env = gym.make("DroneControl-v0")
obs, info = env.reset(seed=0)

state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

actor = Actor()
critic = Critic()

action = actor(state)
q_value = critic(state, action)

print("State shape:", state.shape)
print("Action shape:", action.shape)
print("Action:", action.detach().numpy())
print("Q-value shape:", q_value.shape)
print("Q-value:", q_value.item())