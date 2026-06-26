import gymnasium as gym
import drone_dispatch_env
import numpy as np

env = gym.make("DroneControl-v0")

obs, info = env.reset(seed=0)

print("Observation:", obs)
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

total_reward = 0.0
terminated = False
truncated = False
steps = 0

while not terminated and not truncated:
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

print("Episode finished.")
print("Steps:", steps)
print("Total reward:", total_reward)