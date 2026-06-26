import gymnasium as gym
import drone_dispatch_env
import numpy as np


env = gym.make("DroneControl-v0")

state, info = env.reset(seed=100)

print("Initial state:", state)

total_reward = 0.0
terminated = False
truncated = False
step = 0

while not terminated and not truncated:
    action = np.array([0.5, 0.0], dtype=np.float32)

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step += 1

    if step <= 10 or terminated or truncated:
        print(
            "Step:", step,
            "| State:", state,
            "| Reward:", reward,
            "| Terminated:", terminated,
            "| Truncated:", truncated,
            "| Info:", info
        )

print("Total reward:", total_reward)
print("Final step:", step)
print("Final state:", state)