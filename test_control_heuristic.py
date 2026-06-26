import gymnasium as gym
import drone_dispatch_env
import numpy as np


def choose_action(state):
    dx = float(state[0])
    dy = float(state[1])
    heading = float(state[4])

    target_heading = np.arctan2(dy, dx)
    heading_error = np.arctan2(
        np.sin(target_heading - heading),
        np.cos(target_heading - heading)
    )

    speed = 1.0
    heading_delta = np.clip(heading_error / 0.35, -1.0, 1.0)

    return np.array([speed, heading_delta], dtype=np.float32)


env = gym.make("DroneControl-v0")

state, info = env.reset(seed=100)

total_reward = 0.0
terminated = False
truncated = False
step = 0

while not terminated and not truncated:
    action = choose_action(state)

    state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step += 1

    if step <= 10 or terminated or truncated:
        print(
            "Step:", step,
            "| Action:", action,
            "| Distance:", state[3],
            "| Battery:", state[2],
            "| Reward:", reward,
            "| Terminated:", terminated,
            "| Truncated:", truncated
        )

print("Total reward:", total_reward)
print("Final step:", step)
print("Final state:", state)