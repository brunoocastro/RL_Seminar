import gymnasium as gym 

def run(
        render = True,
        map_name = "8x8",
        is_slippery = True,
):
    # slippery means that the agent may not always move in the intended direction - it might slip to the side
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, map_name=map_name, render_mode="human" if render else None)

    print("Environment info:")
    print(f"- Action space: {env.action_space}")
    print(f"- Observation space: {env.observation_space}")
    print(f"- Spec: {env.spec.to_json()}")

    reseted_env = env.reset()
    print(f"\nInitial State: {reseted_env}")
    state = reseted_env[0] # States: 0 to 63 for 8x8 grid (total 64 states, 0 is start on top left and 63 is goal on bottom right)
    print(f"\nTable state: {state}")

    terminated = False # True when fall in hole or reach goal
    truncated = False # True when reach max steps (100 by default)

    while not (terminated or truncated):
        action = env.action_space.sample()  # Random action
        print(f"\nTaking action: {action}") # Actions: 0=Left, 1=Down, 2=Right, 3=Up

        new_state, reward, terminated, truncated, info = env.step(action)
        print(f"New State: {new_state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

        env.render()

        state = new_state # Update state to new state for next iteration

    env.close()

if __name__ == "__main__":
    run()