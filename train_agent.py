import gymnasium as gym
import numpy as np
import argparse
import os

def policy_action(params, observation):
    # The policy is a linear mapping from the 8-dimensional observation to 4 action scores.
    W = params[:8 * 4].reshape(8, 4)
    b = params[8 * 4:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)

def evaluate_policy(params, episodes=3, render=False):
    total_reward = 0.0
    for _ in range(episodes):
        env = gym.make('LunarLander-v3', render_mode='human' if render else 'rgb_array')
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += episode_reward
    return total_reward / episodes

def pso(num_particles, num_iterations, w, c1, c2, load=False):
    best_params = None  
    best_reward = -np.inf
    num_params = 8 * 4 + 4
    particles = np.random.rand(num_particles, num_params)
    velocities = np.zeros((num_particles, num_params))
    personal_best_params = particles.copy()
    personal_best_rewards = np.full(num_particles, -np.inf)

    if load and os.path.exists("sav.npz"):
        print("Loading saved data...")
        data = np.load("sav.npz", allow_pickle=True)
        particles = data['particles']
        velocities = data['velocities']
        personal_best_params = data['personal_best_params']
        personal_best_rewards = data['personal_best_rewards']
        best_params = data['best_params']
        best_reward = data['best_reward']
        print(f"Loaded best reward: {best_reward:.2f}")

    print("Training the agent using PSO...")
    for i in range(num_particles):
        personal_best_rewards[i] = evaluate_policy(particles[i])
        if personal_best_rewards[i] > best_reward:
            best_reward = personal_best_rewards[i]
            best_params = particles[i].copy()
    print(f"Initial best reward: {best_reward:.2f}")
    for _ in range(num_iterations):
        max_reward = -np.inf
        for i in range(num_particles):
            # Update the velocity
            velocities[i] = w * velocities[i] + c1 * np.random.rand() * (personal_best_params[i] - particles[i]) + c2 * np.random.rand() * (best_params - particles[i])
            particles[i] += velocities[i]
            # Evaluate the new particle
            reward = evaluate_policy(particles[i])
            if i == 0:
                max_reward = reward
            max_reward = max(max_reward, reward)
            if reward > personal_best_rewards[i]:
                personal_best_rewards[i] = reward
                personal_best_params[i] = particles[i].copy()
                if reward > best_reward:
                    best_reward = reward
                    best_params = particles[i].copy()
                    np.savez("sav.npz", particles=particles, velocities=velocities, personal_best_params=personal_best_params, personal_best_rewards=personal_best_rewards, best_params=best_params, best_reward=best_reward)
                    print(f"Saved best reward: {best_reward:.2f}")
        print(f"Iteration {_ + 1}/{num_iterations}, best reward: {best_reward:.2f}")
        print(f"Current reward: {max_reward:.2f}")
    return best_params    

def train_and_save(filename, num_particles = 100, num_iterations = 1000, c1 = 2.0, c2 = 2.0, w = 0.7, load = False):
    best_params = pso(num_particles, num_iterations, w, c1, c2, load)
    np.save(filename, best_params)
    print(f"Saved best policy to {filename}")
    return best_params


def load_policy(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    best_params = np.load(filename)
    print(f"Loaded best policy from {filename}")
    return best_params

def play_policy(best_params, episodes=5):
    test_reward = evaluate_policy(best_params, episodes=episodes, render=True)
    print(f"Average reward of the best policy over {episodes} episodes: {test_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using GA with SBX and polynomial mutation.")
    parser.add_argument("--train", action="store_true", help="Train the policy using GA and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy.npy", help="Filename to save/load the best policy.")
    parser.add_argument("--num_particles", type=int, default=100, help="Number of particles in the PSO algorithm.")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations in the PSO algorithm.")
    parser.add_argument("--c1", type=float, default=2.0, help="C1 parameter in the PSO algorithm.")
    parser.add_argument("--c2", type=float, default=2.0, help="C2 parameter in the PSO algorithm.")
    parser.add_argument("--w", type=float, default=0.7, help="W parameter in the PSO algorithm.")
    parser.add_argument("--load", action="store_true", help="Load the best policy.")
    args = parser.parse_args()

    if args.train:
        # Train and save the best policy
        best_params = train_and_save(filename=args.filename, num_particles=args.num_particles, num_iterations=args.num_iterations, c1=args.c1, c2=args.c2, w=args.w, load = args.load)
    elif args.play:
        # Load and play with the best policy
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=5)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")