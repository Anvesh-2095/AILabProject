import gymnasium as gym
import numpy as np
import argparse
import os
from multiprocessing import Pool

def policy_action(params, observation):
    # The policy is a linear mapping from the 8-dimensional observation to 4 action scores.
    W = params[:8 * 4].reshape(8, 4)
    b = params[8 * 4:].reshape(4)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)

def evaluate_policy(policy, policy_action, total_episodes=100):
    total_reward = 0.0
    for _ in range(total_episodes):
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy_action(policy, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += episode_reward
    return total_reward / total_episodes

def evaluate_policy_single(params):
    return evaluate_policy(params, policy_action)

def pso(num_particles, num_iterations, w, c1, c2, load=False, filename="best_policy.npy"):
    num_params = 8 * 4 + 4

    best_avg = -np.inf
    no_improvement_count = 0  # Counter for iterations without improvement

    if load and os.path.exists("sav_best_avg.npz"):
        print("Loading saved data...")
        data = np.load("sav_best_avg.npz", allow_pickle=True)
        particles = data['particles']
        velocities = data['velocities']
        personal_best_params = data['personal_best_params']
        personal_best_rewards = data['personal_best_rewards']
        best_params = data['best_params']
        best_reward = data['best_reward']
        best_avg = data['best_avg']
        print(f"Loaded best reward: {best_reward:.2f}")
    else:
        particles = np.random.randn(num_particles, num_params) * 0.1
        velocities = np.zeros((num_particles, num_params))
        personal_best_params = particles.copy()
        personal_best_rewards = np.full(num_particles, -np.inf)
        best_params, best_reward = None, -np.inf

    print("Training the agent using PSO...")
    with Pool() as pool:  # Create a multiprocessing pool
        for iteration in range(num_iterations):

            if no_improvement_count >= 10:  # Trigger partial random restart
                print("No improvement for 10 iterations. Performing partial random restart...")
                num_reset_particles = int(0.3 * num_particles)  # 30% of particles
                reset_indices = np.random.choice(num_particles, num_reset_particles, replace=False)
                particles[reset_indices] = np.random.rand(num_reset_particles, num_params)
                velocities[reset_indices] = np.zeros((num_reset_particles, num_params))
                personal_best_params[reset_indices] = particles[reset_indices].copy()
                personal_best_rewards[reset_indices] = -np.inf
                no_improvement_count = 0  # Reset counter
                
            # Evaluate all particles in parallel
            rewards = pool.map(evaluate_policy_single, particles)
            total_reward = sum(rewards)
            avg_reward = total_reward / num_particles
            if avg_reward > best_avg:
                best_avg = avg_reward
                no_improvement_count = 0  # Reset counter on improvement
                print(f"Best average reward: {best_avg:.2f}")
                np.savez("sav_best_avg.npz", particles=particles, velocities=velocities, personal_best_params=personal_best_params, personal_best_rewards=personal_best_rewards, best_params=best_params, best_reward=best_reward, best_avg = best_avg)
                print(f"Saved best reward: {best_reward:.2f}")
                np.save(filename, best_params)
                print(f"Saved best policy to {'best_avg_policy.npy'}")
            else:
                no_improvement_count += 1

            for i in range(num_particles):
                reward = rewards[i]
                if reward > personal_best_rewards[i]:
                    personal_best_rewards[i] = reward
                    personal_best_params[i] = particles[i].copy()
                    if reward > best_reward:
                        best_reward = reward
                        best_params = particles[i].copy()
                # Update the velocity
                inertia = max(0.2, w - (iteration / num_iterations) * (w - 0.2))
                velocities[i] = (
                    inertia * velocities[i]
                    + c1 * np.random.rand() * (personal_best_params[i] - particles[i])
                    + c2 * np.random.rand() * (best_params - particles[i])
                )
                velocities[i] = np.clip(velocities[i], -1, 1)
                particles[i] += velocities[i]
                
                particles[i] = np.clip(particles[i], -5, 5)

            print(f"Iteration {iteration + 1}/{num_iterations}, best reward: {best_reward:.2f}")
    return best_params

def train_and_save(filename, num_particles = 100, num_iterations = 1000, c1 = 2.8, c2 = 2.5, w = 1.2, load = False):
    best_params = pso(num_particles, num_iterations, w, c1, c2, load, filename)
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