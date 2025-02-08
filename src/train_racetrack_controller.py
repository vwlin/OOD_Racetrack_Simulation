import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import os
import tensorboard
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_base', type=str, default='racetrack_ppo', help='Base directory to save trained model.')
    parser.add_argument('--n_other_vehicles', type=int, default=1, help='Number of additional vehicles besides ego vehicle.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate.')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs.')
    parser.add_argument('--timesteps', type=int, default=100000, help='Timesteps.')
    args = parser.parse_args()
    print(args)
    
    set_random_seed(args.seed)

    # env = make_vec_env("racetrack-v0")
    env = gym.make('racetrack-v0', render_mode='rgb_array')
    env.configure({
        "other_vehicles": args.n_other_vehicles,
    })
    env.reset()

    n_cpu = 6
    batch_size = 64
    model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=args.epochs,
            learning_rate=args.lr,
            gamma=0.9,
            verbose=2,
            seed=args.seed,
            tensorboard_log=os.path.join(args.save_base,f'other_vehicles_{str(args.n_other_vehicles)}', 'logs')
        )
    model.set_random_seed(args.seed)
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.save_base, f'other_vehicles_{str(args.n_other_vehicles)}', 'model'))