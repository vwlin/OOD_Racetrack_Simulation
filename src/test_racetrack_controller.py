import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
sys.path.append("../")
from HighwayEnv.scripts.utils import record_videos

'''
plot_positions - save figure of vehicle trajectories
parameters:
    positions - dictionary of positions
    title - title of figure
    save_pth - path to save figure to
'''
def plot_positions(positions, title, save_pth):
    for i, v in enumerate(positions.keys()):
        positions[v] = np.stack(positions[v])
        l = 'other vehicle'
        if i == 0: l = 'ego vehicle'
        plt.plot(positions[v][:,0], -1*positions[v][:,1], label=l)
    plt.legend()
    plt.title(title)
    plt.savefig(save_pth)
    plt.close()

'''
run_test - test trained model
parameters:
    args - command line args in argparse object
    ood_mode - string defining ID/OOD mode
'''
def run_test(args, ood_mode, n_horizon=12):
    if args.test == 'visuals':
        n_episodes = 10
    elif args.test == 'samples':
        n_episodes = 100
    else:
        raise NotImplementedError

    if ood_mode == 'noisy':
        severities = [0.1, 0.2, 0.3, 0.4]
    elif ood_mode == 'vehicles':
        severities = [args.n_other_vehicles+i for i in range(1,5)] # up to this many extra vehicles
    elif ood_mode == 'id':
        severities = [None]
    else:
        raise NotImplementedError
    print(f'\nRunning test for OOD mode {ood_mode}')

    for s in severities:
        prefix = f'{ood_mode}_'
        if s is not None: # not ID
            print(f'Severity {str(s)}...')
            prefix = f'{prefix}{str(s)}_'

        # set up save locations
        if args.test == 'visuals':
            vid_loc = os.path.join(args.save_base, f'other_vehicles_{str(args.n_other_vehicles)}/{prefix}videos')
            fig_loc = os.path.join(args.save_base, f'other_vehicles_{str(args.n_other_vehicles)}/{prefix}figs')
            os.makedirs(fig_loc, exist_ok=True)
        else:
            sample_loc = os.path.join(args.save_base, f'other_vehicles_{str(args.n_other_vehicles)}/samples')
            os.makedirs(sample_loc, exist_ok=True)
        
        # setup environment
        env = gym.make('racetrack-v0', render_mode='rgb_array')
        if ood_mode == 'vehicles':
            env.unwrapped.configure({"other_vehicles": s})
        else:
            env.unwrapped.configure({"other_vehicles": args.n_other_vehicles})

        if args.test == 'visuals':
            env = record_videos(env, vid_loc)

        # start testing loop
        sample_data = {} # episode -> timestep -> ID -> [x, y, heading]
        ave_steps = 0
        for episode in range(n_episodes):
            positions = {}
            episode_data = {}
            n_steps, steps_after_crash = 0, 0
            done = truncated = crashed = False
            obs, info = env.reset(seed=episode)

            # get noisy observation if necessary
            if ood_mode == 'noisy':
                noisy_obs = env.unwrapped.observation_type.observe(noisy=True, b=s)

            # save initial positions to dict
            episode_data['ego_id'] = str(id(env.unwrapped.observation_type.observer_vehicle))
            episode_data[n_steps] = {}
            for v in env.unwrapped.road.vehicles:
                d = v.to_dict()
                positions[id(v)] = [[d['x'], d['y']]]
                episode_data[n_steps][id(v)] = [d['x'], d['y'], d['heading']]
                episode_data[n_steps]['crashed'] = crashed

            while not (done or truncated):
                # choose and take action
                if ood_mode == 'noisy':
                    action, _ = model.predict(noisy_obs, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

                # get noisy observation if necessary
                if ood_mode == 'noisy':
                    noisy_obs = env.unwrapped.observation_type.observe(noisy=True, b=s)

                # update info about number of steps and whether crash has occurred, extending episode until a crash has occurred n_horizon in the past
                n_steps += 1
                crashed = crashed or env.unwrapped.observation_type.observer_vehicle.crashed
                if crashed:
                    steps_after_crash += 1
                    if steps_after_crash < n_horizon:
                        truncated = done = False

                # save positions to dict
                episode_data[n_steps] = {}
                for v in env.unwrapped.road.vehicles:
                    d = v.to_dict()
                    if not crashed:
                        positions[id(v)].append([d['x'], d['y']])
                    episode_data[n_steps][id(v)] = [d['x'], d['y'], d['heading']]
                    episode_data[n_steps]['crashed'] = crashed
            env.close()
            sample_data[episode] = episode_data
            ave_steps += n_steps
            if episode % 10 == 0:
                print(f'Episode {str(episode)}: average episode length {str(ave_steps/(episode+1))}')

            if args.test == 'visuals':
                plot_positions(positions, f'{str(n_steps)}-step episode', os.path.join(fig_loc, f'{str(episode)}.png'))

        print(f'Over {str(n_episodes)} episodes: average episode length {str(ave_steps/(n_episodes))}')

        # save sample data
        if args.test == 'samples':
            name = ood_mode
            if ood_mode != 'id':
                name = f'{name}_{str(s)}'
            with open(os.path.join(sample_loc, f'{name}.json'), 'w') as f:
                json.dump(sample_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_base', type=str, default='racetrack_ppo', help='Base directory of saved trained model.')
    parser.add_argument('--n_other_vehicles', type=int, default=1, help='Number of additional vehicles besides ego vehicle in in-distribution setting.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--test', type=str, choices=['visuals', 'samples'], default='visuals', help='Save visuals (video and figure) or samples.')
    args = parser.parse_args()
    print(args)
    
    set_random_seed(args.seed)

    # load model
    model_loc = os.path.join(args.save_base, f'other_vehicles_{str(args.n_other_vehicles)}/model')
    model = PPO.load(model_loc)

    # run tests
    run_test(args, 'id')
    run_test(args, 'vehicles')
    run_test(args, 'noisy')