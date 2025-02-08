# OOD Racetrack Simulation
This is a repository for generating simulations in the HighwayEnv racetrack environment. Simulations can be generated in in-distribution or out-of-distribution environments.

This code was used to generate simulation data for the publication *Safety Monitoring for Learning-Enabled Cyber-Physical Systems in Out-of-Distribution Scenarios*, presented at ICCPS 2025. The repository for this publication can be found [here](https://github.com/vwlin/SafetyMonitoring_LECPS_OOD).

## Setup
We recommend setting up dependences in a Conda environment with Python version 3.12.2.
```
conda create --name racetrack python=3.12.2 pip
```
We use an augmented version of the [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) racetrack environment by Edouard Leurent. Our local version includes changes that enable determinism and out-of-distribution testing.
```
pip install -e HighwayEnv
```
Install other dependences as shown below.
```
pip install -r requirements.txt
```
install stable-baselines3==2.3.0 ipython==8.23.0 PyVirtualDisplay==3.0 moviepy==1.0.3 tensorboard==2.16.2`

## Training, Testing, and Simulation
A script is provided to train a PPO controller in the augmented highway environment. For example: 
```
python train_racetrack_controller.py --n_other_vehicles 1 --seed 1 --lr 5e-4 --epochs 10 --timesteps 100000
```
For testing and generating simulation traces, run the test script with argument `test` set to `visuals` or `samples`. The former choice will generate plotted trajectories and videos of the simulation. The latter will save simulated trajectory data to a .json file. For example:
```
python test_racetrack_controller.py --n_other_vehicles 1 --seed 1 --test samples
```