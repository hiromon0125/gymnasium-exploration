import argparse
import os

import gymnasium as gym
from stable_baselines3 import A2C, SAC, TD3
import torch

# create directory for logging and models
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
model_dir = 'models'
log_dir = 'logs'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# env is the gym environment to train our model on
def train(env, sb3_algo):
  match sb3_algo:
    case 'A2C':
      # MlpPolicy is a simple neural network policy that works for environments with continuous and discrete action spaces
      # there are other policies available in stable_baselines3 like CnnPolicy for image-based inputs but will not be used here
      model = A2C('MlpPolicy', env, device=device, verbose=1, tensorboard_log=log_dir)
      # You can set learning rate and gamma in the parameter here, but we will use the default values for now
    case 'SAC':
      model = SAC('MlpPolicy', env, device=device, verbose=1, tensorboard_log=log_dir)
    case 'TD3':
      model = TD3('MlpPolicy', env, device=device, verbose=1, tensorboard_log=log_dir)
    case _:
      raise ValueError(f'Unsupported algorithm: {sb3_algo}')
    
  TIMESTEPS = 25000
  iters = 0
  # We will train the model until we are satisfied with the performance of the model
  while True:
    iters += 1
    
    # We will train the model for <TIMESTEPS> timesteps and save the model
    # a single step is a single action taken by the agent in the environment
    # reset_num_timesteps=False ensures that the model does not reset the number of timesteps 
    # else the model will start training from scratch every iteration (0 - TIMESTEPS)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    # We can save the model at regular intervals so we can test the model while training
    model.save(f'{model_dir}/{sb3_algo}_{iters * TIMESTEPS}')
    
def test(env, sb3_algo, path_to_model):
  match sb3_algo:
    case 'A2C':
      model = A2C.load(path_to_model, env=env)
    case 'SAC':
      model = SAC.load(path_to_model, env=env)
    case 'TD3':
      model = TD3.load(path_to_model, env=env)
    case _:
      raise ValueError(f'Unsupported algorithm: {sb3_algo}')
    
  obs = env.reset()[0]
  done = False
  while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info, _ = env.step(action)
    env.render()
    if done:
      extra_steps -= 1
      if extra_steps < 0:
        break
    
  
if __name__ == '__main__':
  # parse command line arguments
  parser = argparse.ArgumentParser(description='Train or test a model using stable baselines3')
  parser.add_argument('gymenv', type=str, help='The gym environment to train/test the model on i.e. Humanoid-v4')
  parser.add_argument('sb3_algo', type=str, help='The algorithm to use for training/testing the model i.e. A2C, SAC, TD3')
  parser.add_argument('-t', '--train', action='store_true', help='Train the model')
  parser.add_argument('-s', '--test', metavar='path_to_model', help='Path to the model to test')
  args = parser.parse_args()
  
  if args.train:
    gymenv = gym.make(args.gymenv, render_mode=None)
    train(gymenv, args.sb3_algo)
    
  if args.test:
    if os.path.isfile(args.test):
      gymenv = gym.make(args.gymenv, render_mode='human')
      test(gymenv, args.sb3_algo, path_to_model=args.test)
    else:
      print(f'Path to model {args.test} does not exist')
      exit(1)
  