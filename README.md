# Hiro's exploration of gymnasium and reinforcement learning
## video ref: https://www.youtube.com/watch?v=OqvXHi_QtT0

> [!WARNING]
> This project is doable with macbook m1 but it is very difficult to set up. I recommend using a linux machine.

# Supported models
- TD3
- A2C
- SAC
- PPO

# Installation
- Install python 3.10(seems to work the best)
- create a virtual environment `python -m venv .venv`
- activate the virtual environment `source .venv/bin/activate`
- install the requirements `pip install -r requirements.txt`

# Commands
- Training TD3 on Humanoid-v4: `python sb3.py Humanoid-v4 TD3 -t`
- Training A2C on Humanoid-v4: `python sb3.py Humanoid-v4 A2C -t`
- Training SAC on Humanoid-v4: `python sb3.py Humanoid-v4 SAC -t`
- Testing TD3 on Humanoid-v4: `python sb3.py Humanoid-v4 TD3 -s ./models/TD3_xxxx.zip`
- Testing A2C on Humanoid-v4: `python sb3.py Humanoid-v4 A2C -s ./models/A2C_xxxx.zip`
- Testing SAC on Humanoid-v4: `python sb3.py Humanoid-v4 SAC -s ./models/SAC_xxxx.zip`
