# Hiro's exploration of gymnasium and reinforcement learning
## video ref: https://www.youtube.com/watch?v=OqvXHi_QtT0

> [!NOTE]
> Currently, the code is not working as expected.
> For m1 mbp, the gymnasium <0.29.1 is not working and the latest version is not compatible with stable-baselines3.

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
