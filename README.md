# Hiro's exploration of gymnasium and reinforcement learning
## video ref: https://www.youtube.com/watch?v=OqvXHi_QtT0

# Supported models
- TD3
- A2C
- SAC

# Commands
- Training TD3 on Humanoid-v4: `python sb3.py Humanoid-v4 TD3 -t`
- Training A2C on Humanoid-v4: `python sb3.py Humanoid-v4 A2C -t`
- Training SAC on Humanoid-v4: `python sb3.py Humanoid-v4 SAC -t`
- Testing TD3 on Humanoid-v4: `python sb3.py Humanoid-v4 TD3 -s ./models/TD3_xxxx.zip`
- Testing A2C on Humanoid-v4: `python sb3.py Humanoid-v4 A2C -s ./models/A2C_xxxx.zip`
- Testing SAC on Humanoid-v4: `python sb3.py Humanoid-v4 SAC -s ./models/SAC_xxxx.zip`
