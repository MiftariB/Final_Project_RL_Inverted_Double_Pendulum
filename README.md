# Final_Project_RL_Inverted_Double_Pendulum
Inverted Double Pendulum : SearchingHigh-Quality Policies to Control anUnstable Physical System

LINKS: https://github.com/benelot/pybullet-gym
      https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/roboschool/envs/pendulum/inverted_double_pendulum_env.py

## Installing Pybullet-Gym

First, you can perform a minimal installation of OpenAI Gym with
```bash
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

Then, the easiest way to install Pybullet-Gym is to clone the repository and install locally
```bash
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .
```

Important Note: *Do not* use `python setup.py install` as this will not copy the assets (you might get missing SDF file errors).
