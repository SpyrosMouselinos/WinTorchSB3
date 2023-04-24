#! /bin/bash
### Modify this script according to your needs

# Example of downloading PPO Agent for Pong
python -m rl_zoo3.load_from_hub --algo ppo --env PongNoFrameskip-v4 -orga sb3 -f ./downloads/