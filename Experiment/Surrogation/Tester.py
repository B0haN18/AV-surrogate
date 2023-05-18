import pandas as pd
import numpy as np
import os
import time
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import TD3

model = TD3.load("../../codes/models/results.zip")
actions, _ = model.predict([])
print(actions)
