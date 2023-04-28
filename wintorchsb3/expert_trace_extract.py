import random

import numpy.random
from stable_baselines3 import PPO, DQN
from huggingface_sb3 import load_from_hub
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm


class ExtractedModelPolicy(nn.Module):
    def __init__(self, fe, an):
        super().__init__()
        self.fe = fe if fe is not None else nn.Identity()
        self.an = an if an is not None else nn.Identity()

    def forward(self, x):
        feats = self.fe(x)
        actions = self.an(feats)
        return feats, actions

    def predict(self, x):
        feats = self.fe(x)
        actions = self.an(feats)
        return feats, actions


def dqn_load_pong():
    checkpoint = load_from_hub("sb3/dqn-PongNoFrameskip-v4", "dqn-PongNoFrameskip-v4.zip")

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "exploration_schedule": lambda _: 0.0,
        "optimize_memory_usage": False
    }

    model = DQN.load(checkpoint, custom_objects=custom_objects)
    model.policy.set_training_mode(False)
    extracted_preprocessing = lambda x: model.policy.obs_to_tensor(x.transpose((0, 3, 1, 2)).float())

    extracted_model = ExtractedModelPolicy(fe=model.policy.q_net, an=None)

    env = make_atari_env('PongNoFrameskip-v4', n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    return env, extracted_model, extracted_preprocessing


def ppo_load_pong(n_envs=1):
    checkpoint = load_from_hub("sb3/ppo-PongNoFrameskip-v4", "ppo-PongNoFrameskip-v4.zip")

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = PPO.load(checkpoint, custom_objects=custom_objects, exact_match=False)
    if model.policy.normalize_images:
        normalizer = 255.0
    else:
        normalizer = 1.0
    model.policy.set_training_mode(False)
    extracted_preprocessing = lambda x: model.policy.obs_to_tensor(x.transpose((0, 3, 1, 2)))[0].float() / normalizer

    extracted_model = ExtractedModelPolicy(fe=model.policy.features_extractor, an=model.policy.action_net)

    env = make_atari_env('PongNoFrameskip-v4', n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
    buttons = env.action_space.n
    return env, 'Pong', extracted_model, extracted_preprocessing, buttons


def ppo_load_breakout(n_envs=1):
    checkpoint = load_from_hub("sb3/ppo-BreakoutNoFrameskip-v4", "ppo-BreakoutNoFrameskip-v4.zip")

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = PPO.load(checkpoint, custom_objects=custom_objects, exact_match=False)
    if model.policy.normalize_images:
        normalizer = 255.0
    else:
        normalizer = 1.0
    model.policy.set_training_mode(False)
    extracted_preprocessing = lambda x: model.policy.obs_to_tensor(x.transpose((0, 3, 1, 2)))[0].float() / normalizer

    extracted_model = ExtractedModelPolicy(fe=model.policy.features_extractor, an=model.policy.action_net)

    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
    buttons = env.action_space.n
    return env, 'Breakout', extracted_model, extracted_preprocessing, buttons


def ppo_load(game):
    if game == 'Pong':
        return ppo_load_pong()
    elif game == 'Breakout':
        return ppo_load_breakout()


def maybe_extract_trace(game, random_act_prob=0.01):
    if game == 'Pong':
        expected_score = 15
        env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_pong(1)
    elif game == 'Breakout':
        expected_score = 80
        env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_breakout(1)
    else:
        raise NotImplementedError
    observations = []
    img_features = []
    actions_probs = []
    obs = env.reset()
    sum_rewards = 0
    n_examples = 0
    while True:
        observation = extracted_preprocessing(obs)
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
        action = actions.cpu().numpy().reshape((1, buttons))
        action_pos = action.argmax(-1)
        for o in obs:
            observations.append(o)
        for f in feats:
            img_features.append(f)
        for a in action:
            actions_probs.append(a)
        if random.uniform(0, 1) < random_act_prob:
            action_pos = [env.action_space.sample()]
        obs, rewards, done, info = env.step(action_pos)
        sum_rewards += rewards[0]
        n_examples += 1
        if done[0] or (not done[0] and n_examples > 3_000):
            if sum_rewards >= expected_score:
                return observations, img_features, actions_probs, buttons, n_examples
            else:
                n_examples = 0
                sum_rewards = 0
                obs = env.reset()


def extract_expert_trace(game='Pong', n_games=20, random_act_prob=0.01):
    valid_games = 0
    observations = []
    img_features = []
    actions_probs = []
    while valid_games < n_games:
        observations_, img_features_, actions_probs_, buttons, examples = maybe_extract_trace(game=game,
                                                                                              random_act_prob=random_act_prob)
        for f in observations_:
            observations.append(f)
        for f in img_features_:
            img_features.append(f)
        for f in actions_probs_:
            actions_probs.append(f)

        valid_games += 1
    return game, observations, img_features, actions_probs, buttons


class MultiModalDS(Dataset):
    def __init__(self, sources, n_games, args):
        self.GAMES = ['Pong', 'Breakout']
        self.IMAGES = ['Cifar100']
        self.SOUNDS = ['Random']
        self.sources = sources
        self.n_games = n_games
        self.args = args
        self.repack_blobs(self.prepare_sources())


    def prepare_game(self, game, source_id):
        blob = extract_expert_trace(game=game, n_games=self.n_games[source_id],
                                    random_act_prob=self.args[source_id]['random_act_prob'])
        return blob

    def prepare_sources(self):
        source_blobs = []
        for source_id in range(len(self.sources)):
            if self.sources[source_id] in self.GAMES:
                source_blobs.append(self.prepare_game(self.sources[source_id], source_id))
        return source_blobs

    def repack_blobs(self, source_blobs):
        total_examples = 0
        blob_ids = []
        blob_raw_features = []
        blob_expert_features = []
        blob_expert_responses = []
        max_buttons = max(blob[4] for blob in source_blobs)
        for blob in source_blobs:
            new_examples = max(len(blob[1]), len(blob[2]), len(blob[3]))
            total_examples += new_examples
            for f in [blob[0]] * new_examples:
                blob_ids.append(f)
            for f in blob[1]:
                blob_raw_features.append(f)
            for f in blob[2]:
                blob_expert_features.append(f)
            for f in blob[3]:
                if len(f) < max_buttons:
                    blob_expert_responses.append(
                        np.pad(f, pad_width=(0, max_buttons - len(f)), mode='constant', constant_values=min(f)))
                else:
                    blob_expert_responses.append(f)

        self.total_examples = total_examples
        self.blob_ids = blob_ids
        self.blob_raw_features = blob_raw_features
        self.blob_expert_features = blob_expert_features
        self.blob_expert_responses = blob_expert_responses
        self.max_buttons = max_buttons

    # def maybe_execute(self):
    #     if os.path.exists('prepacked_ng_{}_rnd_{}.pkl'):
    #         pass


    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        return self.blob_ids[idx], self.blob_raw_features[idx], self.blob_expert_features[idx], \
               self.blob_expert_responses[idx]


def test_pong():
    REWARDS = 0
    env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_pong(n_envs=1)
    obs = env.reset()
    while True:
        observation, _ = extracted_preprocessing(obs)
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
        action = actions.cpu().numpy().reshape((1, buttons))
        action_pos = action.argmax(-1)
        obs, rewards, done, info = env.step(action_pos)
        REWARDS += rewards[0]
        env.render()
        if done:
            env.close()
            print(REWARDS)
            return


def test_breakout():
    REWARDS = 0
    env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_breakout(n_envs=1)
    obs = env.reset()
    while True:
        observation = extracted_preprocessing(obs)
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
        action = actions.cpu().numpy().reshape((1, buttons))
        action_pos = action.argmax(-1)
        if random.uniform(0, 1) < 0.015:
            action_pos = [env.action_space.sample()]
        obs, rewards, done, info = env.step(action_pos)
        REWARDS += rewards[0]
        env.render()
        if done:
            env.close()
            print(REWARDS)
            return

        # from torch.utils.data import DataLoader

#
# train_dataloader = DataLoader(
#     MultiModalDS(sources=['Pong', 'Breakout'], n_examples=[4, 4], n_jobs=[1, 1],
#                  args=[{'random_act_prob': 0.05}, {'random_act_prob': 0.05}]),
#     batch_size=2, shuffle=True)
# test_dataloader = DataLoader(
#     MultiModalDS(sources=['Pong', 'Breakout'], n_examples=[20, 15], n_jobs=[1, 1], args=[{'random_act_prob': 0.00}]),
#     batch_size=5, shuffle=False)

# for batch in train_dataloader:
#     print('hey')
#     print('hoi')
