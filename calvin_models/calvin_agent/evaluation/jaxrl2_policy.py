import json
from jaxrl_m.vision import encoders
from jaxrl_m.data.calvin_dataset import CalvinDataset
import jax
import orbax.checkpoint
from jaxrl_m.agents import agents
import numpy as np
import os
from jaxrl_m.data.text_processing import text_processors
import wandb
import gym
from gym.spaces import Box, Dict
from jaxrl2.utils.general_utils import add_batch_dim
from ml_collections.config_dict import ConfigDict
from jaxrl2.agents import PixelBCLearner, PixelIQLLearner
from jaxrl2.agents.cql_encodersep_parallel.pixel_cql_learner import PixelCQLLearnerEncoderSepParallel
import copy

class DummyEnv():
    def __init__(self, variant, num_tasks):
        super().__init__()
        obs_dict = dict()
        if not variant.from_states:
            obs_dict['pixels'] = Box(low=0, high=255, shape=(200, 200, 3, 1), dtype=np.uint8)
        if variant.add_states:
            obs_dict['state'] = Box(low=-100000, high=100000, shape=(7,), dtype=np.float32)
        obs_dict['task_id'] = Box(low=0, high=1, shape=(num_tasks,), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.spec = None
        self.action_space = Box(
            np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
            np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
            dtype=np.float32)

    def seed(self, seed):
        pass

class Policy:
    def __init__(self, checkpoint_path, wandb_run_name):
        api = wandb.Api()
        run = api.run(wandb_run_name)

        variant = ConfigDict(run.config)
        self.task_id_mapping = variant.task_id_mapping

        num_tasks = len(self.task_id_mapping)
        print(self.task_id_mapping)
        print(num_tasks)
        env = DummyEnv(variant, num_tasks)
        sample_obs = add_batch_dim(env.observation_space.sample())
        sample_action = add_batch_dim(env.action_space.sample())
        print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])

        if "bc" in variant.algorithm:
            variant['train_kwargs']['use_gaussian_policy'] = True
            agent = PixelBCLearner(variant.seed, sample_obs, sample_action, **variant['train_kwargs'])
        elif 'cql' in variant.algorithm:
            agent = PixelCQLLearnerEncoderSepParallel(variant.seed, sample_obs, sample_action, **variant['train_kwargs'])
        elif 'iql' in variant.algorithm:
            agent = PixelIQLLearner(variant.seed, sample_obs, sample_action, **variant['train_kwargs'])
        print("Loading checkpoint...", checkpoint_path) 
        agent.restore_checkpoint(checkpoint_path)
        print("Checkpoint successfully loaded", checkpoint_path)
        self.agent = agent

    def reset(self):
        pass

    def predict_action(self, subtask : str, image_obs : np.ndarray):
        task_id = self.task_id_mapping[subtask]
        # one-hot encode task id
        task_id_onehot = np.zeros(len(self.task_id_mapping), dtype=np.float32)
        task_id_onehot[task_id] = 1
        obs_input = {"pixels" : image_obs[np.newaxis, ..., np.newaxis], "task_id" : task_id_onehot[np.newaxis, ...]}

        # Query model
        action_prediction = self.agent.eval_actions(obs_input)
        action_prediction = copy.deepcopy(jax.device_get(action_prediction).squeeze())
        if action_prediction[-1] < 0:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1


        return action_prediction