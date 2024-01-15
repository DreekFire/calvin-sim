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
import jax.numpy as jnp

class LCPolicy:
    def __init__(self, checkpoint_path, wandb_run_name, N=16, expectile=0.7):
        # We need to first create a dataset object to supply to the agent
        self.N = N
        self.rng = jax.random.PRNGKey(42)
        self.expectile = expectile
        train_paths = [[
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/language_conditioned/training/A/traj0.tfrecord",
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/language_conditioned/training/A/traj1.tfrecord",
            "/nfs/kun2/users/pranav/calvin_ABCD/tfrecord_datasets/language_conditioned/training/A/traj2.tfrecord"
        ]]
        api = wandb.Api()
        run = api.run(wandb_run_name)



        ACTION_PROPRIO_METADATA = run.config["calvin_dataset_config"]["action_proprio_metadata"]
        action_metadata = {
            "mean": ACTION_PROPRIO_METADATA["action"]["mean"],
            "std": ACTION_PROPRIO_METADATA["action"]["std"],
        }

        train_data = CalvinDataset(
            train_paths,
            42,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            batch_size=256,
            sample_weights=None,
            **run.config["dataset_kwargs"],
        )
        text_processor = text_processors["muse_embedding"](
            **{}
        )
        def process_text(batch):
            batch["goals"]["language"] = text_processor.encode(
                [s for s in batch["goals"]["language"]]
            )
            return batch
        train_data_iter = map(process_text, train_data.tf_dataset.as_numpy_iterator())
        example_batch = next(train_data_iter)

        # Next let's initialize the agent
        encoder_def = encoders[run.config["encoder"]](**run.config["encoder_kwargs"])

        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        agent = agents[run.config["agent"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **run.config["agent_kwargs"],
        )

        print("Loading checkpoint...", checkpoint_path) 
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_path, item=agent,)
        if agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {checkpoint_path}")
        print("Checkpoint successfully loaded", checkpoint_path)
        agent = restored

        self.agent = agent
        self.action_statistics = action_metadata
        self.text_processor = text_processor

        # Prepare action buffer for temporal ensembling
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=np.bool)

    def reset(self):
        self.action_buffer = np.zeros((4, 4, 7))
        self.action_buffer_mask = np.zeros((4, 4), dtype=np.bool)

    def predict_action(self, language_command : str, image_obs : np.ndarray):
        obs_input = {"image" : image_obs[np.newaxis, np.newaxis,...].repeat(self.N, axis = 0)}
        goal_input = {"language" : self.text_processor.encode(language_command)[0][np.newaxis,...]}

        # Query model
        self.rng, key = jax.random.split(self.rng)
        action = self.agent.sample_actions(obs_input, goal_input, seed=key, temperature=1.0)
        action = np.array(action.tolist())

        self.rng, key = jax.random.split(self.rng)
        qs = self.agent.state.apply_fn(
            {"params": self.agent.state.params},
            (obs_input, goal_input),
            action.squeeze(),
            rngs={"dropout": key},
            train=False,
            name="critic"
        )
        print(qs)
        # self.rng, key = jax.random.split(self.rng)
        # vs = self.agent.state.apply_fn(
        #     {"params": self.agent.state.params},
        #     (obs_input, goal_input),
        #     train=False,
        #     rngs={"dropout": key},
        #     name="value"
        # )
        # adv = qs - vs
        # print(adv)
        # self.rng, key = jax.random.split(self.rng)
        # tau_weights = jnp.where(adv > 0, self.expectile, 1 - self.expectile)
        # print(tau_weights)
        # sample_idx = jax.random.choice(key, self.N, p = tau_weights/tau_weights.sum())

        # action_prediction = action[sample_idx].squeeze()

        idx = jnp.argmax(qs)
        action_prediction = action[idx].squeeze()


        # # Scale action
        # #action = np.array(self.action_statistics["std"]) * action + np.array(self.action_statistics["mean"])
        # # Shift action buffer
        # if action.shape[0] > 1:
        #     assert action.shape[0] == 4

        #     self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        #     self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        #     self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        #     self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        #     self.action_buffer_mask = self.action_buffer_mask * np.array([[True, True, True, True],
        #                                                                 [True, True, True, False],
        #                                                                 [True, True, False, False],
        #                                                                 [True, False, False, False]], dtype=np.bool)

        #     # Add to action buffer
        #     self.action_buffer[0] = action
        #     self.action_buffer_mask[0] = np.array([True, True, True, True], dtype=np.bool)
        #     # Ensemble temporally to predict action
        #     action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)
        # else:
        #     action_prediction = action[0]

        # Make gripper action either -1 or 1
        if action_prediction[-1] < 0:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        return action_prediction