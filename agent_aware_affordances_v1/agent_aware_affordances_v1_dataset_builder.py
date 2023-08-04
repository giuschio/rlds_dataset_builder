from typing import Iterator, Tuple, Any
from pathlib import Path
import os

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


# function: get all folders of type iteration
def get_iter_folders(proot: Path):
    subdirs = [Path(f[0]) for f in os.walk(str(proot))]
    subdirs = [str(f) for f in subdirs if 'iteration' in f.name]
    return subdirs


class AgentAwareAffordancesV1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation. Not available for this dataset, will be set to np.zeros.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='State, consists of [end-effector pose (x,y,z,yaw,pitch,roll) in world frame, 1x gripper open/close, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(6,),
                        dtype=np.float32,
                        doc='Robot action, consists of [end-effector velocity (v_x,v_y,v_z,omega_x,omega_y,omega_z) in world frame',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'input_point_cloud': tfds.features.Tensor(
                        shape=(50000,3),
                        dtype=np.float32,
                        doc='Point cloud (geometry only) of the object at the beginning of the episode (world frame) as a numpy array (50000,3).'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='../experiment_data_v1'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        print('generating examples')

        def _parse_example(episode_path:str):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            task = 'open' if 'open' in episode_path else 'close'
            print(f'task is {task}')

            epath = Path(episode_path)

            if not os.path.isfile(str(epath/'pcd_np.npy')):
                # check if episode is empty
                return None
            

            point_cloud = np.load(epath/'pcd_np.npy').astype(np.float32)
            # ee_trajectory is [x,y,z,yaw,pitch,roll]
            ee_trajectory = np.load(epath/'ee_poses.npy').astype(np.float32)
            # ee_velocity is [v_x,v_y,v_z,omega_x,omega_y,omega_z]
            ee_velocity = np.load(epath/'ee_velocities.npy').astype(np.float32)
            # oven opening angle
            object_states = np.load(epath/'oven_states.npy').astype(np.float32)

            point_cloud /= 4.0
            ee_trajectory[:, :3] /= 4.0
            ee_velocity[:, :3] /= 4.0

            episode_length = max(object_states.shape)

            if task == 'open':
                task_success = (object_states[-1] - object_states[0]) > 10
                instruction = 'open the oven'
            else:
                task_success = (object_states[0] - object_states[-1]) > 10
                instruction = 'close the oven'


            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(episode_length):
                # compute Kona language embedding
                language_embedding = self._embed([instruction])[0].numpy()

                if float(i == (episode_length - 1)):
                    reward = 1.0 if task_success else 0.0
                else:
                    reward = 0.

                gripper_state = np.array([0]).astype(np.float32)
                state = np.concatenate([ee_trajectory[i], gripper_state, object_states[i]]).astype(np.float32)

                episode.append({
                    'observation': {
                        'image': np.asarray(np.zeros((64, 64, 3)), dtype=np.uint8),
                        'state': state,
                    },
                    'action': ee_velocity[i],
                    'discount': 1.0,
                    'reward': np.array(reward).astype(np.float32),
                    'is_first': i == 0,
                    'is_last': i == (episode_length - 1),
                    'is_terminal': i == (episode_length - 1),
                    'language_instruction': instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'input_point_cloud': point_cloud
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = get_iter_folders(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            res = _parse_example(sample)
            if res is not None:
                yield res

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

