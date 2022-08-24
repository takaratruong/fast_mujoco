import wandb
import gym
from utils.utils import VideoCallback
#

from algs.amp_ppo import RL

if __name__ == '__main__':
    # Create Envs
    train_envs = gym.wrappers.RecordEpisodeStatistics(gym.vector.make('Humanoid_Treadmill', num_envs=50, new_step_api=True, asynchronous=True), 100)

    vid_env = VideoCallback(gym.make('Humanoid_Treadmill', render_mode='rgb_array'))

    # Instantiate algorithm
    ppo = RL(train_envs, vid_env, args)

    # Train
    ppo.train()


