import gym
import fast_mujoco # GYM CANT FIND IT OTHERWISE
import wandb
from algs.amp_ppo import RL
from utils.config_loader import load_args
import time
import numpy as np

if __name__ == '__main__':
    args = load_args()

    # env = gym.vector.make(args.env_id, num_envs=3, args=args, asynchronous=True, new_step_api=True, render_mode='human', autoreset=False)

    # temp = env.get_attr('get_obs')
    # print(np.asarray(temp).shape)
    #
    # print(temp.shape)

    # state = env.reset()
    # for i in range(1000):
    #     print('-'*60)
    #     print(i)
    #     # action = np.random.rand(env.action_space.sample().shape[0],env.action_space.sample().shape[1] )
    #     print('state')
    #     print(state)
    #
    #     next_state, reward, terminated, done, info = env.step(env.action_space.sample()*.1)
    #
    #     print('terminated', np.array(terminated))
    #     print('unmodified next state')
    #     print(next_state)
    #
    #     if np.sum(np.array(terminated)) >= 1:
    #         print('final observation')
    #         print(info['final_observation'])
    #         next_state[np.where(terminated==True)] = info['final_observation'][np.where(terminated==True)][0]
    #
    #     print('modified next state')
    #     print(next_state)
    #     print()
    #     state = next_state
    #
    # env.close()
    # assert False

    run = None
    run_id = ''
    if args.wandb:
        run = wandb.init(project=args.project, config=args, name=args.exp_name, monitor_gym=True)
        run_id = f"/{run.id}"
        wandb.define_metric("step", hidden=True)
        wandb.define_metric("eval/reward", step_metric="step")
        wandb.define_metric("eval/ep_len", step_metric="step")

        wandb.define_metric("train/critic loss", step_metric="step")
        wandb.define_metric("train/actor loss", step_metric="step")
        wandb.define_metric("train/disc loss", step_metric="step")


    # Create a list of envs for training where the last one is also used to record videos
    envs = [lambda: gym.make(args.env_id, args=args, new_step_api=True, autoreset=True) for _ in range(args.num_envs - 1)] + \
           [lambda: gym.wrappers.RecordVideo(gym.make(args.env_id, args=args, new_step_api=True, autoreset=True, render_mode='rgb_array'), video_folder='results/videos' + run_id, name_prefix="rl-video", episode_trigger=lambda x: x % args.vid_rec_freq == 0, new_step_api=True)]

    # Vectorize environments w/ multi-processing
    envs = gym.vector.AsyncVectorEnv(envs, new_step_api=True)

    # Wrap to record ep rewards and ep lengths
    envs = gym.wrappers.RecordEpisodeStatistics(envs, new_step_api=True, deque_size=50)

    # Initialize RL and Train
    ppo = RL(envs, args)
    ppo.train()

    # Close
    envs.close()

    if args.wandb:
        run.finish()
