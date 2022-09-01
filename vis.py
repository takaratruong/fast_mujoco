import gym
import fast_mujoco # GYM CANT FIND IT OTHERWISE
import wandb
from algs.amp_ppo import RL
from utils.config_loader import load_args
import time
import numpy as np
from algs.amp_models import ActorCriticNet
import torch


if __name__ == '__main__':
    args = load_args()
    env = gym.vector.make(args.env_id, num_envs=1, args=args, new_step_api=True, render_mode='human')

    model = ActorCriticNet(env.single_observation_space.shape[0], env.single_action_space.shape[0], [256, 256])
    # model = ActorCriticNet(env.observation_space.shape[0], env.action_space.shape[0], [512, 512])
    # policy_path = 'results/models/baseline_zm/baseline_zm_iter3000.pt'
    policy_path = 'results/models/gym/gym_iter1600.pt'

    model.load_state_dict(torch.load(policy_path))  # relative file path
    model.cuda()

    state = env.reset()
    for _ in range(5000):
        # state = np.asarray(env.get_attr('get_obs'))

        with torch.no_grad():
            act = model.sample_best_actions(torch.from_numpy(state).cuda().type(torch.cuda.FloatTensor)).cpu().numpy()
        state, reward, terminated, done, info = env.step(act)
        # print(state)
        # print(reward)
        # print(terminated)
        # print(done)
        # print(info)
        # print()
