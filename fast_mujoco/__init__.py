from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec
import gym
#
# _load_env_plugins()

register(
    id="Humanoid_Treadmill",
    entry_point="fast_mujoco.envs:HumanoidTreadmillEnv",
)

