from collections import namedtuple
from dataclasses import dataclass
import os
import sys
import importlib.util as _il
import inspect
from typing import Sequence, Tuple, Type
from torch import Tensor
import torch
from tud_rl import logger

import gym
from gym.envs.registration import EnvSpec


def spec_gen(id: str, path: os.PathLike) -> EnvSpec:
    """
    Generate a gym.EnvSpec
    which will be used to dynamically register a 
    gym Env at ``path``, with the name given by ``id``

    The idea is to regsiter a python file with a single
    `gym.Env` class anywhere on your machine as a gym 
    Environment for the GAIL algorithm. 
    """
    _classes = []

    # Load the python file at `path` as 
    # a module at `aisgail.__env__`
    module_spec = _il.spec_from_file_location("aisgail.__env__",path)
    mod = _il.module_from_spec(module_spec) # type: ignore
    sys.modules["tud_rl.__currentenv__"] = mod
    module_spec.loader.exec_module(mod) # type: ignore
    # Walk over all module objects and filter for gym-like classes
    for _, obj in inspect.getmembers(mod):
        if (inspect.isclass(obj) 
        and all(hasattr(obj, a) for a in ["reset","step","render"])
        and issubclass(obj,gym.Env)):
            _classes.append(obj)
    _check_results(_classes)
    _envclass = _classes[0]
    return EnvSpec(id=id, entry_point=_envclass)

def _check_results(classes: Sequence[Tuple[str,Type[gym.Env]]]) -> None:
    if len(classes) > 1:
        raise RuntimeError(
            "More than one Environment found in file at provided path or in its imports. "
            f"Found classes {classes}. "
            "Please make sure to specify only one environment per file."
        )
    elif not classes:
        raise RuntimeError(
            "No environment file found for provided path."
        )

def make_env(id: str, path: os.PathLike, **env_settings) ->gym.Env:

    logger.info(
        f"Registering environment `{id}` "
        f"from given path `{path}`"
        )
    spec_ = spec_gen(id,path)
    return gym.make(spec_, **env_settings)

@dataclass
class GailConfig:
    """
    Dataclass for parsing yaml config files
    """
    env_id: str
    env_path: os.PathLike
    train: bool
    gamma: float
    clip_eps: float
    lr_actor: float
    lr_critic: float
    lr_discriminator: float
    train_steps: int
    max_episode_len: int
    update_steps: int
    checkpoint_steps: int
    num_epochs: int
    num_d_epochs: int
    seed: int

@dataclass
class PreparedBuffer:
    state_hist: torch.Tensor
    action_hist: torch.Tensor
    action_log_prob_hist: torch.Tensor
    learner_state_actions: torch.Tensor

class TemporaryBuffer:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.is_terminal = []

    def reset(self):
        self.__init__()
        
    def prepare(self, device, n_actions):
        state_hist = torch.stack(
            self.states).to(device)
        action_hist = torch.stack(
            self.actions).to(device)
        action_log_prob_hist = torch.stack(
            self.action_log_probs).to(device)
        
        # One-hot-enocde actions 
        # from recorded trajectory
        one_hot_actions = torch.eye(n_actions)[action_hist.long()]
        one_hot_actions = one_hot_actions.to(self.device)
        
        # Stack state and actions for discriminator input
        learner_state_actions = torch.cat([state_hist, one_hot_actions], dim=1)
        learner_state_actions = learner_state_actions.to(device)
        
        return PreparedBuffer(
            state_hist,action_hist,
            action_log_prob_hist,
            learner_state_actions
        )