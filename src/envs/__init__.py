from functools import partial
import sys
import os

#smacv2
from smacv2.env.multiagentenv import MultiAgentEnv 
from smacv2.env.starcraft2.starcraft2 import StarCraft2Env 
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


# RiftRunner
from .RiftRunner import RiftRunner

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv

except:
    gfootball = False


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)


# RiftRunner
REGISTRY["RiftRunner"] = partial(env_fn, env=RiftRunner)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
