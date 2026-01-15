from __future__ import annotations

from typing import Any, Union

import jax
import jax.numpy as jnp

from .core.goals import MAX_GOAL_ENCODING_LEN
from .environment import Environment, EnvParamsT
from .types import EnvCarryT, IntOrArray, State, TimeStep


class Wrapper(Environment[EnvParamsT, EnvCarryT]):
    def __init__(self, env: Environment[EnvParamsT, EnvCarryT]):
        self._env = env

    # Question: what if wrapper adds new parameters to the dataclass?
    # Solution: do this after applying the wrapper:
    #   env_params = wrapped_env.default_params(**dataclasses.asdict(original_params))
    def default_params(self, **kwargs) -> EnvParamsT:
        return self._env.default_params(**kwargs)

    def num_actions(self, params: EnvParamsT) -> int:
        return self._env.num_actions(params)

    def observation_shape(self, params: EnvParamsT) -> Union[tuple[int, int, int], dict[str, Any]]:
        return self._env.observation_shape(params)

    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]:
        return self._env._generate_problem(params, key)

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        return self._env.reset(params, key)

    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        return self._env.step(params, timestep, action)

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]):
        return self._env.render(params, timestep)


# gym and gymnasium style reset (on the same step with termination)
class GymAutoResetWrapper(Wrapper):
    def __auto_reset(self, params, timestep):
        key, _ = jax.random.split(timestep.state.key)
        reset_timestep = self._env.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    # TODO: add last_obs somewhere in the timestep? add extras like in Jumanji?
    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep


# dm_env and envpool style reset (on the next step after termination)
class DmEnvAutoResetWrapper(Wrapper):
    def step(self, params, timestep, action):
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self._env.reset(params, timestep.state.key),
            lambda: self._env.step(params, timestep, action),
        )
        return timestep


# Yes, these are a bit stupid, but a tmp workaround to not write an actual system for spaces.
# May be, in the future, I will port the entire API to some existing one, like functional Gymnasium.
# For now, faster to do this stuff with dicts instead...
# NB: if you do not want to use this (due to the dicts as obs),
# just get needed parts from the original TimeStep and State dataclasses
class DirectionObservationWrapper(Wrapper):
    def observation_shape(self, params):
        base_shape = self._env.observation_shape(params)
        obs_shape = extend_obs(base_shape, direction=4)
        return obs_shape

    def __extend_obs(self, timestep):
        direction = jax.nn.one_hot(timestep.state.agent.direction, num_classes=4)
        extended_obs = extend_obs(timestep.observation, direction=direction)
        return timestep.replace(observation=extended_obs)

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__extend_obs(timestep)
        return timestep


class RulesAndGoalsObservationWrapper(Wrapper):
    def observation_shape(self, params):
        base_shape = self._env.observation_shape(params)
        obs_shape = extend_obs(
            base_shape, goal_encoding=params.ruleset.goal.shape, rule_encoding=params.ruleset.rules.shape
        )
        return obs_shape

    def __extend_obs(self, timestep):
        goal_encoding = timestep.state.goal_encoding
        rule_encoding = timestep.state.rule_encoding
        extended_obs = extend_obs(timestep.observation, goal_encoding=goal_encoding, rule_encoding=rule_encoding)
        return timestep.replace(observation=extended_obs)

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__extend_obs(timestep)
        return timestep


class GoalObservationWrapper(Wrapper):
    def observation_shape(self, params) -> dict[str, tuple[int, ...]]:
        base_shape = self._env.observation_shape(params)
        return extend_obs(base_shape, goal_encoding=MAX_GOAL_ENCODING_LEN)

    def __extend_obs(self, timestep) -> TimeStep:
        extended_obs = extend_obs(timestep.observation, goal_encoding=timestep.state.goal_encoding)
        return timestep.replace(observation=extended_obs)

    def reset(self, params, key) -> TimeStep:
        timestep = self._env.reset(params, key)
        return self.__extend_obs(timestep)

    def step(self, params, timestep, action) -> TimeStep:
        timestep = self._env.step(params, timestep, action)
        return self.__extend_obs(timestep)


class PrevActionWrapper(Wrapper[EnvParamsT, tuple[EnvCarryT, Array]]):
    """Put the previous action into the observation."""

    def observation_shape(self, params) -> dict[str, tuple[int, ...]]:
        shape = self._env.observation_shape(params)
        return extend_obs(shape, prev_action=self._env.num_actions(params), reward=(), done=())

    def __extend_timestep(self, timestep: TimeStep, prev_action, action) -> TimeStep:
        extended_state = timestep.state.replace(carry=(timestep.state.carry, action))
        obs = extend_obs(
            timestep.observation,
            prev_action=prev_action,
            reward=timestep.reward,
            done=timestep.last(),
        )
        return timestep.replace(state=extended_state, observation=obs)

    def reset(self, params, key) -> TimeStep:
        timestep = self._env.reset(params, key)
        zero_action = jnp.asarray(0, dtype=jnp.uint8)
        return self.__extend_timestep(timestep, zero_action, zero_action)

    def step(self, params, timestep, action) -> TimeStep:
        base_carry, prev_action = timestep.state.carry

        # base environment step
        base_state = timestep.state.replace(carry=base_carry)
        timestep = timestep.replace(state=base_state)
        timestep = self._env.step(params, timestep, action)

        # add taken action to carry
        action = jnp.asarray(action, dtype=jnp.uint8)
        return self.__extend_timestep(timestep, prev_action, action)


def extend_obs(base, **kwargs):
    if isinstance(base, dict):
        assert "img" in base, base
        out = {**base, **kwargs}
    else:
        out = {"img": base, **kwargs}
    return out
