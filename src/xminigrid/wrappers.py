from __future__ import annotations

from typing import Any, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer

from xminigrid.core.goals import AgentOnTileGoal
from xminigrid.core.grid import check_walkable

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
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {**base_shape, **{"direction": 4}}
        else:
            obs_shape = {
                "img": self._env.observation_shape(params),
                "direction": 4,
            }
        return obs_shape

    def __extend_obs(self, timestep):
        direction = jax.nn.one_hot(timestep.state.agent.direction, num_classes=4)
        if isinstance(timestep.observation, dict):
            assert "img" in timestep.observation
            extended_obs = {
                **timestep.observation,
                **{"direction": direction},
            }
        else:
            extended_obs = {
                "img": timestep.observation,
                "direction": direction,
            }

        timestep = timestep.replace(observation=extended_obs)
        return timestep

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
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {
                **base_shape,
                **{
                    "goal_encoding": params.ruleset.goal.shape,
                    "rule_encoding": params.ruleset.rules.shape,
                },
            }
        else:
            obs_shape = {
                "img": self._env.observation_shape(params),
                "goal_encoding": params.ruleset.goal.shape,
                "rule_encoding": params.ruleset.rules.shape,
            }
        return obs_shape

    def __extend_obs(self, timestep):
        goal_encoding = timestep.state.goal_encoding
        rule_encoding = timestep.state.rule_encoding
        if isinstance(timestep.observation, dict):
            assert "img" in timestep.observation
            extended_obs = {
                **timestep.observation,
                **{
                    "goal_encoding": goal_encoding,
                    "rule_encoding": rule_encoding,
                },
            }
        else:
            extended_obs = {
                "img": timestep.observation,
                "goal_encoding": goal_encoding,
                "rule_encoding": rule_encoding,
            }

        timestep = timestep.replace(observation=extended_obs)
        return timestep

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__extend_obs(timestep)
        return timestep


class DistanceToGoalRewardWrapper(Wrapper):
    """Potential-based reward shaping that encourages the agent to move closer to the goal."""

    def __init__(self, env, scale: float = 0.25, normalize: bool = True):
        super().__init__(env)
        self.scale = scale
        self.normalize = normalize

    def _compute_distance(self, grid: Integer[Array, " h w c"], row: Integer[Array, ""], col: Integer[Array, ""]):
        H, W, _ = grid.shape
        coords = jnp.mgrid[:H, :W].reshape(2, -1)
        walkable = jax.vmap(check_walkable, in_axes=(None, 1))(grid, coords).reshape(H, W)
        dist = jnp.full((H, W), jnp.inf).at[row, col].set(0.0)

        def step(dist, _):
            dist = jax.tree.reduce(
                jnp.minimum,
                [jnp.roll(dist, a, axis) + 1 for a in (1, -1) for axis in (0, 1)],
                dist,
            )
            dist = jnp.where(walkable, dist, jnp.inf)
            dist = dist.at[row, col].set(0.0)
            return dist, None

        dist_matrix, _ = jax.lax.scan(step, dist, length=H * W)
        return dist_matrix / (H * W if self.normalize else 1)

    def _potential(self, state: State):
        ar, ac = state.agent.position
        return state.carry[ar, ac]

    def reset(self, params: Any, key: Array) -> TimeStep:
        timestep = self._env.reset(params, key)
        state = timestep.state
        tile = AgentOnTileGoal.decode(state.goal_encoding).tile
        gr, gc = (state.grid == tile).all(axis=-1).nonzero(size=1)
        grid = self._compute_distance(state.grid, gr, gc)
        return timestep.replace(state=state.replace(carry=grid))

    def step(self, params, timestep, action):
        next_timestep = self._env.step(params, timestep, action)
        phi_s = self._potential(timestep.state)
        phi_s_next = self._potential(next_timestep.state)
        shaping = self.scale * (next_timestep.discount * phi_s_next - phi_s)
        new_reward = next_timestep.reward + shaping
        return next_timestep.replace(reward=new_reward)
