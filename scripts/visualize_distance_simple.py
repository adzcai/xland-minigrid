"""Visualize the distance matrix computed by DistanceToGoalRewardWrapper at reset time."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from xminigrid.envs import FourRooms
from xminigrid.wrappers import DistanceToGoalRewardWrapper


def main():
    # Create environment and wrapper
    env = FourRooms()
    params = env.default_params(height=9, width=9, view_size=5, max_steps=50)
    wrapped = DistanceToGoalRewardWrapper(env, scale=0.1, normalize=True)

    # Reset to generate a problem and compute distance matrix
    key = jax.random.PRNGKey(0)
    timestep = wrapped.reset(params, key)
    
    # Extract distance matrix from state.carry
    dist_matrix = timestep.state.carry.potential
    agent_pos = timestep.state.agent.position

    H, W = dist_matrix.shape
    print(f"Distance matrix shape: {H}x{W}")
    print(f"Agent position: {agent_pos}")
    print("Distance matrix stats:")
    print(f"  Min: {np.nanmin(dist_matrix):.6f}")
    print(f"  Max: {np.nanmax(dist_matrix):.6f}")
    print(f"  Mean: {np.nanmean(dist_matrix):.6f}")
    print(f"  Inf count: {np.sum(np.isinf(dist_matrix))}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot distance heatmap
    im = ax.imshow(dist_matrix, cmap="viridis", origin="upper")
    
    # Mark agent position
    ax.scatter([agent_pos[1]], [agent_pos[0]], color="red", s=200, marker="*", label="Agent", zorder=5)
    
    # Add grid lines
    for i in range(H + 1):
        ax.axhline(i - 0.5, color="black", linewidth=0.5, alpha=0.2)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color="black", linewidth=0.5, alpha=0.2)
    
    ax.set_title("Distance to Goal (computed at reset)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    
    cbar = plt.colorbar(im, ax=ax, label="Distance")
    
    fig.tight_layout()
    fig.savefig("/tmp/distance_simple.png", dpi=150, bbox_inches="tight")
    print("\nSaved: /tmp/distance_simple.png")


if __name__ == "__main__":
    main()
