import jax
import jax.numpy as jnp
from jumanji.environments.packing.bin_pack.env import ExtendedBinPack
from jumanji.environments.packing.bin_pack.generator import (
    ExtendedRandomGenerator,
)
from jumanji.environments.packing.bin_pack.types import Observation, State
from jumanji.types import TimeStep, termination
from jumanji import specs
import functools
from typing import Tuple, List

n_items = 10
max_num_ems = 3 * n_items

# Create an instance of the extended generator with rotation but not values
generator = ExtendedRandomGenerator(
    max_num_items=n_items,
    max_num_ems=max_num_ems,
    is_rotation_allowed=True,
    is_value_based=False,
)

# Create an instance of the extended environment
env = ExtendedBinPack(
    generator=generator,
    obs_num_ems=max_num_ems,
    is_rotation_allowed=True,
    is_value_based=False,
    full_support=True,
    debug=True # Enable debug mode to see extra information
)

# Generate a random key
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, n_items)  # Pre-split keys for the loop

# Reset the environment
state, timestep = env.reset(keys[0])

# Get the number of values for the action spec outside the jitted function
# Convert to Python integers using .item()
num_orientations, num_ems, num_items = [x.item() for x in env.action_spec().num_values]

# Function to select a random action
def select_action(
    key: jax.Array, observation: Observation, num_orientations: int, num_ems: int, num_items: int
) -> jax.Array:
    """Randomly sample valid actions, as determined by `observation.action_mask`."""
    flat_action_mask = observation.action_mask.flatten()

    # Create a probability array where valid actions have equal probability and invalid actions have 0 probability
    probabilities = jnp.where(flat_action_mask, 1 / jnp.sum(flat_action_mask), 0)

    # Choose an action index based on the probabilities
    chosen_action_flat = jax.random.choice(
        key, jnp.arange(flat_action_mask.shape[0]), p=probabilities
    )
    # Reverse the flattening process
    orientation_ems_id, item_id = jnp.divmod(chosen_action_flat, num_items)
    orientation, ems_id = jnp.divmod(orientation_ems_id, num_ems)
    action = jnp.array([orientation, ems_id, item_id], jnp.int32)
    # action[0]: orientation of the item to place (0 to 5)
    # action[1]: EMS ID where to place the item (0 to obs_num_ems - 1) - Index of the EMS in the *observed, sorted EMS list* (largest EMS is index 0)
    # action[2]: Item ID to place (0 to max_num_items - 1)
    return action

# Jit the step function and the action selection function
step_fn = jax.jit(env.step)
# select_action_fn = jax.jit(select_action, static_argnums=(2, 3, 4))
select_action_fn = select_action

# Initialize step count, key index, and state history list
step_count = 0
key_index = 1 # Start from the second key as the first one was used for reset
state_history: List[State] = [state] # Initialize with the reset state

# Regular Python while loop
while True:
    if not jnp.any(timestep.observation.action_mask) or timestep.last():
        print("Episode finished.")
        break

    action_key = keys[key_index] # Use pre-split keys
    action = select_action_fn(action_key, timestep.observation, num_orientations, num_ems, num_items)
    next_state, next_timestep = step_fn(state, action)

    # Print information for each step
    print(f"Step: {step_count}")
    print(f"  Action: {action} ", end="") # action[0]: orientation, action[1]: ems_id, action[2]: item_id
    print(f"(orientation={action[0]}, ems_id={action[1]}, item_id={action[2]})")
    print(f"  Reward: {next_timestep.reward}")
    print(f"  Discount: {next_timestep.discount}")
    print(f"  Placed items: {jnp.sum(next_state.items_placed)}")
    print(
      f"  Volume utilization: {next_timestep.extras['volume_utilization'] if next_timestep.extras else None}"
    )
    print(
      f"  Invalid action: {next_timestep.extras['invalid_action'] if next_timestep.extras else None}"
    )
    if next_timestep.extras and 'invalid_ems_from_env' in next_timestep.extras:
        invalid_ems_detected = jnp.any(next_timestep.extras['invalid_ems_from_env']).item() # Check if any invalid EMS exists
        print(f"  Invalid EMS from Env: {invalid_ems_detected}") # Print boolean if any invalid EMS

    # Update state and timestep and append new state to history
    state, timestep = next_state, next_timestep
    state_history.append(state) # Store the state after each step
    step_count += 1
    key_index += 1
    if key_index >= len(keys): # Prevent index out of bound in case episode is very long
        print("Ran out of keys, stopping episode prematurely.")
        break

# Animate the trajectory and save as GIF
# env.animate(states=state_history, save_path="binpack_trajectory.gif")
# print(f"Animation saved to binpack_trajectory.gif")