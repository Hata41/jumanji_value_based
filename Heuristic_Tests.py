import jax
import jax.numpy as jnp
from jumanji.environments.packing.bin_pack.env import ExtendedBinPack
from jumanji.environments.packing.bin_pack.generator import (
    ExtendedRandomGenerator,
)
from jumanji.environments.packing.bin_pack.types import Observation, State, item_volume
import functools
from typing import Tuple, List, Dict

# Define enum class with action selction type : 
# RANDOM, FIRST_FIT, FIRST_FIT_DECREASING_ITEM_FIRST, FIRST_FIT_DECREASING_EMS_FIRST
class ActionSelectionType:
    RANDOM = "random"
    FIRST_FIT_DECREASING_ITEM_FIRST = "first_fit_decreasing_item_first"
    FIRST_FIT_DECREASING_EMS_FIRST = "first_fit_decreasing_ems_first"
    
ACTION_TYPE = ActionSelectionType.FIRST_FIT_DECREASING_EMS_FIRST

# Define the function to select the first valid action based on the First Fit Decreasing heuristic
n_items = 200
max_num_ems = 5 * n_items

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
    obs_num_ems=max_num_ems,  # Use all EMS for first-fit, not just the observed ones
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

actual_nb_items = state.nb_items

# Get the number of values for the action spec outside the jitted function
# Convert to Python integers using .item()
num_orientations, num_ems, num_items = [x.item() for x in env.action_spec().num_values]

# Function to select a random action
def select_random_action(
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

def select_action_first_fit_decreasing(observation: Observation, num_orientations: int, num_ems: int, num_items: int, mode: str = "item_first"
) -> jax.Array:
    """Selects the first valid action based on First Fit Decreasing heuristic.
    Sorts items in decreasing order of volume, then tries to place them in EMSs.

    Args:
        observation: The current observation.
        num_orientations: Number of orientations.
        num_ems: Number of EMSs.
        num_items: Number of items.
        mode: "item_first" or "ems_first" to choose the iteration order.
    """
    item_vols = item_volume(observation.items)
    flat_item_vols = item_vols.flatten()
    sorted_item_indices = jnp.argsort(flat_item_vols)[::-1]  # Decreasing volume order

    reshaped_action_mask = observation.action_mask.reshape(
        num_orientations, num_ems, num_items
    )

    default_action = jnp.array([-1, -1, -1], jnp.int32) # Default invalid action

    def item_first_scan_body(carry: Tuple[jax.Array, bool], item_flat_index: int) -> Tuple[Tuple[jax.Array, bool], None]:
        action_found, action = carry
        item_id = item_flat_index % num_items # Get original item index
        orientation_ems_id = item_flat_index // num_items
        orientation = orientation_ems_id % num_orientations

        def ems_scan_body(carry_ems: Tuple[jax.Array, bool], ems_id: int) -> Tuple[Tuple[jax.Array, bool], None]:
            action_found_ems, action_ems = carry_ems
            current_action = jnp.array([orientation, ems_id, item_id], jnp.int32)
            action_ems_updated = jax.lax.cond(
                (jnp.logical_not(action_found_ems) & reshaped_action_mask[orientation, ems_id, item_id]),
                lambda: current_action,
                lambda: action_ems # Corrected: return action_ems (previous action)
            )
            action_found_ems_updated = jax.lax.cond(
                (jnp.logical_not(action_found_ems) & reshaped_action_mask[orientation, ems_id, item_id]),
                lambda: True,
                lambda: action_found_ems # Corrected: return action_found_ems (previous action_found)
            )
            return (action_found_ems_updated, action_ems_updated), None

        initial_carry_ems = (action_found, action) # Initialize with outer carry state
        (action_found_item, action_item), _ = jax.lax.scan(
            ems_scan_body, initial_carry_ems, jnp.arange(num_ems)
        )
        return (action_found_item, action_item), None


    def ems_first_scan_body(carry: Tuple[jax.Array, bool], ems_id: int) -> Tuple[Tuple[jax.Array, bool], None]:
        action_found, action = carry

        def item_scan_body(carry_item: Tuple[jax.Array, bool], item_flat_index: int) -> Tuple[Tuple[jax.Array, bool], None]:
            action_found_item, action_item = carry_item
            item_id = item_flat_index % num_items # Get original item index
            orientation_ems_id = item_flat_index // num_items
            orientation = orientation_ems_id % num_orientations

            current_action = jnp.array([orientation, ems_id, item_id], jnp.int32)
            action_item_updated = jax.lax.cond(
                (jnp.logical_not(action_found_item) & reshaped_action_mask[orientation, ems_id, item_id]),
                lambda: current_action,
                lambda: action_item # Corrected: return action_item (previous action)
            )
            action_found_item_updated = jax.lax.cond(
                (jnp.logical_not(action_found_item) & reshaped_action_mask[orientation, ems_id, item_id]),
                lambda: True,
                lambda: action_found_item # Corrected: return action_found_item (previous action_found)
            )
            return (action_found_item_updated, action_item_updated), None


        initial_carry_item = (action_found, action) # Initialize with outer carry state
        (action_found_ems, action_ems), _ = jax.lax.scan(
            item_scan_body, initial_carry_item, sorted_item_indices
        )
        return (action_found_ems, action_ems), None


    initial_carry = (False, default_action)

    if mode == "item_first":
        (action_found, action), _ = jax.lax.scan(item_first_scan_body, initial_carry, sorted_item_indices)
    elif mode == "ems_first":
        (action_found, action), _ = jax.lax.scan(ems_first_scan_body, initial_carry, jnp.arange(num_ems))
    else:
        raise ValueError(f"Invalid mode: {mode}. Mode must be 'item_first' or 'ems_first'.")

    return action

# Jit the step function and the action selection functions
step_fn = jax.jit(env.step)

select_random_action_fn = jax.jit(select_random_action, static_argnums=(2, 3, 4)) # random action

select_action_first_fit_decreasing_item_first_fn = jax.jit(functools.partial(
    select_action_first_fit_decreasing, 
    mode="item_first"), 
    static_argnums=(1, 2, 3)) # first fit decreasing action item_first mode

select_action_first_fit_decreasing_ems_first_fn = jax.jit(functools.partial(
    select_action_first_fit_decreasing,
    mode="ems_first"), 
    static_argnums=(1, 2, 3)) # first fit decreasing action ems_first mode


# Initialize step count, key index, and state history list
step_count = 0
key_index = 1 # Start from the second key as the first one was used for reset
state_history: List[State] = [state] # Initialize with the reset state

if ACTION_TYPE == ActionSelectionType.RANDOM:
    action_selection_fn = select_random_action_fn
elif ACTION_TYPE == ActionSelectionType.FIRST_FIT_DECREASING_ITEM_FIRST:
    action_selection_fn = lambda action_key, observation, num_orientations, num_ems, num_items: select_action_first_fit_decreasing_item_first_fn(observation, num_orientations, num_ems, num_items)    
elif ACTION_TYPE == ActionSelectionType.FIRST_FIT_DECREASING_EMS_FIRST:
    action_selection_fn = lambda action_key, observation, num_orientations, num_ems, num_items: select_action_first_fit_decreasing_ems_first_fn(observation, num_orientations, num_ems, num_items)


# Regular Python while loop
while True:
    if not jnp.any(timestep.observation.action_mask) or timestep.last():
        print("Episode finished.")
        break

    action_key = keys[key_index] # Use pre-split keys
    
    # Select action
    action = action_selection_fn(action_key, timestep.observation, num_orientations, num_ems, num_items) 

    # Step in environnement
    next_state, next_timestep = step_fn(state, action)

    # Print information for each step
    # print(f"Step: {step_count}")
    # print(f"  Action: {action} ", end="") # action[0]: orientation, action[1]: ems_id, action[2]: item_id
    # print(f"(orientation={action[0]}, ems_id={action[1]}, item_id={action[2]})")
    # print(f"  Reward: {next_timestep.reward}")
    # print(f"  Discount: {next_timestep.discount}")
    # print(f"  Placed items: {jnp.sum(next_state.items_placed)}")
    # print(
    #   f"  Volume utilization: {next_timestep.extras['volume_utilization'] if next_timestep.extras else None}"
    # )
    # print(
    #   f"  Invalid action: {next_timestep.extras['invalid_action'] if next_timestep.extras else None}"
    # )
    if next_timestep.extras and 'invalid_ems_from_env' in next_timestep.extras:
        invalid_ems_detected = jnp.any(next_timestep.extras['invalid_ems_from_env']).item() # Check if any invalid EMS exists
        # print(f"  Invalid EMS from Env: {invalid_ems_detected}") # Print boolean if any invalid EMS

    # Update state and timestep and append new state to history
    state, timestep = next_state, next_timestep
    state_history.append(state) # Store the state after each step
    step_count += 1
    key_index += 1
    
print(f"Total steps: {step_count}/{actual_nb_items} c.a.d {step_count/actual_nb_items:.2f} packed ratio")
print(f"Used action selection type: {ACTION_TYPE}")


if ACTION_TYPE == ActionSelectionType.RANDOM:
    gif_save_path = f"random_{num_items}.gif"
elif ACTION_TYPE == ActionSelectionType.FIRST_FIT_DECREASING_ITEM_FIRST:
    gif_save_path = f"item_first_{num_items}.gif"
elif ACTION_TYPE == ActionSelectionType.FIRST_FIT_DECREASING_EMS_FIRST:
    gif_save_path = f"ems_first_{num_items}.gif"

# Animate the trajectory and save as GIF
env.animate(states=state_history, save_path=gif_save_path)
