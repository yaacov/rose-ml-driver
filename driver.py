"""
Torch driver

This dreiver uses a deep neural network model designed for driving decisions based on the current state of the world,
It is as good as the trained models it uses.

This driver requires PyTorch to be installed, see: https://pytorch.org/get-started/locally/

Trained models are expected to be in the checkpoints directory.
"""

import os
import sys

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to the system path
sys.path.append(script_directory)

# Try to import pytorch
try:
    import torch
except ImportError:
    print("Error: torch module not found. Please install it before proceeding.")
    print("       see: https://pytorch.org/get-started/locally/")
    exit()

from model import DriverModel, outputs_to_action, view_to_inputs

"""
Torch Car

This is a driver choosing the next action by consolting a neural network.
"""
driver_name = "Torch Car"


# Get the trained model
# ----------------------------------------------------------------------------------


# Model trained using a simulator
checkpoint = os.path.join(script_directory, "checkpoints", "driver.pth")

model = DriverModel()
model.load_state_dict(torch.load(checkpoint))
model.eval()


# Drive
# ----------------------------------------------------------------------------------


def build_lane_view(world, height, lane):
    """
    Build a 3xN 2D array representation of the world based on the car's lane and x position.

    Args:
        world (World): An instance of the World class providing read-only
                       access to the current game state.
        height (int): The height of the returned 2D array.
        lane (int) : The car's current lane, 0 or 1. This determines the starting x-coordinate for the 2D array.

    Returns:
        list[list[str]]: 3xN array representation of the world view from the car, where N is the specified height.
                          The bottom line is one line above the car's y position, and the top line is the line height lines above that.
                          The array provides a view of the world from the car's perspective, with the car's y position excluded.

    Notes:
        The function uses the car's y position to determine the vertical range of the 2D array.
        The starting x-coordinate is determined by the car's lane. If the lane is 0, the starting x is 0. If the lane is 1, the starting x is 3.
        The function also provides a wrapper around world.get to handle negative y values, returning an empty string for such cases.
    """
    car_y = world.car.y

    # Determine the starting x-coordinate for the 2D array based on the car's x position
    start_x = 0 if lane == 0 else 3

    # Calculate the starting y-coordinate based on the car's y position and the desired height
    start_y = car_y - height

    # Wrapper around world.get to handle negative y values
    def get_value(j, i):
        if i < 0:
            return ""
        return world.get((j, i))

    # Generate the 2D array from start_y up to world.car.y
    array = [
        [get_value(j + start_x, i) for j in [0, 1, 2]] for i in range(start_y, car_y)
    ]

    return array


def drive(world):
    """
    Determine the appropriate driving action based on the current state of the world.

    The function first constructs a 3xN 2D view of the world based on the car's position.
    This view is then converted to an input tensor format suitable for the model.
    Depending on the car's x position within its lane, the function uses one of two models (`model_x1` or `model_x0`)
    to predict the best action. If the world view was flipped (because the car is on the rightmost side of its lane),
    the action might be flipped back (e.g., from LEFT to RIGHT or vice versa).

    Args:
        world (World): An instance of the World class providing read-only access to the current game state.

    Returns:
        str: The determined action for the car to take. Possible actions include those defined in the `actions` class.

    Notes:
        The function uses two models (`model_x1` and `model_x0`) to predict actions based on the car's x position within its lane.
        The `flip_world` flag determines if the world view was flipped horizontally, which affects the final action decision.
    """
    view_height = 4

    lane = 0 if world.car.x < 3 else 1
    x_in_lane = world.car.x % 3

    # Convert real world input, into a tensor
    view = build_lane_view(world, view_height, lane)
    input_tensor = view_to_inputs(view, x_in_lane).unsqueeze(0)

    # Use neural network model to get the outputs tensor
    output = model(input_tensor)

    # Convert the output tensor into a real world response
    action = outputs_to_action(output, world)

    return action
