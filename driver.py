"""
Torch driver

This dreiver uses a deep neural network model designed for driving decisions based on the current state of the world,
It is as good as the trained models it uses.

This driver requires PyTorch to be installed, see: https://pytorch.org/get-started/locally/

Trained models are expected to be in the checkpoints directory.
"""

import os

try:
    import torch
except ImportError:
    print("Error: torch module not found. Please install it before proceeding.")
    print("       see: https://pytorch.org/get-started/locally/")
    exit()

import torch.nn as nn
import torch.nn.functional as F


"""
Torch Car

This is a driver choosing the next action by consolting a neural network.
"""
driver_name = "Torch Car"


# Game classes
# ----------------------------------------------------------------------------------


class actions:
    NONE = "none"
    RIGHT = "right"
    LEFT = "left"
    PICKUP = "pickup"
    JUMP = "jump"
    BRAKE = "brake"

    ALL = (NONE, RIGHT, LEFT, PICKUP, JUMP, BRAKE)


class obstacles:
    NONE = ""
    CRACK = "crack"
    TRASH = "trash"
    PENGUIN = "penguin"
    BIKE = "bike"
    WATER = "water"
    BARRIER = "barrier"

    ALL = (NONE, CRACK, TRASH, PENGUIN, BIKE, WATER, BARRIER)


# PyTorch NN driving model
# ----------------------------------------------------------------------------------
class DriverModel(nn.Module):
    """
    A deep neural network model designed for driving decisions based on the current state of the world.

    Architecture:
    - Input layer: Size determined by 3 (width) x 4 (height) x 7 (possible obstacles) = 84 neurons.
    - Hidden Layer 1: 512 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 2: 256 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 3: 128 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 4: 64 neurons, followed by batch normalization and 50% dropout.
    - Hidden Layer 5: 32 neurons, followed by batch normalization and 50% dropout.
    - Output Layer: 3 neurons, representing the possible driving decisions (e.g., left, forward, right).

    Activation Function:
    - ReLU activation function is used for all hidden layers.
    - The output layer does not have an activation function, making this model suitable for use with a softmax function externally or a criterion like CrossEntropyLoss which combines softmax and NLLLoss.

    Regularization:
    - Dropout with a rate of 50% is applied after each hidden layer to prevent overfitting.

    Note:
    - The model expects a flattened version of the 3x4x7 input tensor, which should be reshaped to (batch_size, 84) before being passed to the model.
    """
    def __init__(self):
        super(DriverModel, self).__init__()

        self.fc1 = nn.Linear(3 * 4 * 7, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(0.5)

        self.fc6 = nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        x = F.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)

        x = self.fc6(x)
        return x


# Get some trained models
# ----------------------------------------------------------------------------------

# The checkpoints directory include trainde models
#       Note: car on right is mirror of left, no need for a new model
script_dir = os.path.dirname(os.path.abspath(__file__))

# Model trained when car on left (x=0)
checkpoint_in_x0 = os.path.join(script_dir, "checkpoints", "driver-x0.pth")
# Model trained when car in the middle (x=1)
checkpoint_in_x1 = os.path.join(script_dir, "checkpoints", "driver-x1.pth")


model_x0 = DriverModel()
model_x0.load_state_dict(torch.load(checkpoint_in_x0))
model_x0.eval()

model_x1 = DriverModel()
model_x1.load_state_dict(torch.load(checkpoint_in_x1))
model_x1.eval()


def view_to_inputs(array):
    """
    Convert a 2D array representation of the world into a tensor suitable for model input.

    The function maps each obstacle in the 2D array to a one-hot encoded tensor.
    The resulting tensor is then flattened and an additional dimension is added to represent the batch size.

    Args:
        array (list[list[str]]): 2D array representation of the world with obstacles as strings.

    Returns:
        torch.Tensor: A tensor of shape (1, height * width * num_obstacle_types) suitable for model input.

    Notes:
        The function uses a predefined mapping of obstacles to indices for the one-hot encoding.
    """
    OBSTACLE_TO_INDEX = {
        obstacle: index for index, obstacle in enumerate(obstacles.ALL)
    }

    height = len(array)
    width = len(array[0])
    tensor = torch.zeros((height, width, 7))

    for i in range(height):
        for j in range(width):
            obstacle = array[i][j]
            tensor[i, j, OBSTACLE_TO_INDEX[obstacle]] = 1

    return tensor.view(-1).unsqueeze(0)


def outputs_to_action(output, world):
    """
    Convert the model's output tensor into a driving action based on the current state of the world.

    The function first determines the car's intended position (left, forward, right) based on the model's output.
    If the position is forward, the function checks the obstacle in front of the car and maps it to an appropriate action.

    Args:
        output (torch.Tensor): The model's output tensor, typically representing probabilities for each position.
        world (World): An instance of the World class providing read-only access to the current game state.

    Returns:
        str: The determined action for the car to take. Possible actions include those defined in the `actions` class.

    Notes:
        The function uses a predefined mapping of obstacles to actions to determine the appropriate action when moving forward.
    """
    positions = ["left", "forward", "right"]
    obstacle_action_map = {
        obstacles.PENGUIN: actions.PICKUP,
        obstacles.CRACK: actions.JUMP,
        obstacles.WATER: actions.BRAKE,
    }

    obstacle = world.get((world.car.x, world.car.y - 1))
    position_index = torch.argmax(output).item()
    position = positions[position_index]

    if position == "left":
        return actions.LEFT
    elif position == "right":
        return actions.RIGHT
    else:
        return obstacle_action_map.get(obstacle, actions.NONE)


# Drive
# ----------------------------------------------------------------------------------


def build_lane_view(world, height, lane, flip=False):
    """
    Build a 3xN 2D array representation of the world based on the car's lane and x position.

    Args:
        world (World): An instance of the World class providing read-only
                       access to the current game state.
        height (int): The height of the returned 2D array.
        lane (int) : The car's current lane, 0 or 1. This determines the starting x-coordinate for the 2D array.
        flip (bool, optional): If True, columns 0 and 2 will be flipped. Defaults to False.

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

    # Determine the column order based on the flip argument
    column_order = [2, 1, 0] if flip else [0, 1, 2]

    # Generate the 2D array from start_y up to world.car.y
    array = [
        [get_value(j + start_x, i) for j in column_order] for i in range(start_y, car_y)
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
    flip_world = x_in_lane == 2

    view = build_lane_view(world, view_height, lane, flip_world)

    input_tensor = view_to_inputs(view)

    if x_in_lane == 1:
        output = model_x1(input_tensor)
    else:
        output = model_x0(input_tensor)

    action = outputs_to_action(output, world)

    if flip_world:
        action_flip = {actions.RIGHT: actions.LEFT, actions.LEFT: actions.RIGHT}
        action = action_flip.get(action, action)

    return action
