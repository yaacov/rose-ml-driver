"""
Torch driver training

This script is used to train a deep learning model for a driving simulator. The model is trained using PyTorch.

The script generates a 4x3 2D array with random obstacles and simulates the driver's decision based on the obstacle in front of the car. 
The driver's decision and the 2D array are used to generate a batch of samples for training. 

The model is trained for a specified number of epochs. In each epoch, the model is trained over a number of batches. 
For each batch, the model's parameters are updated based on the computed loss between the model's predictions and the actual targets.

The script requires PyTorch to be installed. See: https://pytorch.org/get-started/locally/

The trained model is saved in the checkpoints directory.

Usage:
    python train.py
"""

try:
    import torch
except ImportError:
    print("Error: torch module not found. Please install it before proceeding.")
    print("       see: https://pytorch.org/get-started/locally/")
    exit()

import random
import torch.nn as nn
import torch.optim as optim

from driver import DriverModel, action_to_outputs, actions, obstacles, view_to_inputs

OBSTACLE_TO_INDEX = {
    "": 0,
    "crack": 1,
    "trash": 2,
    "penguin": 3,
    "bike": 4,
    "water": 5,
    "barrier": 6,
}

# Car lane, 0 - left, 1 - middle
car_x = 1

# checkpoint_in - if not empty, load this model before starting the training
# checkpoint_out - save trained model as
checkpoint_in = ""
checkpoint_out = f"driver-x{car_x}.pth"

# Training parameters
num_epochs = 10
batch_size = 200
learning_rate = 0.001


# Create model, loss function, and optimizer
model = DriverModel()

# Read starting checkpoint, if available
if checkpoint_in != "":
    model.load_state_dict(torch.load(checkpoint_in))
    model.eval()

# Set loss function and backprpogation method
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def generate_obstacle_array():
    """
    Generates a 4x3 2D array with random obstacles.

    Returns:
        list[list[str]]: 4x3 2D array with random obstacles.
    """
    array = [["" for _ in range(3)] for _ in range(4)]

    for i in range(4):
        obstacle = random.choice(list(OBSTACLE_TO_INDEX.keys()))
        position = random.randint(0, 2)
        array[i][position] = obstacle

    return array


def driver_simulator(array, car_x):
    """
    Simulates the driver's decision based on the obstacle in front of the car.

    Args:
        array (list[list[str]]): 2D array representation of the world with obstacles as strings.
        car_x (int): The car's x position.

    Returns:
        str: The determined action for the car to take. Possible actions include those defined in the `actions` class.
    """
    obstacle = array[3][car_x]

    if obstacle == obstacles.PENGUIN:
        return actions.PICKUP
    elif obstacle == obstacles.WATER:
        return actions.BRAKE
    elif obstacle == obstacles.CRACK:
        return actions.JUMP
    elif obstacle == obstacles.NONE:
        return actions.NONE
    else:
        return actions.RIGHT if (car_x % 3) == 0 else actions.LEFT


def generate_batch(batch_size):
    """
    Generates a batch of samples for training.

    Args:
        batch_size (int): The number of samples in the batch.

    Returns:
        tuple: A tuple containing two tensors. The first tensor contains the inputs for the model, and the second tensor contains the target outputs.
    """
    inputs = []
    targets = []
    for _ in range(batch_size):
        array = generate_obstacle_array()
        correct_output = driver_simulator(array, car_x)

        input_tensor = view_to_inputs(array)
        target_tensor = action_to_outputs(correct_output)

        inputs.append(input_tensor)
        targets.append(target_tensor)
    return torch.stack(inputs), torch.stack(targets)


# Training loop
for epoch in range(num_epochs):
    # Initialize running loss to 0.0 at the start of each epoch
    running_loss = 0.0

    # Assuming you have a dataset size, calculate the number of batches
    num_batches = 100

    # Loop over each batch
    for i in range(num_batches):
        # Get a batch of training data
        inputs, targets = generate_batch(batch_size)

        # Reset the gradients in the optimizer (i.e., make it forget the gradients computed in the previous iteration)
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(inputs)

        # Compute loss: calculate the batch loss based on the difference between the predicted outputs and the actual targets
        loss = criterion(outputs, targets)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {running_loss / num_batches}")

torch.save(model.state_dict(), checkpoint_out)

print("Finished Training")
