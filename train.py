"""
Torch driver training

This driver requires PyTorch to be installed, see: https://pytorch.org/get-started/locally/

Trained models are expected to be in the checkpoints directory.
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

def generate_obstacle_array():
    array = [["" for _ in range(3)] for _ in range(4)]

    for i in range(4):
        obstacle = random.choice(list(OBSTACLE_TO_INDEX.keys()))
        position = random.randint(0, 2)
        array[i][position] = obstacle

    return array

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

if checkpoint_in != "":
    model.load_state_dict(torch.load(checkpoint_in))
    model.eval()

# Set loss function and backprpogation method
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def driver_simulator(array, car_x):
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

# Helper function to generate a batch of samples
def generate_batch(batch_size):
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
    running_loss = 0.0

    # Assuming you have a dataset size, calculate the number of batches
    num_batches = 100

    for i in range(num_batches):
        # Get training data
        inputs, targets = generate_batch(batch_size)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Loss: {running_loss / num_batches}")

torch.save(model.state_dict(), checkpoint_out)

print("Finished Training")
