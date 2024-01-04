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
import argparse
import torch.nn as nn
import torch.optim as optim

from model import DriverModel, action_to_outputs, actions, obstacles, view_to_inputs

OBSTACLE_TO_INDEX = {
    "": 0,
    "crack": 1,
    "trash": 2,
    "penguin": 3,
    "bike": 4,
    "water": 5,
    "barrier": 6,
}

# Training parameters
num_epochs = 0
batch_size = 0
learning_rate = 0

# Set loss function and backprpogation method
criterion = None
optimizer = None

# Create model, loss function, and optimizer
model = DriverModel()


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
        car_x = random.choice([0,1,2])
        array = generate_obstacle_array()
        correct_output = driver_simulator(array, car_x)

        input_tensor = view_to_inputs(array, car_x)
        target_tensor = action_to_outputs(correct_output)

        inputs.append(input_tensor)
        targets.append(target_tensor)
    return torch.stack(inputs), torch.stack(targets)


# Training loop
def main():
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--checkpoint-in', default="", help='Path to the input checkpoint file.')
    parser.add_argument('--checkpoint-out', default="", help='Path to the output checkpoint file.')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--batch-size', type=int, default=200, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for training.')
    args = parser.parse_args()

    # Training parameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    # Set loss function and backprpogation method
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Read starting checkpoint, if available
    if args.checkpoint_in != "":
        model.load_state_dict(torch.load(args.checkpoint_in))
        model.eval()
    
    # Run training
    main()
    print("Finished Training")

    torch.save(model.state_dict(), args.checkpoint_out or f"driver.pth")
    