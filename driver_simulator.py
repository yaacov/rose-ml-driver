from model import actions, obstacles

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
    elif car_x == 0 or car_x == 3:
        return actions.RIGHT
    elif car_x == 2 or car_x == 5:
        return actions.LEFT
    else:
        return actions.RIGHT