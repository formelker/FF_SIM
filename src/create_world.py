__author__ = ["Ali Awada", "Melker Forslund"]
"""
This module creates a world and saves it to a file
"""
# imports
import os
import random
import numpy as np
import pickle

# local imports
from parameters import WORLD_SIZE, WORLD_FILE_NAME
from world import World, create_fuel_map_gaussian

def save_world(world, filename):
    with open(filename, "wb") as f:
        pickle.dump(world, f)

def load_world(filename):
    with open(filename, "rb") as f:
        world = pickle.load(f)
    return world

if __name__ == "__main__":
    mean_fuel = 1.0
    variation = 0.0
    fuel_map = create_fuel_map_gaussian(WORLD_SIZE, mean_fuel=mean_fuel, variation=variation)
    world = World(WORLD_SIZE, fuel_map=fuel_map)   
    print("World of size ", WORLD_SIZE, " created.")
    # save the world
    save_world(world, WORLD_FILE_NAME)
    print(f"World saved to {WORLD_FILE_NAME}")