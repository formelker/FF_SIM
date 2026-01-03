__author__ = ["Ali Awada", "Melker Forslund"]

#imports

# local imports
from parameters import *

# cell states
UNBURNED = 0
BURNING = 1
LIT_OUT = 2
BURNED = 3

class Cell:
    def __init__(self, fuel, temperature=AMBIENT_TEMPERATURE, moisture=None):
        self.fuel = fuel                        # amount of fuel 0-1
        self.moisture = moisture                # moisture content 0-1
        self.temperature = temperature          # temperature in kelvin
        self.state = UNBURNED                   # cell state
        self.wind_speed = None                  # wind speed in m/s
        self.wind_direction = None              # wind direction in degrees
        self.wind_u = None                      # wind u component
        self.wind_v = None                      # wind v component
