__author__ = ["Ali Awada", "Melker Forslund"]

#imports 
import numpy as np

#local imports
from parameters import *

from cell import Cell, UNBURNED, BURNING, LIT_OUT, BURNED
from visualize_world import FUEL_RANGE, MOISTURE_RANGE

class World:
    """
    A world is a grid of cells - optimized with direct numpy array storage
    """
    def __init__(self, size, fuel_map=None):
        self.size = size
        self.fuel_map = fuel_map
        # Store background wind for relaxation
        self.vu_background = None
        self.vv_background = None

        # Direct numpy array storage for fast access. Initialized later for faster saving and simulation
        self._temperature = np.full((size, size), AMBIENT_TEMPERATURE, dtype=np.float64)
        self._fuel = fuel_map.copy() if fuel_map is not None else np.zeros((size, size), dtype=np.float64)
        self._moisture = np.zeros((size, size), dtype=np.float64)
        self._wind_u = np.zeros((size, size), dtype=np.float64)
        self._wind_v = np.zeros((size, size), dtype=np.float64)
        self._states = np.full((size, size), UNBURNED, dtype=np.int32)

        # Keep grid for backward compatibility but it's not used in hot paths
        self.grid = np.empty((size, size), dtype=object)
        self.create_world()

    def _ensure_arrays_initialized(self):
        """Initialize arrays from grid if they don't exist (backward compatibility for pickled worlds)"""
        if not hasattr(self, '_temperature'):
            self._temperature = np.zeros((self.size, self.size), dtype=np.float64)
            self._fuel = np.zeros((self.size, self.size), dtype=np.float64)
            self._moisture = np.zeros((self.size, self.size), dtype=np.float64)
            self._wind_u = np.zeros((self.size, self.size), dtype=np.float64)
            self._wind_v = np.zeros((self.size, self.size), dtype=np.float64)
            self._states = np.zeros((self.size, self.size), dtype=np.int32)
            # Copy from grid cells
            for i in range(self.size):
                for j in range(self.size):
                    cell = self.grid[i, j]
                    self._temperature[i, j] = cell.temperature
                    self._fuel[i, j] = cell.fuel
                    self._moisture[i, j] = cell.moisture if cell.moisture is not None else 0.0
                    self._wind_u[i, j] = cell.wind_u if cell.wind_u is not None else 0.0
                    self._wind_v[i, j] = cell.wind_v if cell.wind_v is not None else 0.0
                    self._states[i, j] = cell.state

    def create_world(self):
        """Creates a world with random cells (for backward compatibility)"""
        for i in range(self.size):
            for j in range(self.size):
                fuel = self.fuel_map[i, j]
                cell = Cell(fuel)
                self.grid[i, j] = cell

    def set_moisture_map(self, moisture_map):
        self._ensure_arrays_initialized()
        self._moisture = moisture_map.astype(np.float64)
        # Update grid for backward compatibility
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i, j].moisture = moisture_map[i, j]

    def set_wind_map(self, wind_map):
        self._ensure_arrays_initialized()
        self._wind_u = wind_map["vu"].astype(np.float64)
        self._wind_v = wind_map["vv"].astype(np.float64)
        # Store background wind for relaxation
        self.vu_background = wind_map["vu_background"]
        self.vv_background = wind_map["vv_background"]
        # Update grid for backward compatibility
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i, j].wind_speed = wind_map["wind_speed"][i, j]
                self.grid[i, j].wind_direction = wind_map["wind_direction"][i, j]
                self.grid[i, j].wind_u = wind_map["vu"][i, j]
                self.grid[i, j].wind_v = wind_map["vv"][i, j]

    def get_temperature_array(self):
        self._ensure_arrays_initialized()
        return self._temperature

    def get_fuel_array(self):
        self._ensure_arrays_initialized()
        return self._fuel

    def get_moisture_array(self):
        self._ensure_arrays_initialized()
        return self._moisture

    def get_wind_arrays(self):
        self._ensure_arrays_initialized()
        return self._wind_u, self._wind_v

    def get_state_array(self):
        self._ensure_arrays_initialized()
        return self._states.astype(np.float64)

    def update_from_arrays(self, temperature, fuel, moisture, vu, vv, states):
        self._ensure_arrays_initialized()
        # Direct array assignment, copied before function call to avoid reference issues
        self._temperature = temperature
        self._fuel = fuel
        self._moisture = moisture
        self._wind_u = vu
        self._wind_v = vv
        self._states = states.astype(np.int32)

    def set_ignition_point(self, x, y):
        """
        Set the ignition point at coordinates (x, y)
        """
        self._ensure_arrays_initialized()
        self._temperature[x, y] = IGNITION_TEMP
        self._moisture[x, y] = IGNITION_MOISTURE_LEVEL                  
        self._states[x, y] = BURNING     
        # Also update grid for backward compatibility
        self.grid[x, y].temperature = IGNITION_TEMP
        self.grid[x, y].moisture = IGNITION_MOISTURE_LEVEL
        self.grid[x, y].state = BURNING

    def create_snapshot(self):
        """
        Create a snapshot of the world state for saving
        """
        self._ensure_arrays_initialized()
        return WorldSnapshot(
            size=self.size,
            temperature=self._temperature.copy(),
            fuel=self._fuel.copy(),
            moisture=self._moisture.copy(),
            wind_u=self._wind_u.copy(),
            wind_v=self._wind_v.copy(),
            states=self._states.copy()
        )

# WorldSnapshot class for light and fast storage of world states
class WorldSnapshot:
    """
    Lightweight snapshot of world states.
    Stores only arrays, not Cell objects.
    """
    def __init__(self, size, temperature, fuel, moisture, wind_u, wind_v, states):
        self.size = size
        self._temperature = temperature
        self._fuel = fuel
        self._moisture = moisture
        self._wind_u = wind_u
        self._wind_v = wind_v
        self._states = states
    
    def get_temperature_array(self):
        return self._temperature
    
    def get_fuel_array(self):
        return self._fuel
    
    def get_moisture_array(self):
        return self._moisture
    
    def get_wind_arrays(self):
        return self._wind_u, self._wind_v
    
    def get_state_array(self):
        return self._states

def create_moisture_map_gaussian(size, moisture_level=0.2, variation=0.05):
    """
    Using gaussian distribution to create a moisture map
    """
    moisture_values = np.random.normal(moisture_level, variation, (size, size))
    moisture_values = np.clip(moisture_values, MOISTURE_RANGE[0], MOISTURE_RANGE[1])
    return moisture_values

def create_fuel_map_gaussian(size, mean_fuel=0.6, variation=0.1):
    """
    Using gaussian distribution to create a fuel map
    """
    fuel_values = np.random.normal(mean_fuel, variation, (size, size))
    fuel_values = np.clip(fuel_values, FUEL_RANGE[0], FUEL_RANGE[1])
    return fuel_values

def create_wind_field_map(size, speed_mean, speed_variation, direction_degrees, direction_variation):

    # wind speed with variation m/s
    wind_speed = np.random.normal(speed_mean, speed_variation, (size, size))
    wind_speed = np.clip(wind_speed, 0.0, None) 
    
    # wind direction with variation
    wind_direction = np.random.normal(direction_degrees, direction_variation, (size, size))

    # convert degrees to radians
    theta = np.deg2rad(wind_direction)

    vu = wind_speed * np.sin(theta)  
    vv = -wind_speed * np.cos(theta) 

    return {"wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "vu": vu,
            "vv": vv,
            "vu_background": vu.copy(),
            "vv_background": vv.copy()}