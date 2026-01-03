__author__ = ["Ali Awada", "Melker Forslund"]

"""
Wildfire Simulation Parameters
"""
import os

# ============================================================================
# SIMULATION STEP AND TIME CONTROL
# ============================================================================
DT = 1.0                                    # seconds - time step
sim_time = 30                               # minutes - simulation time
NUMBER_OF_STEPS = int(sim_time * 60 / DT)   # number of time steps, if "None" then simulate until fire dies out
 
# ============================================================================
# FILE PATHS
# ============================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
WORLD_FILE_NAME = os.path.join(_PROJECT_ROOT, "world.pkl")
SAVE_DATA_FILE_NAME = "simulation_data"

# ============================================================================
# WORLD/GRID PARAMETERS
# ============================================================================
WORLD_SIZE = 100                # Grid size (number of cells)
DX = 1.0                        # Cell size (m)
AMBIENT_TEMPERATURE = 293.0     # K - ambient temperature (20 C)

# ============================================================================
# FUEL PROPERTIES
# ============================================================================
FUEL_MASS_PER_AREA = 1.0                                    # kg/m^2 (total fuel load)
HEAT_OF_COMBUSTION = 18e6                                   # J/kg (energy released per kg fuel burned)
MAX_FUEL_LOAD = FUEL_MASS_PER_AREA * HEAT_OF_COMBUSTION     # J/m^2 (total energy per m2 fuel bed)
SPECIFIC_HEAT_FUEL = 1800.0                                 # J/kg * K ( how much energy to raise 1 kg fuel by 1 K)
MAX_MOISTURE_CONTENT = 0.3                                  # kg/m^2 (max moisture content in fuel bed)

# ============================================================================
# IGNITION AND COMBUSTION
# ============================================================================
PYROLYSIS_TEMP = 473.0                                              # K - thermal decomposition begin, pyrolysis
PYROLYSIS_TEMP_EXTRA = 50.0                                         # K - extra temp for ensuring pyrolysis
IGNITION_TEMP_BASE = 523.0                                          # K - base ignition temp for dry fuel
IGNITION_TEMP = IGNITION_TEMP_BASE + (900 - IGNITION_TEMP_BASE)     # inital ignition temp for point zero 

MOISTURE_IGNITION_COEFFICIENT = 10.0  # Coefficient for moisture effect on ignition temp
IGNITION_MOISTURE_LEVEL = 0.0         # Moisture level at ignition point

TAU_BURN = 120.0                    # s - burn time constant for dry fuel
MOISTURE_BURN_REDUCTION = 0.6       # Fractional increase in burn time at max moisture
MOISTURE_EXTINCTION = 0.95          # Fraction of moisture that extinguishes fire completely

# ============================================================================
# HEAT TRANSFER - RADIATION
# ============================================================================
RADIATION_BOOST_FACTOR = 0.05       # Scaling factor to adjust radiation heat transfer 
RADIATION_KERNEL_RADIUS = 2         # Cells to consider for radiation transfer (neighborhood size)

# ============================================================================
# HEAT TRANSFER - DIFFUSION
# ============================================================================
FUEL_THERMAL_CONDUCTIVITY = 0.1     # W/m * K (material conductivity, how fast heat diffuses through fuel bed)

# ============================================================================
# HEAT TRANSFER - WIND
# ============================================================================
WIND_HEAT_TRANSFER_COEFF = 4.0      # W/m^2 * K per m/s (how strongly wind affects convective heat transfer)
WIND_RADIATION_FACTOR = 0.0         # Disabled (using advection only for wind)
WIND_RADIATION_CUTOFF = 3.5         # m/s (cutoff below which wind radiation is ignored)

# ============================================================================
# COOLING - ATMOSPHERIC LOSS
# ============================================================================
AMBIENT_TEMPERATURE_COOLING_COEFFICIENT = 0.005  # 1/s (atmospheric cooling rate)

# ============================================================================
# EVAPORATION
# ============================================================================
MOISTURE_EVAPORATION_RATE = 0.002   # 1/s at boiling point (slow evaporation, fire must pre evaporate moist fuel)
EVAPORATION_RATE_FACTOR_MAX = 2.0   # Max increase in evaporation rate at high temps
MOISTURE_DAMPING_COEFFICIENT = 0.1  # Dampening of evaporation effect on temperature (to avoid instability)

# ============================================================================
# FLAME SUSTAINABILITY
# ============================================================================
MIN_THERMAL_MASS = 500.0            # J/m2 * K (prevents division issues)
FLAME_TEMP_MIN = 600.0              # K, minimum sustained flame temp
FLAME_EXTRA_HEAT_BOOST = 2000.0     # W/m^2 boost to sustain flame
BURNED_OUT_FUEL_THRESHOLD = 0.05    # Fuel fraction below which fire dies
FLAME_SUSTAIN_FUEL_THRESHOLD = 0.1  # Fuel fraction below which flame cannot sustain

# ============================================================================
# NUMERICAL LIMITS
# ============================================================================
WIND_SPEED_CAP = 30.0               # m/s
EPS_WIND_NORM = 1e-10               # to avoid division by zero
MAX_TEMPERATURE = 1400              # Kelvin

# ============================================================================
# WIND DYNAMICS
# ============================================================================
ENABLE_WIND_SHIFT = False
TAU_RELAXATION = 30.0               # s - relaxation time towards background wind
WIND_SHIFT_INTERVAL = 0             # seconds
WIND_SHIFT_MAGNITUDE = 0            # degrees
WIND_SPEED_VARIATION_FACTOR = 0     # fraction of speed
BUOYANCY_COEFFICIENT = 1e-5         # wind change per K temperature gradient

# ============================================================================
# PARAMETER SWEEPS
# ============================================================================
moisture_levels = [0, 0.1, 0.2, 0.3]
moisture_variations = [0.0]
wind_speeds = [0, 5, 10, 15]
wind_speed_variations = [0]
wind_directions = [0]
wind_direction_variations = [0]
turbulence_STDs = [0]
