__author__ = ["Ali Awada", "Melker Forslund"]

import numpy as np 
from PDEs import compute_gradients_with_bc
from parameters import ENABLE_WIND_SHIFT, TAU_RELAXATION, WIND_SHIFT_INTERVAL, WIND_SHIFT_MAGNITUDE, WIND_SPEED_VARIATION_FACTOR, DX, AMBIENT_TEMPERATURE, BUOYANCY_COEFFICIENT

# update background wind function not used in report presented simulations but kept for future useage
def update_background_wind(initial_speed, initial_direction, current_time, enable_shift=ENABLE_WIND_SHIFT):
    if not enable_shift:
        return initial_speed, initial_direction
    
    shift_number = int(current_time / WIND_SHIFT_INTERVAL)
    
    np.random.seed(shift_number + 1000)  # Offset to avoid conflicts
    
    # Random direction shift (degrees)
    direction_shift = np.random.uniform(-WIND_SHIFT_MAGNITUDE, WIND_SHIFT_MAGNITUDE)
    new_direction = initial_direction + direction_shift
    
    # Random speed variation
    speed_factor = 1.0 + np.random.uniform(-WIND_SPEED_VARIATION_FACTOR, WIND_SPEED_VARIATION_FACTOR)
    new_speed = initial_speed * speed_factor
    new_speed = max(0.0, new_speed) 
    
    # Reset random seed
    np.random.seed(None)

    return new_speed, new_direction

def create_wind_vectors_from_speed_direction(speed, direction_degrees, shape):
    # Convert to radians 
    theta = np.deg2rad(direction_degrees)
    
    if np.isscalar(speed):
        vu = speed * np.sin(theta) * np.ones(shape)
        vv = -speed * np.cos(theta) * np.ones(shape)
    else:
        vu = speed * np.sin(theta)
        vv = -speed * np.cos(theta) 
    
    return vu, vv

def update_wind_field(vu, vv, temperature, vu_background, vv_background, dt, turbulence_std=0):
    # Relaxation towards background wind
    relax_du = -(vu - vu_background) / TAU_RELAXATION
    relax_dv = -(vv - vv_background) / TAU_RELAXATION

    # Fire induced wind (buoyancy effect)
    # Heated air rises and creates inflow towards hot regions
    # Gradient points toward hot region, so wind should follow gradient
    dT_dx, dT_dy = compute_gradients_with_bc(temperature, DX)
    
    # Scale buoyancy by temperature excess over ambient temperature
    dT = np.maximum(temperature - AMBIENT_TEMPERATURE, 0.0)
    
    buoyancy_du = BUOYANCY_COEFFICIENT * dT * dT_dx
    buoyancy_dv = BUOYANCY_COEFFICIENT * dT * dT_dy

    # Turbulence, sqrt(dt) scaling for slower time correlation
    if turbulence_std > 0:
        turbulence_du = np.random.normal(0, turbulence_std * np.sqrt(dt), vu.shape)
        turbulence_dv = np.random.normal(0, turbulence_std * np.sqrt(dt), vv.shape)
    else:
        turbulence_du = 0.0
        turbulence_dv = 0.0

    # Update wind vectors:
    du_dt = relax_du + buoyancy_du
    dv_dt = relax_dv + buoyancy_dv
    
    new_vu = vu + du_dt * dt + turbulence_du
    new_vv = vv + dv_dt * dt + turbulence_dv

    return new_vu, new_vv

