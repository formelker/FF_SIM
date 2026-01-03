__author__ = ["Ali Awada", "Melker Forslund"]

"""

PDE solver step for wildfire simulation:

Inputs: Fuel array, Temperature array, Moisture array, Wind arrays, World parameters
Outputs: Updated Fuel array, Temperature array, Moisture array

Define fuel as energy content per cell (J/m^2), and moisture is thus mass of water per cell (kg/m^2).

Now the fuel and moisture are a value between 0 and 1 representing the fraction of initial fuel and moisture remaining, but
they can be converted to energy and mass by multiplying with initial fuel load (kg/m^2) and initial moisture content (kg/m^2).

"""

#global imports
import numpy as np

#local imports
from parameters import *
from cell import UNBURNED, BURNING, LIT_OUT, BURNED
from parameters import IGNITION_TEMP_BASE, MOISTURE_IGNITION_COEFFICIENT, MOISTURE_BURN_REDUCTION, MOISTURE_EXTINCTION, FUEL_MASS_PER_AREA, WIND_HEAT_TRANSFER_COEFF

# Physical constants
WATER_EVAPORATION_TEMP = 373                # Water evaporation temperature (K)
LATENT_HEAT_OF_WATER_EVAPORATION = 2257e3   # Latent heat of water evaporation (J/kg)
SPECIFIC_HEAT_WATER = 4.19e3                # Specific heat of water (J/kg*K)
STEFAN_BOLTZMANN = 5.670374419e-8           # (W/m^2/K^4)

def compute_radiation(temperature, moisture=None):
    """
    Radiative heat transfer for fire spread -> cells receive heat from hotter neighbors.

    Physics: Hot cells radiate energy that heats neighboring cells.
    For fire spread: we only care about the incomming radiation from hot neighbors,
    not the energy lost to cold neighbors which is handled by atmospheric cooling.

    Q_received = sum of (sigma * T_neighbor^4) from neighbors hotter than this cell

    Moisture effect: Wet cells receive reduced radiation because energy goes into
    evaporating surface moisture before heating the fuel. This is a key mechanism
    for moisture dampening fire spread.

    This creates the fire spread mechanism while avoiding unrealistic heat loss
    that would extinguish the fire source immediately.
    """

    T = temperature
    radius = RADIATION_KERNEL_RADIUS
    pad = radius
    T_pad = np.pad(T, pad, mode="edge")
    T4_pad = T_pad ** 4
    T_center = T            # Current cell temperatures
    T4_center = T ** 4
    ny, nx = T.shape
    received = np.zeros_like(T, dtype=float)
    radiated = np.zeros_like(T, dtype=float)

    # Precompute distance weights
    y_idx, x_idx = np.ogrid[-radius:radius+1, -radius:radius+1]
    dist = np.sqrt(x_idx**2 + y_idx**2)
    dist[radius, radius] = 1.0  
    weight = 1.0 / (dist**2)
    weight[radius, radius] = 0.0 # No self-radiation
    weight[dist == 0] = 0.0

    # For each offset in the window accumulate radiation from neighbors
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):

            if dy == 0 and dx == 0:
                continue

            w = weight[dy+radius, dx+radius]

            if w == 0:
                continue

            # For each cell, this offset is the neighbor
            T_neighbor = T_pad[pad+dy:pad+dy+ny, pad+dx:pad+dx+nx]
            T4_neighbor = T4_pad[pad+dy:pad+dy+ny, pad+dx:pad+dx+nx]
            hotter_mask = T_neighbor > T_center

            # Energy received by this cell from hotter neighbor
            received += w * np.where(hotter_mask, T4_neighbor - T4_center, 0)

            # Energy radiated from this cell to colder neighbor
            colder_mask = T_neighbor < T_center
            radiated += w * np.where(colder_mask, T4_center - T4_neighbor, 0)

    # Apply Stefan-Boltzmann and boost factor 
    received_flux = RADIATION_BOOST_FACTOR * STEFAN_BOLTZMANN * received
    radiated_flux = RADIATION_BOOST_FACTOR * STEFAN_BOLTZMANN * radiated

    # Moisture dampening: wet cells absorb less radiation effectively
    if moisture is not None:
        moisture_damping = 1.0 - MOISTURE_DAMPING_COEFFICIENT * moisture
        received_flux = received_flux * moisture_damping

    # Net radiative flux: gain from neighbors minus loss to neighbors
    radiation_flux = received_flux - radiated_flux

    return radiation_flux

def compute_thermal_mass(moisture, fuel):
    """
    Compute the thermal mass, heat capacity per unit area [J/m^2*K].
    Includes both fuel and moisture contributions.
    
    fuel: fraction of initial fuel remaining (0-1)
    moisture: fraction of max moisture content (0-1)
    
    Thermal mass = mass * specific_heat for each component
    """
    
    # Fuel mass: FUEL_MASS_PER_AREA is the initial dry fuel mass in kg/m^2
    fuel_mass_per_area = fuel * FUEL_MASS_PER_AREA          # [kg/m^2]
    moisture_kg_per_m2 = moisture * MAX_MOISTURE_CONTENT    # [kg/m^2]
    
    # Total thermal mass from fuel and moisture
    thermal_mass = fuel_mass_per_area * SPECIFIC_HEAT_FUEL + moisture_kg_per_m2 * SPECIFIC_HEAT_WATER  # [J/m^2*K]
    # Prevent extremely low thermal mass for numerical stability so that division by thermal mass is safe
    thermal_mass = np.maximum(thermal_mass, MIN_THERMAL_MASS)
    return thermal_mass

def compute_diffusion(temperature):
    """
    Compute diffusion heat flux based on temperature gradients.
    Q = k * laplacian(T) [W/m^2]
    """

    temperature_padding = np.pad(temperature, pad_width=1, mode="edge")
    laplacian_T = (
                    temperature_padding[:-2, 1:-1] +        # up
                    temperature_padding[2:, 1:-1] +         # down
                    temperature_padding[1:-1, :-2] +        # left
                    temperature_padding[1:-1, 2:] -         # right
                    4 * temperature_padding[1:-1, 1:-1]     # center
                  ) / (DX ** 2)

    diffusion_flux = FUEL_THERMAL_CONDUCTIVITY * laplacian_T  # [W/m^2]

    return diffusion_flux

def compute_gradients_with_bc(temperature, dx=DX):
    """
    Compute spatial gradients using central differences zero edge gradient boundary conditions.
    """
    Tpad = np.pad(temperature, pad_width=1, mode="edge")
    dT_dx = (Tpad[1:-1, 2:] - Tpad[1:-1, :-2]) / (2 * dx)
    dT_dy = (Tpad[2:, 1:-1] - Tpad[:-2, 1:-1]) / (2 * dx)
    return dT_dx, dT_dy

def compute_advection_upwind(temperature, wind_u, wind_v, moisture=None):
    """
    Wind-enhanced heat transfer to downwind cells.

    Physical model: wind tilts flames toward downwind cells.
    This is modeled as:

    Q = h_wind * |v| * (T_upwind - T_current)

    where h_wind is the wind heat transfer coefficient [W/m^2·K per m/s]

    Moisture effect -> wet downwind cells receive reduced heat because energy
    is diverted to evaporating moisture before the fuel can heat up.
    """
    
    T = temperature
    Tpad = np.pad(T, pad_width=1, mode="edge")

    # Get upwind temperatures for each wind direction
    T_west = Tpad[1:-1, :-2]   # Cell to west (column i-1)
    T_east = Tpad[1:-1, 2:]    # Cell to east (column i+1)
    T_north = Tpad[:-2, 1:-1]  # Cell to north (row j-1)
    T_south = Tpad[2:, 1:-1]   # Cell to south (row j+1)

    # Select upwind temperature based on wind direction
    T_upwind_u = np.where(wind_u >= 0, T_west, T_east)
    T_upwind_v = np.where(wind_v >= 0, T_north, T_south)

    wind_u_capped = np.clip(wind_u, -WIND_SPEED_CAP, WIND_SPEED_CAP)
    wind_v_capped = np.clip(wind_v, -WIND_SPEED_CAP, WIND_SPEED_CAP)

    # Heat transfer scales with wind speed and temperature difference
    # Only positive heat transfer, heating downwind cells, not cooling upwind
    heat_from_u = WIND_HEAT_TRANSFER_COEFF * np.abs(wind_u_capped) * np.maximum(0, T_upwind_u - T)
    heat_from_v = WIND_HEAT_TRANSFER_COEFF * np.abs(wind_v_capped) * np.maximum(0, T_upwind_v - T)

    # Total wind-driven heat flux [W/m^2]
    advection_flux = heat_from_u + heat_from_v

    # Moisture dampening: wet cells receive less effective heat from wind
    if moisture is not None:
        moisture_damping = 1.0 - MOISTURE_DAMPING_COEFFICIENT * moisture  # Tunable via parameters
        advection_flux = advection_flux * moisture_damping

    return advection_flux

def compute_wind_enhanced_radiation(temperature, states, wind_u, wind_v):
    """
    Wind tilts flames, favouring radiation downwind.
    
    Model: For BURNING cells, additional radiation is favoured downwind.
    This creates the characteristic elongated fire shape in windy conditions.
    
    The effect scales with wind speed but is bounded to prevent instability.
    Downwind cells receive extra heat from upwind burning cells.
    """
    burning_mask = (states == BURNING).astype(float)
    T = temperature
    
    # Calculate T^4 for burning cells only  
    burning_T4 = burning_mask * (T ** 4)
    
    # Pad for neighbor access  
    T4_pad = np.pad(burning_T4, 1, mode="constant", constant_values=0)
    
    # Get wind speed magnitude at each cell
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    wind_speed_capped = np.minimum(wind_speed, WIND_SPEED_CAP)  # Cap at WIND_SPEED_CAP for stability
    
    # Normalize wind direction
    wind_norm_u = wind_u / (wind_speed + EPS_WIND_NORM)
    wind_norm_v = wind_v / (wind_speed + EPS_WIND_NORM)
    
    # Heat received from upwind burning cells to downwind cells
    # Wind direction determines which neighbors contribute
    heat_from_west = T4_pad[1:-1, :-2] * np.maximum(0, wind_norm_u) 
    heat_from_east = T4_pad[1:-1, 2:] * np.maximum(0, -wind_norm_u)  
    heat_from_north = T4_pad[:-2, 1:-1] * np.maximum(0, wind_norm_v)
    heat_from_south = T4_pad[2:, 1:-1] * np.maximum(0, -wind_norm_v) 

    # Total enhanced radiation downwind 
    enhanced_rad = WIND_RADIATION_FACTOR * wind_speed_capped * STEFAN_BOLTZMANN * (
                    heat_from_west + heat_from_east + heat_from_north + heat_from_south)
    
    return enhanced_rad

def compute_advection(temperature, wind_u, wind_v, moisture=None):
    """
    Wind carries heat -> affect fire spread.

    Moisture is passed through to apply dampening on receiving cells.
    """
    return compute_advection_upwind(temperature, wind_u, wind_v, moisture)

def compute_combustion(temperature, fuel, moisture, states):
    """
    Fuel burning releases heat -> heat source term.
    
    Physics: burning fuel releases energy at rate proportional to remaining fuel.
    dF/dt = -F / tau_burn  [1/s]
    Heat release = -dF/dt * MAX_FUEL_LOAD [W/m^2]
    
    Moisture effect: wet fuels burn slower, energy consumed to evaporate moisture first.
    Effective tau increases with moisture.
    
    Burning cells maintain a minimum flame temperature through
    sustained combustion, this prevents immediate extinction.
    
    Returns:
    - combustion_flux: heat released [W/m^2]
    - dF_dt: fuel consumption rate [1/s] (negative)
    """

    burning_mask = (states == BURNING)
    combustion_flux = np.zeros_like(temperature, dtype=float)
    dF_dt = np.zeros_like(temperature, dtype=float)
    
    # Effective burn time increases with moisture
    # At moisture=0: tau_eff = TAU_BURN
    # At moisture=1: tau_eff = TAU_BURN / 0.3  = 0.3x burn rate
    moisture_factor = np.maximum(1.0 - MOISTURE_BURN_REDUCTION * moisture, 0.1)
    tau_effective = TAU_BURN / moisture_factor
   
    # Fuel consumption rate [1/s]
    # Negative because fuel decreases
    dF_dt[burning_mask] = -fuel[burning_mask] / tau_effective[burning_mask]

    # Heat release [W/m^2] = rate of energy release
    # fuel is fraction, MAX_FUEL_LOAD is total energy [J/m^2]
    # -dF_dt * MAX_FUEL_LOAD gives positive heat release
    combustion_flux[burning_mask] = -dF_dt[burning_mask] * MAX_FUEL_LOAD  # [W/m^2]
    
    # Flames do not just extinguish instantly, they maintain temperature while fuel remains
    cooling_cells = burning_mask & (temperature < FLAME_TEMP_MIN) & (fuel > FLAME_SUSTAIN_FUEL_THRESHOLD)
    
    # Add extra heat to maintain flame for cells with fuel
    # This represents the positive feedback of combustion
    extra_heat_needed = (FLAME_TEMP_MIN - temperature[cooling_cells]) * FLAME_EXTRA_HEAT_BOOST  # W/m^2
    combustion_flux[cooling_cells] += np.maximum(extra_heat_needed, 0)
    
    return combustion_flux, dF_dt

def compute_evaporation(temperature, moisture):
    """
    Moisture evaporates, cooling the cell -> heat sink term.
    
    Physics: Water evaporation absorbs latent heat, cooling the fuel.
    Rate increases with temperature, faster evaporation when hot.
    
    Returns:
    - evaporation_flux: heat absorbed [W/m^2] (negative = cooling)
    - dM_dt_fraction: moisture loss rate [1/s] (negative)
    """
    # Rate factor scales evaporation with temperature
    # Below ambient no evaporation
    # At boiling (373K): rate_factor = 1.0, above boiling max rate_factor = EVAPORATION_RATE_FACTOR_MAX  
    rate_factor = np.clip(
                          (temperature - AMBIENT_TEMPERATURE) / (WATER_EVAPORATION_TEMP - AMBIENT_TEMPERATURE),
                          0.0, 
                          EVAPORATION_RATE_FACTOR_MAX
                         )
    
    # Moisture loss rate [1/s]
    # MOISTURE_EVAPORATION_RATE is base rate at boiling point
    dM_dt_fraction = np.where(moisture > 0, 
                              -MOISTURE_EVAPORATION_RATE * rate_factor * moisture, 
                              0.0)
    
    # Heat absorbed by evaporation [W/m^2]
    # dM_dt in kg/m^2/s = dM_dt_fraction * MAX_MOISTURE_CONTENT
    # Energy = mass_rate * latent_heat
    dM_dt_kg = dM_dt_fraction * MAX_MOISTURE_CONTENT  # [kg/m^2/s]
    evaporation_flux = LATENT_HEAT_OF_WATER_EVAPORATION * dM_dt_kg  # [W/m^2] (negative = cooling)
    
    return evaporation_flux.astype(float), dM_dt_fraction.astype(float)

def compute_atmospheric_cooling(temperature, thermal_mass):
    """
    Atmospheric cooling -> heat loss to environment.
    Q = - AMBIENT_TEMPERATURE_COOLING_RATE * thermal_mass * (T - T_ambient) [W/m^2]

    AMBIENT_TEMPERATURE_COOLING_RATE is a constant defining the rate of cooling [1/s]
    """
    cooling_flux = - AMBIENT_TEMPERATURE_COOLING_COEFFICIENT * thermal_mass * (temperature - AMBIENT_TEMPERATURE)  # [W/m^2]
    
    return cooling_flux

def update_states(temperature, fuel, moisture, states):
    """
    Update cell states to be BURNING, LIT_OUT or BURNED based on temperature, fuel, moisture.
    
    Moisture effects on state transitions:
    1. Ignition temperature increases with moisture (wet fuel harder to ignite)
    2. Fire extinguishes if moisture is above extinction threshold
    3. Burning cells become LIT_OUT if temperature drops below pyrolysis temp
    4. Cells become BURNED when fuel is depleted below threshold
    5. LIT_OUT cells can reignite if conditions allow
    """

    new_states = states.copy()

    # Moisture dependent ignition temperature
    ignition_temp_effective = IGNITION_TEMP_BASE + MOISTURE_IGNITION_COEFFICIENT * moisture
    
    # Ignition requires high enough temp, fuel present and moisture below extinction
    ignition_mask = (((states == UNBURNED) | (states == LIT_OUT)) & 
                     (temperature >= ignition_temp_effective) & 
                     (fuel > 0) &
                     (moisture < MOISTURE_EXTINCTION)
                    )
    new_states[ignition_mask] = BURNING

    # Burned out when fuel depletes below threshold
    burned_out_mask = (states == BURNING) & (fuel <= BURNED_OUT_FUEL_THRESHOLD)
    new_states[burned_out_mask] = BURNED

    # Fire extinguishes if temperature drops below pyrolysis or moisture too high
    lit_out_mask = (states == BURNING) & ((temperature < PYROLYSIS_TEMP) | 
                                          (moisture >= MOISTURE_EXTINCTION)
                                         )
    new_states[lit_out_mask] = LIT_OUT

    return new_states

def pde_step(temperature, fuel, moisture, wind_u, wind_v, states, dt):

    """
    1. Wind updates -> wind responds to fire, buoyancy effects [m/s]
    this is handled in main file for now.

    2. Thermal mass: wet cells resist heating -> moisture damping energy [J/m^2 * K]
    - thermal_mass = volumetric_heat_capacity_fuel * fuel_bed_depth + MOISTURE_CONTENT * SPECIFIC_HEAT_WATER, 
        volumetric_heat_capacity_fuel [J/m^3*K]
        fuel_bed_depth [m]
        moisture_content [kg/m^2] 

    3. All heat transfer mechanisms combined [W/m^2]:
    - Diffusion: fire spread to neighbors. K is thermal conductivity.
    - Advection: wind carries heat -> wind effect on spread.
    - Combustion fuel burning releases heat -> heat source term.  
    - Evaporation: moisture evaporates, cooling the cell -> heat sink term.
    - Radiation: radiative heat transfer between cells -> long-range heat transfer. 
    - Cooling: atmospheric cooling -> heat loss to environment.

    Energy balance equation: 
    total_heat_flux = diffusion + advection + combustion + evaporation + radiation + cooling [W/m^2]

    4. Update temperature based on net heat flux:
    dTemperature/dt = total_heat_flux / thermal_mass [K/s]
    new_temperature = current_temperature + (dTemperature/dt) * dt [K]

    5. Update fuel 
    dFuel/dt = - -fuel / TAU_BURN  (where burning) [J/(m^2*s)] = [W/m^2], where TAU_BURN is characteristic burn time
    new_fuel = current_fuel + (dFuel/dt) * dt [J/m^2]

    6. Update moisture
    dMoisture/dt = - evaporation_rate / TAU_EVAP (where evaporating) [kg/m^2 * s], where TAU_EVAP is characteristic evaporation time
    new_moisture = current_moisture + (dMoisture/dt) * dt [kg/m^2]

    7. Update states based on temperature, fuel, moisture and current states
    """
    
    # 1. Update wind field
    # Done in main file!

    # 2. Compute thermal mass
    thermal_mass = compute_thermal_mass(moisture, fuel)     # [J/m^2*K]

    # 3. Compute total heat flux
    diffusion = compute_diffusion(temperature)                                      # [W/m^2]
    advection = compute_advection(temperature, wind_u, wind_v, moisture)            # [W/m^2] - includes moisture dampening
    combustion, dF_dt = compute_combustion(temperature, fuel, moisture, states)     # [W/m^2]
    evaporation, dM_dt = compute_evaporation(temperature, moisture)                 # [W/m^2]
    cooling = compute_atmospheric_cooling(temperature, thermal_mass)                # [W/m^2]
    
    # Wind-dependent suppression of isotropic radiation
    wind_speed_mean = np.mean(np.sqrt(wind_u**2 + wind_v**2))
    radiation_scale = 1.0 / (1.0 + wind_speed_mean / WIND_RADIATION_CUTOFF)
    radiation = compute_radiation(temperature, moisture) * radiation_scale  # [W/m^2]

    total_heat_flux = diffusion + advection + combustion + evaporation + cooling + radiation  # [W/m^2]

    # 4. Update temperature
    new_temperature = temperature + (total_heat_flux / thermal_mass) * dt
    new_temperature = np.clip(new_temperature, AMBIENT_TEMPERATURE, MAX_TEMPERATURE)  # Ensure temperature does not go below ambient temperature

    # 5. Update fuel
    new_fuel = fuel + dF_dt * dt
    new_fuel = np.clip(new_fuel, 0.0, 1.0)  # Ensure fuel stays within [0, 1]

    # 6. Update moisture
    new_moisture = moisture + dM_dt * dt
    new_moisture = np.clip(new_moisture, 0.0, 1.0)  # Ensure moisture stays within [0, 1]

    # 7. Update states
    new_states = update_states(new_temperature, new_fuel, new_moisture, states)
    
    return new_temperature, new_fuel, new_moisture, new_states