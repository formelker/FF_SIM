__author__ = ["Ali Awada", "Melker Forslund"]
"""
This main file is for running multiple simulations in parallel with different parameters.
It can not visualize the simulation in real-time.
"""

# imports
import numpy as np
import keyboard
import PyQt6.QtWidgets as QtWidgets
import pickle
import os
from multiprocessing import Pool, cpu_count
from functools import partial

# local imports
from create_world import load_world
from world import create_moisture_map_gaussian, create_wind_field_map
from wind import update_wind_field, update_background_wind, create_wind_vectors_from_speed_direction
from PDEs import pde_step 
from parameters import *
from visualize_world import WorldVisualizer, WindVisualizer, StateVisualizer, MAX_WIND_SPEED_FOR_PLOT
from cell import BURNING, UNBURNED, LIT_OUT, BURNED

# Performance tuning parameters
STORE_EVERY_N_STEPS = 1     # Store snapshots every N steps (saves memory/time)
VERBOSE_WORKERS = False     # Set to False to reduce worker process output
MAX_WORKERS = 4             # Set to a number to limit cores used (None = use all)

REPEATS_PER_PARAMETER_SET = 1  # Number of repeats per parameter combination (to average out randomness if it exists)

def run_single_simulation(params_tuple):
    """
    Run a single simulation with given parameters.
    This function is designed to be called by multiprocessing Pool.
    
    Inputs:
    - params_tuple: tuple containing (simulation_number, total_simulations, moisture_level, 
                    moisture_variation, wind_speed, wind_speed_variation, wind_direction, 
                    wind_direction_variation, turbulence_std)
    
    Outputs:
    - tuple: (simulation_parameters dict, result data)
    """
    (sim_num, total_sims, moisture_level, moisture_variation, wind_speed, 
     wind_speed_variation, wind_direction, wind_direction_variation, turbulence_std) = params_tuple
    
    if VERBOSE_WORKERS:
        print(f"\n{'='*60}")
        print(f"Running simulation {sim_num}/{total_sims} (PID: {os.getpid()})")
        print(f"{'='*60}")
        print(f"  Moisture level:           {moisture_level}")
        print(f"  Moisture variation:       {moisture_variation}")
        print(f"  Wind speed:               {wind_speed} m/s")
        print(f"  Wind variation:           {wind_speed_variation}")
        print(f"  Wind direction:           {wind_direction}°")
        print(f"  Wind direction variation: {wind_direction_variation}°")
        print(f"  Turbulence std:           {turbulence_std} m/s")
        print()
    else:
        print(f"Starting simulation {sim_num}/{total_sims}...")
    
    # Load world and set parameters
    world = load_world(WORLD_FILE_NAME)
    moisture_map = create_moisture_map_gaussian(WORLD_SIZE, moisture_level=moisture_level, variation=moisture_variation)
    wind_map = create_wind_field_map(world.size, wind_speed, wind_speed_variation, wind_direction, wind_direction_variation)
    
    world.set_moisture_map(moisture_map)
    world.set_wind_map(wind_map)
    world.set_ignition_point(WORLD_SIZE // 2, WORLD_SIZE // 2)
    
    # Create unique chunk base name for this simulation (includes sim_num and PID for uniqueness)
    sim_chunk_base = f"{SAVE_DATA_FILE_NAME}_sim{sim_num}_pid{os.getpid()}"
    
    # Run simulation
    result = run_simulation(world, NUMBER_OF_STEPS, DT, wind_speed, wind_direction, turbulence_std, 
                            visualize=False, store_interval=STORE_EVERY_N_STEPS,
                            chunk_base_name=sim_chunk_base)
    
    simulation_parameters = {"moisture_level": moisture_level,
                             "moisture_variation": moisture_variation,
                             "wind_speed": wind_speed,
                             "wind_speed_variations": wind_speed_variation,
                             "wind_direction": wind_direction,
                             "wind_direction_variation": wind_direction_variation,
                             "turbulence_std": turbulence_std,
                             "sim_num": sim_num
                            }
    
    if VERBOSE_WORKERS:
        print(f"Simulation {sim_num}/{total_sims} completed.")
    else:
        print(f"Completed simulation {sim_num}/{total_sims}")
    
    return (simulation_parameters, result)

def run_simulation(world, number_of_steps, dt, mean_wind_speed, wind_direction_degrees, turbulence_std=0.3, visualize=False, store_interval=1, chunk_base_name=None):
    """
    Main sim loop.
    Works like main.py but without visualization and chunk saving.
    """

    plot_data = []
    chunk_idx = 0
    step = 0
    CHUNK_SIZE = 1000  # Save every 1000 steps (tune as needed based on your ram and storage)
    
    # Use provided chunk_base_name or default to SAVE_DATA_FILE_NAME
    if chunk_base_name is None:
        chunk_base_name = SAVE_DATA_FILE_NAME
    
    chunk_files = []  # Keep track of saved chunk files

    initial_wind_speed = mean_wind_speed
    initial_wind_direction = wind_direction_degrees
    
    # Store initial state
    plot_data.append(world.create_snapshot())

    while step < number_of_steps if number_of_steps is not None else True:
        
        # Check for user interrupt 
        if keyboard.is_pressed('esc'):
            print("Simulation interrupted by user.")
            break
        
        # Update wind field
        current_time = step * dt
        current_wind_speed, current_wind_direction = update_background_wind(initial_wind_speed, initial_wind_direction, current_time)

        # Update background wind vectors if wind shift is enabled
        if ENABLE_WIND_SHIFT:
            world.vu_background, world.vv_background = create_wind_vectors_from_speed_direction(current_wind_speed, current_wind_direction, world.vu_background.shape)

        wind_u, wind_v = world.get_wind_arrays()
        
        # update wind 
        wind_u, wind_v = update_wind_field(wind_u, wind_v, world.get_temperature_array(), world.vu_background, world.vv_background, dt, turbulence_std)
        
        temperature = world.get_temperature_array()
        fuel = world.get_fuel_array()
        moisture = world.get_moisture_array()
        states = world.get_state_array()
        
        # Perform PDE step
        temperature, fuel, moisture, states = pde_step(temperature, fuel, moisture, wind_u, wind_v, states, dt=dt)
        
        # Update world states
        world.update_from_arrays(temperature, fuel, moisture, wind_u, wind_v, states)
        
        # store data at intervals
        if step % store_interval == 0 or (number_of_steps is not None and step == number_of_steps - 1):
            plot_data.append(world.create_snapshot())
        
        # Save chunk if reached CHUNK_SIZE
        if len(plot_data) >= CHUNK_SIZE:
            
            chunk_file = f"{chunk_base_name}_chunk{chunk_idx}.pkl"
            
            with open(chunk_file, "wb") as f:
                pickle.dump(plot_data, f)
            
            chunk_files.append(chunk_file)
            
            if VERBOSE_WORKERS:
                print(f"Saved chunk {chunk_idx} at step {step} to {chunk_file}")
            
            plot_data = []

            chunk_idx += 1
        
        # check if fire is burned out
        if not np.any(states == BURNING):
            
            # Store final state if not already stored
            if step % store_interval != 0:
                plot_data.append(world.create_snapshot())

            steps_str = f"{step}/{number_of_steps}" if number_of_steps is not None else f"{step}"
            
            print(f"\nFire burned out at step {steps_str} ({step * dt / 60:.1f} minutes)")
            
            # Save final chunk if there is any remaining data
            if plot_data:
                chunk_file = f"{chunk_base_name}_chunk{chunk_idx}.pkl"
                
                with open(chunk_file, "wb") as f:
                    pickle.dump(plot_data, f)
                    
                chunk_files.append(chunk_file)
                
                # Print final chunk save info
                if VERBOSE_WORKERS:
                    print(f"Saved final chunk {chunk_idx} at step {step} to {chunk_file}")
            break

        step += 1
        
        # Print progress
        if VERBOSE_WORKERS and step % 50 == 0:
            steps_str = f"{step}/{number_of_steps}" if number_of_steps is not None else f"{step}"
            print(f"Step {steps_str}", end='\r', flush=True)
    
    # Save any remaining data as final chunk
    if plot_data:
        chunk_file = f"{chunk_base_name}_chunk{chunk_idx}.pkl"
        
        with open(chunk_file, "wb") as f:
            pickle.dump(plot_data, f)
        
        chunk_files.append(chunk_file)
        
        if VERBOSE_WORKERS:
            print(f"Saved final chunk {chunk_idx} to {chunk_file}")
    
    return chunk_files

if __name__ == "__main__":

    for repeat in range(REPEATS_PER_PARAMETER_SET):
        print(f"\n=== Starting repeat {repeat + 1}/{REPEATS_PER_PARAMETER_SET} ===\n")
    
        number_of_steps = NUMBER_OF_STEPS  # total number of simulation steps
        dt = DT                            # Time step

        print("Starting parallel simulation with parameters: ")
        print(f"Simulation steps: {number_of_steps}")
        print(f"Time step: {dt} s")
        if number_of_steps is not None:
            print(f"Simulation time: {number_of_steps * dt} s, or {number_of_steps * dt / 60:.2f} minutes")
        
        # Determine number of CPU cores to use
        available_cores = cpu_count()
        num_cores = MAX_WORKERS if MAX_WORKERS is not None else available_cores
        num_cores = min(num_cores, available_cores) # Limit to max available cores
        
        print(f"Available CPU cores: {available_cores}")
        print(f"Using {num_cores} parallel processes")
        print(f"Storing snapshots every {STORE_EVERY_N_STEPS} steps")
        print(f"Verbose worker output: {VERBOSE_WORKERS}")
        print()

        simulation_data = []

        # Generate all parameter combinations
        param_combinations = []
        for moisture_level in moisture_levels:
            for moisture_variation in moisture_variations:
                for wind_speed in wind_speeds:
                    for wind_speed_variation in wind_speed_variations:
                        for wind_direction in wind_directions:
                            for wind_direction_variation in wind_direction_variations:
                                for turbulence_std in turbulence_STDs:
                                    param_combinations.append((
                                        moisture_level,
                                        moisture_variation,
                                        wind_speed,
                                        wind_speed_variation,
                                        wind_direction,
                                        wind_direction_variation,
                                        turbulence_std
                                    ))
        
        tot_comb = len(param_combinations)
        print(f"Total parameter combinations to run: {tot_comb}")
        print(f"{'='*60}\n")
        
        # Create tuples with simulation numbers for tracking
        params_with_numbers = [(i+1, tot_comb) + params for i, params in enumerate(param_combinations)]
        
        # Run simulations in parallel using multiprocessing Pool
        # chunksize helps balance load distribution
        chunksize = max(1, tot_comb // (num_cores * 4))  # 4 chunks per worker
        print(f"Starting parallel simulations with chunksize={chunksize}...")
        if NUMBER_OF_STEPS is not None:
            print(f"Each simulation will run for up to {NUMBER_OF_STEPS} steps.")
        
        with Pool(processes=num_cores) as pool:
            simulation_data = pool.map(run_single_simulation, params_with_numbers, chunksize=chunksize)
        
        print("\n" + "="*60)
        print("All simulations completed!")
        print("="*60)
        print("Saving simulation data...")

        # save all data in one single file per repeat if possible
        with open(SAVE_DATA_FILE_NAME + f"_repeat{repeat+1}.pkl", "wb") as f:
            pickle.dump(simulation_data, f)

        print(f"Simulation data saved to {SAVE_DATA_FILE_NAME}_repeat{repeat+1}.pkl")
    
    print("\nAll repeats completed. Program finished.")


