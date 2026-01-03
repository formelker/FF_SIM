__author__ = ["Ali Awada", "Melker Forslund"]

# imports
import numpy as np
import keyboard
import PyQt6.QtWidgets as QtWidgets
import pickle

# local imports
from create_world import load_world
from world import create_moisture_map_gaussian, create_wind_field_map
from wind import update_wind_field
from cell import BURNING, UNBURNED, LIT_OUT, BURNED
from PDEs import pde_step
from parameters import *
from visualize_world import WorldVisualizer, WindVisualizer, StateVisualizer, MAX_WIND_SPEED_FOR_PLOT

VISUALIZE = True  # Whether to visualize the simulation during runtime or not
VISUALIZE_INTERVAL = 10 # visualize every N steps

def run_simulation(world, number_of_steps, dt, mean_wind_speed, wind_direction_degrees, turbulence_std=0.3, visualize=VISUALIZE):
    plot_data = []

    #save first state of world
    plot_data.append(world.create_snapshot())
   
    step = 0

    visualizer = None
    wind_visualizer = None
    state_visualizer = None
    application = None

    if visualize:
        application = QtWidgets.QApplication([])
        visualizer = WorldVisualizer(world)
        visualizer.show()
        
        # Create separate wind visualization window with static range based on SPEED_MEAN
        wind_visualizer = WindVisualizer(world, max_speed_range=MAX_WIND_SPEED_FOR_PLOT, 
                                        speed_mean=mean_wind_speed, direction_degrees=wind_direction_degrees)
        wind_visualizer.show()
        
        # Create separate state visualization window
        state_visualizer = StateVisualizer(world)
        state_visualizer.show()


    paused = False
    PAUSE_KEY = '§'
    print("Press '§' to pause/resume visualization at any time.")

    while step < number_of_steps if number_of_steps is not None else True:
        if keyboard.is_pressed('esc'):
            print("Simulation interrupted by user.")
            break

        # Pause/resume logic
        if keyboard.is_pressed(PAUSE_KEY):
            if not paused:
                print("\nSimulation paused. Press '§' again to resume.")
                paused = True
                # Wait for key release to avoid instant resume
                while keyboard.is_pressed(PAUSE_KEY):
                    pass

        while paused:
            if visualize:
                visualizer.update_visualization()
                wind_visualizer.update_visualization()
                state_visualizer.update_visualization()
                application.processEvents()
            # Resume if PAUSE_KEY pressed again
            if keyboard.is_pressed(PAUSE_KEY):
                print("\nResuming simulation.")
                paused = False
                while keyboard.is_pressed(PAUSE_KEY):
                    pass
            # Allow user to quit while paused
            if keyboard.is_pressed('esc'):
                print("Simulation interrupted by user (while paused).")
                return plot_data

        # 1. get current wind and update wind field
        wind_u, wind_v = world.get_wind_arrays()
        wind_u, wind_v = update_wind_field(wind_u, wind_v, world.get_temperature_array(), world.vu_background, world.vv_background, dt, turbulence_std)

        # 2. PDE step to update temperature, fuel, moisture, and states
        temperature = world.get_temperature_array()
        fuel = world.get_fuel_array()
        moisture = world.get_moisture_array()
        states = world.get_state_array()

        temperature, fuel, moisture, states = pde_step(temperature, fuel, moisture, wind_u, wind_v, states, dt=dt)

        # 3. Update world data
        world.update_from_arrays(temperature, fuel, moisture, wind_u, wind_v, states)

        # 4. Store data
        plot_data.append(world.create_snapshot())

        # 5. Visualize progress if visualize=True
        if visualize and step % VISUALIZE_INTERVAL == 0:
            # Update visualizations
            visualizer.update_visualization()
            wind_visualizer.update_visualization()
            state_visualizer.update_visualization()
            application.processEvents()

        # 6. Check if fire has burned out (no BURNING cells remain)
        if not np.any(states == BURNING):
            steps_str = f"{step}/{number_of_steps}" if number_of_steps is not None else f"{step}"
            print(f"\nFire burned out at step {steps_str} ({step * dt / 60:.1f} minutes)")
            break

        # Loop         
        step += 1
        steps_str = f"{step}/{number_of_steps}" if number_of_steps is not None else f"{step}"
        print(f"Completed step {steps_str}", end='\r', flush=True)

    if application is not None:
        visualizer.close()
        wind_visualizer.close()
        state_visualizer.close()
        application.quit()

    return plot_data

if __name__ == "__main__":
    
    number_of_steps = NUMBER_OF_STEPS  
    dt = DT               

    print("Starting simulation with parameters: ")
    print(f"Simulation steps: {number_of_steps}")
    print(f"Time step: {dt} s")
    if number_of_steps is not None:
        print(f"Simulation time: {number_of_steps * dt} s, or {number_of_steps * dt / 60:.2f} minutes")
    print("Press 'esc' to interrupt the simulation.")

    simulation_data = []

    # Generate all combinations of parameters using nested loops    
    param_combinations = zip(moisture_levels, moisture_variations, wind_speeds, wind_speed_variations, wind_directions, wind_direction_variations, turbulence_STDs)
    tot_comb = len(moisture_levels) * len(moisture_variations) * len(wind_speeds) * len(wind_speed_variations) * len(wind_directions) * len(wind_direction_variations) * len(turbulence_STDs)
    print(f"Total parameter combinations to run: {tot_comb}")
    
    current_comb = 0
    for moisture_level in moisture_levels:
        for moisture_variation in moisture_variations:
            for wind_speed in wind_speeds:
                for wind_speed_variation in wind_speed_variations:
                    for wind_direction in wind_directions:
                        for wind_direction_variation in wind_direction_variations:
                            for turbulence_std in turbulence_STDs:
                                current_comb += 1
                                print(f"\nRunning simulation {current_comb}/{tot_comb} with parameters:")
                                print(f"\n{'='*60}")
                                print(f"{'='*60}")
                                print(f"  Moisture level:           {moisture_level}")
                                print(f"  Moisture variation:       {moisture_variation}")
                                print(f"  Wind speed:               {wind_speed} m/s")
                                print(f"  Wind variation:           {wind_speed_variation}")
                                print(f"  Wind direction:           {wind_direction} degrees")
                                print(f"  Wind direction variation: {wind_direction_variation}")
                                print(f"  Turbulence std:           {turbulence_std} m/s")
                                print()

                                world = load_world(WORLD_FILE_NAME)
                                moisture_map = create_moisture_map_gaussian(WORLD_SIZE, moisture_level=moisture_level, variation=moisture_variation)
                                wind_map = create_wind_field_map(world.size, wind_speed, wind_speed_variation, wind_direction, wind_direction_variation)

                                world.set_moisture_map(moisture_map)
                                world.set_wind_map(wind_map)
                                world.set_ignition_point(WORLD_SIZE // 2, WORLD_SIZE // 2)

                                result = run_simulation(world, number_of_steps, dt, wind_speed, wind_direction, turbulence_std)
                                print("Simulation run complete.")
                                simulation_parameters = {
                                    "moisture_level": moisture_level,
                                    "moisture_variation": moisture_variation,
                                    "wind_speed": wind_speed,
                                    "wind_speed_variation": wind_speed_variation,
                                    "wind_direction": wind_direction,
                                    "wind_direction_variation": wind_direction_variation,
                                    "turbulence_std": turbulence_std
                                }

                                simulation_data.append((simulation_parameters, result))

                                print("Simulation completed.")
    
    print("All simulations completed")
    print("Saving simulation data...")

    # save all data
    with open(SAVE_DATA_FILE_NAME, "wb") as f:
        pickle.dump(simulation_data, f)

    print(f"Simulation data saved to {SAVE_DATA_FILE_NAME}")
    print("Program finished.")
