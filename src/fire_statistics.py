__author__ = ["Ali Awada", "Melker Forslund"]

"""
Forest Fire Simulation Statistics and Visualization

This module analyzes and visualizes forest fire simulation data including:
- Fire intensity heatmap
- Spread rate over time
- Burned area progression
- Fire behavior vs moisture levels and wind speed

"""

import matplotlib.animation as animation
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from world import WorldSnapshot
from parameters import *
from cell import BURNING, LIT_OUT, BURNED
from visualize_world import TEMPERATURE_RANGE


# Number of replicate files to load (can be changed by user, not used for the chunk loading functions)
NUM_FILES_TO_LOAD = 1

# Enable/disable standard deviation calculations and error bars
COMPUTE_STD = False

# Base filename pattern (without _repeat suffix)
BASE_FILENAME = SAVE_DATA_FILE_NAME

def plot_2d_intensity_heatmap(all_stats, compute_std=None):
    """
    Create a 2d heatmap showing mean burning temperature (fire intensity) 
    as a function of both moisture and wind speed.
    """

    moisture_wind_data = {}
    
    for stats in all_stats:
        wind_speed = stats["wind_speed"]
        moisture = stats["moisture"]
        mean_temp = stats["mean_burning_temp"]
        
        key = (moisture, wind_speed)

        if key not in moisture_wind_data:
            moisture_wind_data[key] = []
        
        moisture_wind_data[key].append(mean_temp)
    
    # Average replicates
    for key in moisture_wind_data:
        moisture_wind_data[key] = np.mean(moisture_wind_data[key])
    
    # Get unique values
    moistures = sorted(set(m for m, w in moisture_wind_data.keys()))
    wind_speeds = sorted(set(w for m, w in moisture_wind_data.keys()))
    
    # Create 2D array
    heatmap_data = np.zeros((len(wind_speeds), len(moistures)))
    for i, ws in enumerate(wind_speeds):
        for j, m in enumerate(moistures):
            heatmap_data[i, j] = moisture_wind_data.get((m, ws), AMBIENT_TEMPERATURE)
    
    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap_data, aspect="auto", cmap="hot", origin="lower")
    
    cbar = plt.colorbar(im, ax=plt.gca())
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Mean burning temperature (K)", fontsize=15)
    plt.xlabel("Fuel moisture level", fontsize=15)
    plt.ylabel("Wind speed (m/s)", fontsize=15)
    plt.title("Fire intensity map: mean burning temperature vs moisture and wind", 
              fontsize=15)
    
    # Set tick labels
    plt.xticks(range(len(moistures)), [f"{m:.2f}" for m in moistures], fontsize=12)
    plt.yticks(range(len(wind_speeds)), [f"{w}" for w in wind_speeds], fontsize=12)
    plt.tight_layout()
    plt.savefig("fire_intensity_2d_heatmap.png", dpi=300)
    plt.show()

def load_simulation_data(filename):
    """
    Load pickled simulation data from file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)

def discover_chunk_files(base_path=".", base_name=None):
    """
    Discover all chunk files in a directory and group them by simulation.
    
    Args:
        base_path: Directory to search for chunk files
        base_name: Base name pattern (default: SAVE_DATA_FILE_NAME)
    
    Returns:
        Dictionary mapping simulation identifiers to sorted lists of chunk file paths
    """

    if base_name is None:
        base_name = SAVE_DATA_FILE_NAME
    
    # Handle both old format (with .pkl in base) and new format (without .pkl)
    # Pattern: {base_name}[.pkl]_sim{N}_pid{PID}_chunk{N}.pkl
    pattern = os.path.join(base_path, f"{base_name}*_sim*_chunk*.pkl")
    all_chunks = glob.glob(pattern)
    
    # Also check for .pkl in the base name (legacy format)
    if not all_chunks:
        pattern = os.path.join(base_path, f"{base_name}.pkl*_sim*_chunk*.pkl")
        all_chunks = glob.glob(pattern)
    
    if not all_chunks:
        print(f"  No chunk files found matching pattern: {pattern}")
        return {}
    
    # Group by simulation (sim_N_pid_P)
    sim_chunks = {}
    
    for chunk_path in all_chunks:
        filename = os.path.basename(chunk_path)
        # Extract sim and pid from filename
        match = re.search(r"_sim(\d+)_pid(\d+)_chunk(\d+)\.pkl$", filename)
        
        if match:
            sim_num = int(match.group(1))
            pid = match.group(2)
            chunk_num = int(match.group(3))
            sim_key = f"sim{sim_num}_pid{pid}"
            
            if sim_key not in sim_chunks:
                sim_chunks[sim_key] = []

            sim_chunks[sim_key].append((chunk_num, chunk_path))
    
    # Sort chunks by chunk number for each simulation
    for sim_key in sim_chunks:
        sim_chunks[sim_key] = [path for _, path in sorted(sim_chunks[sim_key])]
    
    print(f"  Found {len(sim_chunks)} simulations with chunk files")

    return sim_chunks

def load_statistics_from_chunks(base_path=".", base_name=None, dt=DT):
    """
    Load statistics directly from discovered chunk files (no main repeat file needed).
    
    Args:
        base_path: Directory containing chunk files
        base_name: Base name pattern for files
        dt: Timestep in seconds
    
    Returns:
        List of statistics dictionaries
    """

    sim_chunks = discover_chunk_files(base_path, base_name)
    
    if not sim_chunks:
        print("  ERROR: No simulation data found!")
        return []
    
    all_stats = []
    
    for sim_idx, (sim_key, chunk_files) in enumerate(sorted(sim_chunks.items())):
        print(f"  Processing {sim_key} ({len(chunk_files)} chunks)...")
        
        try:
            # Initialize accumulators
            first_snapshot = None
            last_snapshot = None
            total_snapshots = 0
            temp_sum = None
            burning_temps = []
            spread_rates = []
            prev_cells = None
            duration_steps = 0
            fire_still_burning = True
            
            # Process chunks one at a time
            for chunk_idx, chunk_file in enumerate(chunk_files):
                try:
                    chunk_data = load_simulation_data(chunk_file)
                except FileNotFoundError:
                    print(f"    Warning: Chunk {chunk_file} not found, skipping")
                    continue
                except Exception as e:
                    print(f"    Warning: Error loading {chunk_file}: {e}")
                    continue
                
                for snapshot in chunk_data:
                    # Store first and last snapshots
                    if first_snapshot is None:
                        first_snapshot = snapshot
                        prev_cells = count_affected_cells(snapshot.get_state_array())
                        temp_sum = np.zeros_like(snapshot.get_temperature_array(), dtype=np.float64)
                    
                    last_snapshot = snapshot
                    total_snapshots += 1
                    
                    # Accumulate temperature sum for average
                    temp_sum += snapshot.get_temperature_array()
                    
                    # Track burning temperatures
                    state_array = snapshot.get_state_array()
                    temp_array = snapshot.get_temperature_array()
                    burning_mask = (state_array == BURNING)
                    if burning_mask.any():
                        burning_temps.append(temp_array[burning_mask].mean())
                    
                    # Track spread rates
                    curr_cells = count_affected_cells(state_array)
                    if prev_cells is not None:
                        rate = (curr_cells - prev_cells) * (DX**2) / dt
                        spread_rates.append(rate)
                    prev_cells = curr_cells
                    
                    # Check if fire is still burning
                    if fire_still_burning and not np.any(state_array == BURNING):
                        fire_still_burning = False
                        duration_steps = total_snapshots
                
                del chunk_data  # Free memory immediately
            
            # Skip if no data was loaded
            if first_snapshot is None or total_snapshots == 0:
                print(f"    Warning: No data loaded for {sim_key}, skipping")
                continue
            
            # Extract simulation number from key for parameter reconstruction
            match = re.search(r"sim(\d+)", sim_key)
            sim_num = int(match.group(1)) if match else sim_idx + 1
            
            # Create statistics dict (parameters unknown from chunks alone)
            stats = {"parameters": {"sim_num": sim_num, "sim_key": sim_key},
                     "wind_speed": None, 
                     "moisture": None,   
                     "wind_direction": None,  
                    }
            
            # Final burned area
            final_state = last_snapshot.get_state_array()
            burned_cells = np.sum((final_state == BURNED) | (final_state == BURNING))
            stats["burned_area"] = burned_cells * (DX ** 2)
            
            # Spread rate
            initial_cells = count_affected_cells(first_snapshot.get_state_array())
            final_cells = count_affected_cells(final_state)
            total_time = (total_snapshots - 1) * dt
            stats["avg_spread_rate"] = ((final_cells - initial_cells) * DX**2) / total_time if total_time > 0 else 0.0
            
            # Fire duration
            if not fire_still_burning:
                stats["duration_minutes"] = (duration_steps * dt) / 60
            else:
                stats["duration_minutes"] = (total_snapshots * dt) / 60
            
            # Fire shape
            burned_mask = (final_state == BURNED) | (final_state == BURNING)
            if burned_mask.any():
                center_y, center_x = np.array(burned_mask.shape) // 2
                burned_coords = np.where(burned_mask)
                
                max_north = center_y - burned_coords[0].min()
                max_south = burned_coords[0].max() - center_y
                max_east = burned_coords[1].max() - center_x
                max_west = center_x - burned_coords[1].min()
                
                parallel_extent = max_north + max_south
                perpendicular_extent = max_east + max_west
            
                stats["elongation_ratio"] = parallel_extent / perpendicular_extent if perpendicular_extent > 0 else 1.0
            
            else:
                stats["elongation_ratio"] = 1.0
            
            # Peak spread rate timing
            stats["peak_time_minutes"] = (np.argmax(spread_rates) * dt) / 60 if spread_rates else 0
            
            # Fuel consumption
            initial_fuel = first_snapshot.get_fuel_array()
            final_fuel = last_snapshot.get_fuel_array()
            total_initial = initial_fuel.sum()
            stats["fuel_consumed_pct"] = ((total_initial - final_fuel.sum()) / total_initial * 100) if total_initial > 0 else 0
            
            # Mean burning temperature
            stats["mean_burning_temp"] = np.mean(burning_temps) if burning_temps else AMBIENT_TEMPERATURE
            
            # Average temperature map
            stats["avg_temp_map"] = temp_sum / total_snapshots
            
            all_stats.append(stats)
            
            # Free memory
            del temp_sum, burning_temps, spread_rates, first_snapshot, last_snapshot
            
            print(f"    Extracted stats: {stats['burned_area']:.0f} m^2 burned, {stats['duration_minutes']:.1f} min")
            
        except MemoryError:
            print(f"    ERROR: Out of memory processing {sim_key}")
            continue
    
    print(f"  Total statistics extracted: {len(all_stats)} simulations\n")

    return all_stats

def extract_statistics_from_data(data, dt):
    """
    Extract summary statistics from simulation data without keeping full world states.
    Returns a list of statistics dictionaries, one per simulation.
    """
    stats_list = []
    
    for parameters, world_states in data:
        # Extract all needed statistics for this simulation
        stats = {"parameters": parameters.copy(),
                 "wind_speed": parameters["wind_speed"],
                 "moisture": parameters["moisture_level"],
                 "wind_direction": parameters["wind_direction"],
                 }
        
        # Final burned area
        final_state = world_states[-1].get_state_array()
        burned_cells = np.sum((final_state == BURNED) | (final_state == BURNING))
        stats["burned_area"] = burned_cells * (DX ** 2)
        
        # Spread rate
        initial_cells = count_affected_cells(world_states[0].get_state_array())
        final_cells = count_affected_cells(final_state)
        total_time = (len(world_states) - 1) * dt
        stats["avg_spread_rate"] = ((final_cells - initial_cells) * DX**2) / total_time if total_time > 0 else 0.0
        
        # Fire duration
        duration_steps = len(world_states)

        for i, snapshot in enumerate(world_states):
            state_array = snapshot.get_state_array()

            if not np.any(state_array == BURNING):
                duration_steps = i
                break

        stats["duration_minutes"] = (duration_steps * dt) / 60
        
        # Fire shape (for non-zero wind)
        if parameters["wind_speed"] > 0:
            burned_mask = (final_state == BURNED) | (final_state == BURNING)
            
            if burned_mask.any():
                center_y, center_x = np.array(burned_mask.shape) // 2
                burned_coords = np.where(burned_mask)
                max_north = center_y - burned_coords[0].min()
                max_south = burned_coords[0].max() - center_y
                max_east = burned_coords[1].max() - center_x
                max_west = center_x - burned_coords[1].min()
                parallel_extent = max_north + max_south
                perpendicular_extent = max_east + max_west
                stats["elongation_ratio"] = parallel_extent / perpendicular_extent if perpendicular_extent > 0 else 1.0
        
            else:
                stats["elongation_ratio"] = 1.0
        
        else:
            stats["elongation_ratio"] = None
        
        # Peak spread rate timing
        spread_rates = []
        
        for i in range(1, len(world_states)):
            prev_cells = count_affected_cells(world_states[i-1].get_state_array())
            curr_cells = count_affected_cells(world_states[i].get_state_array())
            rate = (curr_cells - prev_cells) * (DX**2) / dt
        
            spread_rates.append(rate)
        
        if spread_rates:
            stats["peak_time_minutes"] = (np.argmax(spread_rates) * dt) / 60
        
        else:
            stats["peak_time_minutes"] = 0
        
        # Fuel consumption
        initial_fuel = world_states[0].get_fuel_array()
        final_fuel = world_states[-1].get_fuel_array()
        total_initial = initial_fuel.sum()
        stats["fuel_consumed_pct"] = ((total_initial - final_fuel.sum()) / total_initial * 100) if total_initial > 0 else 0
        
        # Max/avg temperature
        burning_temps = []
        
        for snapshot in world_states:
            temp_array = snapshot.get_temperature_array()
            state_array = snapshot.get_state_array()
            burning_mask = (state_array == BURNING)
            
            if burning_mask.any():
                burning_temps.append(temp_array[burning_mask].mean())
        
        stats["mean_burning_temp"] = np.mean(burning_temps) if burning_temps else AMBIENT_TEMPERATURE
        
        # Average temperature map (for heatmap plot)
        temp_sum = np.zeros_like(world_states[0].get_temperature_array())
        
        for snapshot in world_states:
            temp_sum += snapshot.get_temperature_array()
        
        stats["avg_temp_map"] = temp_sum / len(world_states)
        
        stats_list.append(stats)
    
    return stats_list

def load_all_statistics_incrementally(num_replicates=None, base_filename=None, dt=DT, base_path="."):
    """
    Load files one at a time, extract statistics, and discard world states to save memory.
    
    If no repeat files are found, falls back to discovering chunk files directly.
    
    Args:
        num_replicates: Number of repeat files to try loading
        base_filename: Base name for data files
        dt: Timestep in seconds
        base_path: Directory to search for files
    
    Returns:
        List of statistics dictionaries (much smaller than full world states).
    """
    
    if num_replicates is None:
        num_replicates = NUM_FILES_TO_LOAD
    if base_filename is None:
        base_filename = BASE_FILENAME
    
    all_stats = []
    files_found = False
    
    for i in range(1, num_replicates + 1):
        # Try both old format (base.pkl_repeat) and new format (base_repeat.pkl)
        filenames_to_try = [
            os.path.join(base_path, f"{base_filename}_repeat{i}.pkl"),
            os.path.join(base_path, f"{base_filename}.pkl_repeat{i}.pkl"),
        ]
        
        filename = None
        for fn in filenames_to_try:
            if os.path.exists(fn):
                filename = fn
                break
        
        if filename is None:
            print(f"  Warning: No repeat file found for repeat {i}, tried: {filenames_to_try}")
            continue
        
        files_found = True
        try:
            print(f"  Processing {filename}...")
            data = load_simulation_data(filename)
            
            # Process each simulation in the data
            for sim_idx, (params, result) in enumerate(data):
                # Check if result is a list of chunk filenames (strings)
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], str):
                    print(f"    Processing chunked simulation {sim_idx+1}/{len(data)} ({len(result)} chunks)...")
                    # Extract statistics incrementally from chunks without loading all at once
                    
                    # Initialize accumulators
                    first_snapshot = None
                    last_snapshot = None
                    total_snapshots = 0
                    temp_sum = None
                    burning_temps = []
                    spread_rates = []
                    prev_cells = None
                    duration_steps = 0
                    fire_still_burning = True
                    
                    # Process chunks one at a time
                    for chunk_idx, chunk_file in enumerate(result):
                        try:
                            chunk_data = load_simulation_data(chunk_file)
                        except FileNotFoundError:
                            print(f"      Warning: Chunk {chunk_file} not found, skipping")
                            continue
                        
                        for i, snapshot in enumerate(chunk_data):
                            # Store first and last snapshots
                            if first_snapshot is None:
                                first_snapshot = snapshot
                                prev_cells = count_affected_cells(snapshot.get_state_array())
                                temp_sum = np.zeros_like(snapshot.get_temperature_array(), dtype=np.float64)
                            
                            last_snapshot = snapshot
                            total_snapshots += 1
                            
                            # Accumulate temperature sum for average
                            temp_sum += snapshot.get_temperature_array()
                            
                            # Track burning temperatures
                            state_array = snapshot.get_state_array()
                            temp_array = snapshot.get_temperature_array()
                            burning_mask = (state_array == BURNING)
                            if burning_mask.any():
                                burning_temps.append(temp_array[burning_mask].mean())
                            
                            # Track spread rates
                            curr_cells = count_affected_cells(state_array)
                            if prev_cells is not None:
                                rate = (curr_cells - prev_cells) * (DX**2) / dt
                                spread_rates.append(rate)
                            prev_cells = curr_cells
                            
                            # Check if fire is still burning
                            if fire_still_burning and not np.any(state_array == BURNING):
                                fire_still_burning = False
                                duration_steps = total_snapshots
                        
                        del chunk_data  # Free memory immediately
                        if (chunk_idx + 1) % 5 == 0:
                            print(f"      Processed chunk {chunk_idx+1}/{len(result)}")
                    
                    # Skip this simulation if no snapshots were loaded
                    if first_snapshot is None or total_snapshots == 0:
                        print(f"    Warning: No data loaded for simulation {sim_idx+1}, skipping")
                        continue
                    
                    # Compute final statistics
                    stats = {"parameters": params.copy(),
                             "wind_speed": params["wind_speed"],
                             "moisture": params["moisture_level"],
                             "wind_direction": params["wind_direction"],
                            }
                    
                    # Final burned area
                    final_state = last_snapshot.get_state_array()
                    burned_cells = np.sum((final_state == BURNED) | (final_state == BURNING))
                    stats["burned_area"] = burned_cells * (DX ** 2)
                    
                    # Spread rate
                    initial_cells = count_affected_cells(first_snapshot.get_state_array())
                    final_cells = count_affected_cells(final_state)
                    total_time = (total_snapshots - 1) * dt
                    stats["avg_spread_rate"] = ((final_cells - initial_cells) * DX**2) / total_time if total_time > 0 else 0.0
                    
                    # Fire duration
                    if not fire_still_burning:
                        stats["duration_minutes"] = (duration_steps * dt) / 60
                    else:
                        stats["duration_minutes"] = (total_snapshots * dt) / 60
                    
                    # Fire shape
                    if params["wind_speed"] > 0:
                        burned_mask = (final_state == BURNED) | (final_state == BURNING)
                        if burned_mask.any():
                            center_y, center_x = np.array(burned_mask.shape) // 2
                            burned_coords = np.where(burned_mask)
                            max_north = center_y - burned_coords[0].min()
                            max_south = burned_coords[0].max() - center_y
                            max_east = burned_coords[1].max() - center_x
                            max_west = center_x - burned_coords[1].min()
                            parallel_extent = max_north + max_south
                            perpendicular_extent = max_east + max_west
                            stats["elongation_ratio"] = parallel_extent / perpendicular_extent if perpendicular_extent > 0 else 1.0
                        else:
                            stats["elongation_ratio"] = 1.0
                    else:
                        stats["elongation_ratio"] = None
                    
                    # Peak spread rate timing
                    stats["peak_time_minutes"] = (np.argmax(spread_rates) * dt) / 60 if spread_rates else 0
                    
                    # Fuel consumption
                    initial_fuel = first_snapshot.get_fuel_array()
                    final_fuel = last_snapshot.get_fuel_array()
                    total_initial = initial_fuel.sum()
                    stats["fuel_consumed_pct"] = ((total_initial - final_fuel.sum()) / total_initial * 100) if total_initial > 0 else 0
                    
                    # Mean burning temperature
                    stats["mean_burning_temp"] = np.mean(burning_temps) if burning_temps else AMBIENT_TEMPERATURE
                    
                    # Average temperature map
                    stats["avg_temp_map"] = temp_sum / total_snapshots
                    
                    all_stats.append(stats)
                    
                    # Free memory
                    del temp_sum, burning_temps, spread_rates, first_snapshot, last_snapshot
                else:
                    # Already in correct format (params, world_states)
                    sim_stats = extract_statistics_from_data([(params, result)], dt)
                    all_stats.extend(sim_stats)
            
            print(f"    Extracted stats from {len(data)} simulations (memory freed)")
            
            # Explicitly delete to free memory immediately
            del data
            
        except FileNotFoundError:
            print(f"  Warning: {filename} not found, skipping")
        except MemoryError:
            print(f"  ERROR: Out of memory processing {filename}")
            break
    
    # If no repeat files were found, try to load from chunk files directly
    if not files_found or len(all_stats) == 0:
        print("  No repeat files found, attempting to load from chunk files directly...")
        chunk_stats = load_statistics_from_chunks(base_path, base_filename, dt)
        all_stats.extend(chunk_stats)
    
    print(f"  Total statistics extracted: {len(all_stats)} simulations\n")
    return all_stats

def count_affected_cells(state_array):
    """
    Count cells that have been affected by fire (burning, lit out, or burned).
    """
    return np.sum((state_array == BURNING) | 
                  (state_array == LIT_OUT) | 
                  (state_array == BURNED))

def plot_spread_rate_comparison(all_stats, compute_std=None):
    """
    Plot average spread rate vs moisture for different wind speeds.
    """

    if compute_std is None:
        compute_std = COMPUTE_STD
    
    # Group by wind speed and moisture to collect replicates
    wind_groups = {}
    
    for stats in all_stats:
        wind_speed = stats["wind_speed"]
        moisture = stats["moisture"]
        avg_rate = stats["avg_spread_rate"]
        
        if wind_speed not in wind_groups:
            wind_groups[wind_speed] = {}

        if moisture not in wind_groups[wind_speed]:
            wind_groups[wind_speed][moisture] = []
        
        wind_groups[wind_speed][moisture].append(avg_rate)
    
    plt.figure(figsize=(12, 6))
    
    # Calculate means and optionally standard deviations
    for wind_speed in sorted(wind_groups.keys()):
        # Get all moisture-mean-std tuples and sort by moisture
        data_points = []
        
        for moisture in wind_groups[wind_speed].keys():
            rates = wind_groups[wind_speed][moisture]
            mean_val = np.mean(rates)
            std_val = np.std(rates) if compute_std and len(rates) > 1 else 0.0
            data_points.append((moisture, mean_val, std_val))
        
        # Sort by moisture level
        data_points.sort(key=lambda x: x[0])
        moistures = [m for m, _, _ in data_points]
        means = [mean for _, mean, _ in data_points]
        
        plt.plot(moistures, means, marker="s", markersize=10, 
                linewidth=2.5, label=f"Wind {wind_speed} m/s", alpha=0.8)
    
    plt.xlabel("Fuel moisture level", fontsize=15)
    plt.ylabel("Average spread rate (m^2/s)", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    title = "Fire spread rate vs moisture and wind speed (mean)"
    plt.title(title, fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("spread_rate_comparison.png")
    plt.show()

def plot_fire_shape_analysis(all_stats):
    """
    Analyze fire shape: comparing length/breadth spread.
    """
    wind_groups = {}

    for stats in all_stats:
        wind_speed = stats["wind_speed"]
        moisture = stats["moisture"]
        elongation_ratio = stats["elongation_ratio"]

        if wind_speed not in wind_groups:
            wind_groups[wind_speed] = {}

        if moisture not in wind_groups[wind_speed]:
            wind_groups[wind_speed][moisture] = []

        if wind_speed == 0:
            wind_groups[wind_speed][moisture].append(1.0)

        elif elongation_ratio is not None:
            wind_groups[wind_speed][moisture].append(elongation_ratio)

    plt.figure(figsize=(12, 6))
    
    for wind_speed in sorted(wind_groups.keys()):
        data_points = []
        
        for moisture in wind_groups[wind_speed].keys():
            elongations = wind_groups[wind_speed][moisture]
        
            if elongations:
                mean_val = np.mean(elongations)
                data_points.append((moisture, mean_val))
        
        data_points.sort(key=lambda x: x[0])
        moistures = [m for m, _ in data_points]
        means = [mean for _, mean in data_points]

        plt.plot(
            moistures,
            means,
            marker="D",
            markersize=10,
            linewidth=2.5,
            label=f"Wind {wind_speed} m/s",
            alpha=0.8
        )

    plt.xlabel("Fuel moisture level", fontsize=15)
    plt.ylabel("Fire elongation ratio (length/breadth)", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Fire shape directional bias vs wind", fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fire_shape_elongation.png")
    plt.show()

def plot_2d_heatmap_comparison(all_stats):
    """
    Create a 2d heatmap showing burned area as function of both moisture and wind.
    """
    # Create grid for heatmap
    moisture_wind_data = {}
    
    for stats in all_stats:
        wind_speed = stats["wind_speed"]
        moisture = stats["moisture"]
        burned_area = stats["burned_area"]
        
        moisture_wind_data[(moisture, wind_speed)] = burned_area
    
    # Get unique values
    moistures = sorted(set(m for m, w in moisture_wind_data.keys()))
    wind_speeds = sorted(set(w for m, w in moisture_wind_data.keys()))
    
    # Create 2D array
    heatmap_data = np.zeros((len(wind_speeds), len(moistures)))
    for i, ws in enumerate(wind_speeds):
        for j, m in enumerate(moistures):
            heatmap_data[i, j] = moisture_wind_data.get((m, ws), 0)
    
    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", origin="lower")
    
    cbar = plt.colorbar(im, ax=plt.gca(), label="Total burned area (m^2)")
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Total burned area (m^2)", fontsize=15)
    plt.xlabel("Fuel moisture level", fontsize=15)
    plt.ylabel("Wind speed (m/s)", fontsize=15)
    plt.title("Fire severity map: burned area vs moisture and wind", fontsize=15)
    
    # Set tick labels
    plt.xticks(range(len(moistures)), [f"{m:.2f}" for m in moistures], fontsize=12)
    plt.yticks(range(len(wind_speeds)), [f"{w}" for w in wind_speeds], fontsize=12)
    plt.tight_layout()
    plt.savefig("burned_area_2d_heatmap.png")
    plt.show()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Forest Fire Simulation Data Analysis (Memory-Efficient Mode)")
    print("="*60)
    print(f"Number of files to load: {NUM_FILES_TO_LOAD}")
    print(f"Base filename pattern: {BASE_FILENAME}_repeat<N>.pkl")
    print(f"Compute STD (error bars): {COMPUTE_STD}")
    print("="*60)
    print("\nExtracting statistics from simulation files...")
    print("(Processing files incrementally to save memory)\n")

    # Load statistics incrementally (process and discard to save memory)
    all_stats = load_all_statistics_incrementally(
        num_replicates=NUM_FILES_TO_LOAD, 
        base_filename=BASE_FILENAME,
        dt=DT
    )

    print(" 1. 2D intensity heatmap of average temperature...")
    plot_2d_intensity_heatmap(all_stats)

    if len(all_stats) == 0:
        print("ERROR: No data loaded. Please check that data files exist.")
        print(f"Expected files: {BASE_FILENAME}_repeat1.pkl to {BASE_FILENAME}_repeat{NUM_FILES_TO_LOAD}.pkl")
        exit(1)

    # Generate comparison plots
    print("\nGenerating comparison plots...")

    print("  2. Spread rate comparison...")
    plot_spread_rate_comparison(all_stats, compute_std=COMPUTE_STD)

    print("  2. Fire shape elongation...")
    plot_fire_shape_analysis(all_stats)
    
    print("  3. 2D parameter heatmap...")
    plot_2d_heatmap_comparison(all_stats)

    print("\nAll plots generated!")