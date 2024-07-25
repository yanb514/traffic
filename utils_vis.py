import matplotlib.pyplot as plt
import os
import pandas as pd
import xml.etree.ElementTree as ET
import utils_data_read as reader
import numpy as np
from collections import OrderedDict
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib.dates as mdates

# Path to your data file
def scatter_time_space(data_path, file_name, highlight_leaders=False):
    plt.rcParams.update({'font.size': 14})
    data_file = os.path.join(data_path, file_name)
    # Initialize variables to track the current vehicle's trajectory
    times = []
    positions = []
    speeds = []
    batch = 1000
    plt.figure(figsize=(8,6))

    # Read the data file line by line
    cnt=0
    with open(data_file, "r") as file:
        next(file)
        for line in file:
            
            # Split the line into columns
            columns = line.strip().split()
            # print(columns)
            # Extract vehicle ID, Frame ID, and LocalY
            # vehicle_id = columns[0]
            time = float(columns[1]) #* 0.1
            local_y = float(columns[3]) #% route_length
            mean_speed = float(columns[4])

            times.append(time)
            positions.append(local_y)
            speeds.append(mean_speed)

            # Check if we encountered data for a new vehicle
            if cnt>batch:
                plt.scatter(times, positions, c=speeds, s=0.1,vmin=0, vmax=30)
                # Start a new batch
                times = []
                positions = []
                speeds = []
                cnt =0

            cnt+=1

    # Add labels and legend
    plt.colorbar(label='Mean Speed')
    plt.xlabel("Time (sec)")
    plt.ylabel("Position (m)")
    plt.title("Time-space diagram")
    

    # go through the file a second time to plot the trip segments that don't have a leader
    if highlight_leaders:
        print("plotting no-leaders part")
        time_no_leader = []
        space_no_leader = []

        with open(data_file, "r") as file:
            for line in file:
                # Split the line into columns
                columns = line.strip().split()

                # Extract vehicle ID, Frame ID, and LocalY
                # leader_id = int(columns[9])
                # if leader_id == -1:
                vehicle_id = columns[0]
                if vehicle_id == "1.1":
                    time_no_leader.append(float(columns[1]) )
                    space_no_leader.append(float(columns[3]))

        plt.scatter(time_no_leader, space_no_leader, c="r", s=0.5,vmin=0, vmax=30)

    # Show the plot
    plt.tight_layout()
    plt.show()
    return


def plot_time_space(data_path, file_name, highlight_leaders=False):
    plt.rcParams.update({'font.size': 14})
    
    data_file = os.path.join(data_path, file_name)
    # Initialize variables to track the current vehicle's trajectory
    current_vehicle_id = None
    current_trajectory = []

    plt.figure(figsize=(16, 9))

    # Read the data file line by line
    with open(data_file, "r") as file:
        next(file)
        for line in file:
            
            # Split the line into columns
            columns = line.strip().split()
            # print(columns)
            # Extract vehicle ID, Frame ID, and LocalY
            vehicle_id = columns[0]
            time = float(columns[1]) * 0.1
            local_y = float(columns[3])
            mean_speed = float(columns[4])

            # Check if we encountered data for a new vehicle
            if vehicle_id != current_vehicle_id:
                # If so, plot the trajectory of the previous vehicle (if any)
                if current_vehicle_id is not None:
                    times, positions, speeds = zip(*current_trajectory)
                    plt.scatter(times, positions, c=speeds, s=0.1)
                
                # Start a new trajectory for the current vehicle
                current_vehicle_id = vehicle_id
                current_trajectory = [(time, local_y, mean_speed)]
            else:
                # Continue adding to the current vehicle's trajectory
                current_trajectory.append((time, local_y, mean_speed))

            

    # Plot the trajectory of the last encountered vehicle (if any)
    if current_vehicle_id is not None:
        times, positions, speeds = zip(*current_trajectory)
        plt.scatter(times, positions, c=speeds, s=0.1)
        
    # Add labels and legend
    # plt.colorbar(label='Mean Speed')
    plt.xlabel("Time (sec)")
    plt.ylabel("Position (m)")
    plt.title("Time-space diagram")

    # go through the file a second time to plot the trip segments that don't have a leader
    if highlight_leaders:
        print("plotting no-leaders part")
        time_no_leader = []
        space_no_leader = []

        with open(data_file, "r") as file:
            for line in file:
                # Split the line into columns
                columns = line.strip().split()

                # Extract vehicle ID, Frame ID, and LocalY
                leader_id = int(columns[9])
                if leader_id == -1:
                    time_no_leader.append(float(columns[1]) * 0.1)
                    space_no_leader.append(float(columns[3]))

        plt.scatter(time_no_leader, space_no_leader, c="r", s=0.5)

    # Show the plot
    plt.show()
    return





def plot_detector_data(xml_file, v=None, rho=None, q=None):
    '''
    plot the flow/density/speed relationship from xml_file (.out.xml)
    v, rho, q are background equilibrium macro quantities, derived from IDM parameters
    '''
    try:
        tree = ET.parse(xml_file)
    except:
        with open(xml_file, 'a') as file:
            file.write("</detector>" + '\n')

    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = {}
    for interval in root.findall('interval'):
        id_value = interval.get('id')
        occupancy = float(interval.get('occupancy'))
        flow = float(interval.get('flow'))
        
        if id_value not in data:
            data[id_value] = {'occupancy': [], 'flow': []}
        
        data[id_value]['occupancy'].append(occupancy)
        data[id_value]['flow'].append(flow)

    plt.figure(figsize=(10, 6))
    for id_value, values in data.items():
        plt.scatter(values['occupancy'], values['flow'], label=id_value)

    plt.xlabel('Occupancy (% of time the detector is occupied by vehicles during a given period)')
    plt.ylabel('Flow (#vehicles/hour)')
    plt.title('Detector Data')
    plt.legend()
    plt.show()

def visualize_fcd(fcd_file, lanes=None):
    # Parse the FCD XML file
    tree = ET.parse(fcd_file)
    root = tree.getroot()
    
    # Extract vehicle data
    data = []
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))
        for vehicle in timestep.findall('vehicle'):
            vehicle_id = vehicle.get('id')
            lane = vehicle.get('lane')
            x = float(vehicle.get('x'))
            y = float(vehicle.get('y'))
            speed = float(vehicle.get('speed'))
            data.append([time, vehicle_id, lane, x, y, speed])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['time', 'vehicle_id', 'lane', 'x', 'y', 'speed'])
    
    # Filter data for specific lanes if provided
    if lanes is not None:
        df = df[df['lane'].isin(lanes)]
    
    # Plot time-space diagrams
    plt.figure(figsize=(15, 10))
    
    if lanes is None:
        plt.title('Time-Space Diagram for All Lanes')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        scatter = plt.scatter(df['time'], df['x'], c=df['speed'], cmap='viridis', s=1)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Speed (m/s)')
    else:
        # for lane in lanes:
            # lane_data = df[df['lane'] == lane]
        lane_data = df[df['lane'].isin(lanes)]
        if not lane_data.empty:
            # plt.subplot(len(lanes), 1, lanes.index(lane) + 1)
            plt.title(f'Time-Space Diagram for Lane: {lane}')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (m)')
            scatter = plt.scatter(lane_data['time'], lane_data['x'], c=lane_data['speed'], cmap='viridis', s=1)
            cbar = plt.colorbar(scatter)
            cbar.set_label('Speed (m/s)')
    
    plt.tight_layout()
    plt.show()


def scatter_fcd(fcd_file, lanes=None):
    # works on I-24 new only
    # Parse the FCD XML file
    tree = ET.parse(fcd_file)
    root = tree.getroot()
    dt = 30 # time batch size for scatter
    start_time = 0

    # Extract vehicle data
    time_arr = []
    x_arr = []
    y_arr = []
    v_arr = []
    x0, y0 = 4048.27, 8091.19
    exclude_edges = ["19447013", "19440938", "27925488", "782177974", "782177973", "19446904"]
    
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))
        if time > 10800:
            break
        if time % 20 == 0:
            for vehicle in timestep.findall('vehicle'):
                if vehicle.get("lane").split("_")[0] not in exclude_edges:
                    x = float(vehicle.get('x'))
                    y = float(vehicle.get('y'))  # y is parsed but not used in this plot
                    speed = float(vehicle.get('speed'))
                    time_arr.append(time)
                    x_arr.append(x)
                    y_arr.append(y)
                    v_arr.append(speed)
            
    # Convert lists to numpy arrays for faster plotting
    time_arr = np.array(time_arr)
    x_arr = np.array(x_arr)
    y_arr = np.array(y_arr)
    v_arr = np.array(v_arr)
    distances = np.sqrt((x_arr - x0)**2 + (y_arr - y0)**2)
    
    # Plot time-space diagrams
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(time_arr, distances, c=v_arr, cmap='viridis', s=0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Speed (m/s)')

    plt.title('Time-Space Diagram for All Lanes')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
        
    plt.tight_layout()
    plt.show()


def scatter_fcd_i24(fcd_file, lanes=None):
    # Parse the FCD XML file
    tree = ET.parse(fcd_file)
    root = tree.getroot()
    x_offset = -1000

    # Extract vehicle data
    time_arr = []
    x_arr = []
    # y_arr = []
    v_arr = []
    # x0, y0 = 4048.27, 8091.19
    exclude_edges = ["E2", "E4", "E6"]
    
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))
        if time > 10800:
            break
        # if time % 20 == 0:
        for vehicle in timestep.findall('vehicle'):
            if vehicle.get("lane").split("_")[0] not in exclude_edges:
                x = float(vehicle.get('x'))
                # y = float(vehicle.get('y'))  # y is parsed but not used in this plot
                speed = float(vehicle.get('speed'))
                time_arr.append(time)
                x_arr.append(x)
                # y_arr.append(y)
                v_arr.append(speed)
            
    # Convert lists to numpy arrays for faster plotting
    time_arr = np.array(time_arr) 
    start_time = pd.Timestamp('2023-11-13 05:00:00')
    time_arr = pd.to_datetime(start_time) + pd.to_timedelta(time_arr, unit='s')
    x_arr = 57.6 - (np.array(x_arr) - x_offset)/1609.34 # start at 0
    
    # x_arr = dist/1609.34- (x_arr -x_offset)/1609.34 +57 # meter to mile
    v_arr = np.array(v_arr) * 2.23694 # m/s to mph

    print("plotting scatter...")
    # Plot time-space diagrams
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(time_arr, x_arr, c=v_arr, cmap='viridis', s=0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Speed (mph)')

    plt.title('Time-Space Diagram for All Lanes')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (mi)')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.gca().invert_yaxis()
        
    plt.tight_layout()
    plt.show()


def plot_rds_vs_sim(rds_dir, sumo_dir, measurement_locations, quantity="volume"):
    '''
    rds_dir: directory for filtered RDS data
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations a list of detectors
    quantity: "volume", "speed" or "occupancy"
    '''
    # Read and extract data
    # _dict: speed, volume, occupancy in a dictionary, each quantity is a matrix [N_det, N_time]
    sim_dict = reader.extract_sim_meas(measurement_locations=measurement_locations, file_dir=sumo_dir)
    rds_dict = reader.rds_to_matrix(rds_file=rds_dir, det_locations=measurement_locations)
    unit_dict = {
        "speed": "mph",
        "volume": "nVeh/hr",
        "occupancy": "%"
    }
    time_interval = 300  # seconds
    start_time_rds = pd.Timestamp('05:00')  # Midnight
    start_time_sim = pd.Timestamp('05:00')  # 5:00 AM
    start_idx_rds = int(5*3600/time_interval)
    
    
    num_points_rds = min(len(rds_dict[quantity][0, :]), int(3*3600/time_interval))
    num_points_sim = min(len(sim_dict[quantity][0, :]), int(3*3600/time_interval)) # at most three hours of simulation
    
    # Create time indices for the x-axes
    time_index_rds = pd.date_range(start=start_time_rds, periods=num_points_rds, freq=f'{time_interval}s')
    time_index_sim = pd.date_range(start=start_time_sim, periods=num_points_sim, freq=f'{time_interval}s')

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 12))
    axes = axes.flatten()
    for i, det in enumerate(measurement_locations):
        
        axes[i].plot(time_index_rds, rds_dict[quantity][i,start_idx_rds:start_idx_rds+num_points_rds],  'go--', label="obs")
        axes[i].plot(time_index_sim, sim_dict[quantity][i,:num_points_sim],  'rs--', label="sim")
        parts = det.split('_')
        axes[i].set_title( f"MM{parts[0]}.{parts[1]} lane {int(parts[2])+1}")

        # Format the x-axis
        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylabel(unit_dict[quantity])


    axes[0].legend()
    plt.tight_layout()
    plt.show()

    return

def format_yticks(y, pos):
    if y >= 1000:
        return f'{y / 1000:.1f}k'
    else:
        return f'{y:.0f}'


def plot_sim_vs_sim(sumo_dir, measurement_locations, quantity="volume"):
    '''
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations: a list of detectors
    quantity: "volume", "speed" or "occupancy"
    '''
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    formatter = FuncFormatter(format_yticks)


    # Read and extract data
    sim1_dict = reader.extract_sim_meas(measurement_locations=measurement_locations, file_dir=sumo_dir)
    sim2_dict = reader.extract_sim_meas(measurement_locations=["trial_" + location for location in measurement_locations], file_dir=sumo_dir)
    
    unit_dict = {
        "speed": "mph",
        "volume": "nVeh / hr",
        "occupancy": "%"
    }
    
    # start_time_rds = pd.Timestamp('00:00')  # Midnight
    # start_time_sim = pd.Timestamp('00:00')  # Midnight
    time_interval = 50  # seconds, set as detector frequency
    
    num_points_rds = len(sim1_dict[quantity][0, :])
    # num_points_sim = len(sim2_dict[quantity][0, :])
    
    # Create time indices for the x-axes
    lanes = sorted(set(int(location.split('_')[1]) for location in measurement_locations))
    detectors = list(OrderedDict.fromkeys(location.split('_')[0] for location in measurement_locations))
    print(detectors)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=len(lanes), ncols=len(detectors), figsize=(14, 10))
    # Determine the y-axis range across all plots
    y_min = 0
    y_max = max(sim1_dict[quantity].max(), sim2_dict[quantity].max()) #+ 200

    
    for lane in lanes:
        leftmost_idx = 99
        for detector in detectors:
            location = f"{detector}_{lane}"
            if location in measurement_locations:
                i = measurement_locations.index(location)
                row = lanes.index(lane)
                col = detectors.index(detector)
                leftmost_idx = min(leftmost_idx, col)
                ax = axes[row, col]
                ax.plot(sim1_dict[quantity][i, :], 'go--', label="ground truth")
                ax.plot(sim2_dict[quantity][i, :], 'rs--', label="default")
                err_abs = np.sum(np.abs(sim1_dict[quantity][i, :] - sim2_dict[quantity][i, :])) / len(sim1_dict[quantity][i, :])
                title = f"{detector.capitalize()} lane {lane + 1}"
                ax.set_title(title, fontsize=22)
                print(f"{detector} lane {lane + 1}, abs.err {err_abs:.1f}")
                
                
                # Format the x-axis
                # ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
                # ax.xaxis.set_major_locator(plt.matplotlib.dates.MinuteLocator(interval=2))
                # ax.tick_params(axis='x', rotation=45)
                # Set the same y-axis range for all subplots
                ax.set_ylim(y_min, y_max)
                # ax.set_xlim(0-0.1, num_points_rds*time_interval/60+0.1)
                ax.set_xlabel("Time (min)")
                ax.set_xticks(range(0, num_points_rds, 2))
                # 

                
            else:
                # If there's no data for this detector-lane combination, turn off the subplot
                fig.delaxes(axes[lanes.index(lane), detectors.index(detector)])
          
        for col_idx, _ax in enumerate(axes[row]):
            if col_idx == leftmost_idx:
                _ax.set_ylabel(unit_dict[quantity])
                _ax.yaxis.set_tick_params(labelleft=True)
                _ax.yaxis.set_major_formatter(formatter)
            else:
                _ax.set_yticklabels([])
    
    # Adjust layout
    axes[0,1].legend()
    plt.tight_layout()
    plt.show()

    return


def read_asm(asm_file):

    # Initialize an empty DataFrame to store the aggregated results
    aggregated_data = pd.DataFrame()

    # Define a function to process each chunk
    def process_chunk(chunk):
        # Calculate aggregated volume, occupancy, and speed for each row
        chunk['total_volume'] = chunk[['lane1_volume', 'lane2_volume', 'lane3_volume', 'lane4_volume']].sum(axis=1)*120 # convert from veh/30s to veh/hr
        chunk['total_occ'] = chunk[['lane1_occ',  'lane2_occ','lane3_occ',  'lane4_occ']].mean(axis=1)
        chunk['total_speed'] = chunk[['lane1_speed',  'lane2_speed', 'lane3_speed','lane4_speed']].mean(axis=1)
        return chunk[['unix_time', 'milemarker', 'total_volume', 'total_occ', 'total_speed']]

    # Read the CSV file in chunks and process each chunk
    chunk_size = 10000  # Adjust the chunk size based on your memory capacity
    for chunk in pd.read_csv(asm_file, chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        aggregated_data = pd.concat([aggregated_data, processed_chunk], ignore_index=True)

    # Define the range of mile markers to plot
    milemarker_min = 54.1
    milemarker_max = 57.6
    start_time = aggregated_data['unix_time'].min()+3600 # data starts at 4AM CST
    
    end_time = start_time + 3*3600 # only select the first 3 hours

    # Filter milemarker within the specified range
    filtered_data = aggregated_data[
        (aggregated_data['milemarker'] >= milemarker_min) &
        (aggregated_data['milemarker'] <= milemarker_max) &
        (aggregated_data['unix_time'] >= start_time) &
        (aggregated_data['unix_time'] <= end_time)
    ]
    # Convert unix_time to datetime if needed and extract hour (UTC to Central standard time in winter)
    filtered_data['unix_time'] = pd.to_datetime(filtered_data['unix_time'], unit='s') - pd.Timedelta(hours=6)

    # Pivot the data for heatmaps
    volume_pivot = filtered_data.pivot(index='milemarker', columns='unix_time', values='total_volume')
    occ_pivot = filtered_data.pivot(index='milemarker', columns='unix_time', values='total_occ')
    speed_pivot = filtered_data.pivot(index='milemarker', columns='unix_time', values='total_speed')

    # Generate y-ticks based on the range of mile markers
    # yticks = range(milemarker_min, milemarker_max + 1)

    # Plot the heatmaps
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 3, 1)
    sns.heatmap(volume_pivot, cmap='viridis', vmin=0, vmax=8000) # convert from 
    plt.title('Volume (nVeh/hr)')
    plt.xlabel('Time (hour of day)')
    plt.ylabel('Milemarker')
    # plt.yticks(ticks=yticks, labels=yticks)
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    sns.heatmap(occ_pivot, cmap='viridis', vmin=0)
    plt.title('Occupancy (%)')
    plt.xlabel('Time (hour of day)')
    plt.ylabel('Milemarker')
    # plt.yticks(ticks=yticks, labels=yticks)
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    sns.heatmap(speed_pivot, cmap='viridis', vmin=0, vmax=80)
    plt.title('Speed (mph)')
    plt.xlabel('Time (hour of day)')
    plt.ylabel('Milemarker')
    # plt.yticks(ticks=yticks, labels=yticks)
    plt.xticks(rotation=45)

    # Adjust x-axis labels to show integer hours
    for ax in plt.gcf().axes:
        # Format the x-axis
        x_labels = ax.get_xticks()
        # new_labels = [pd.to_datetime(volume_pivot.columns[int(l)]).strftime('%H:%M') for l in x_labels if l >= 0 and l < len(volume_pivot.columns)]
        new_labels = [
                pd.to_datetime(volume_pivot.columns[int(l)]).strftime('%H:%M') 
                if i % 2 == 0 else ''
                for i, l in enumerate(x_labels) 
                if l >= 0 and l < len(volume_pivot.columns)
            ]
        ax.set_xticklabels(new_labels)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    # plot_detector_data("det_55_3_2.out.xml")
    measurement_locations = [
                            # '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                             '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4',
                             '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
                             '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                             '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                             '54_1_0', '54_1_1', '54_1_2', '54_1_3' ]
    
    rds_dir = r'C:\Users\yanbing.wang\Documents\traffic\data\RDS\I24_WB_52_60_11132023.csv'
    sumo_dir = r'C:\Users\yanbing.wang\Documents\traffic\sumo\I24scenario'
    plot_rds_vs_sim(rds_dir, sumo_dir, measurement_locations, quantity="volume")

