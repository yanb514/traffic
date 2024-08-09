import gzip
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d

def extract_mile_marker(link_name):
    "link_name: R3G-00I24-59.7W Off Ramp (280)"
    matches = re.findall(r'-([0-9]+(?:\.[0-9]+)?)', link_name)
    return float(matches[1]) if len(matches) > 1 else None

def extract_lane_number(lane_name):
    match = re.search(r'Lane(\d+)', lane_name)
    return int(match.group(1)) if match else None

def is_i24_westbound_milemarker(link_name, min_mile, max_mile):
    if 'I24' not in link_name or 'W' not in link_name:
        return False
    mile_marker = extract_mile_marker(link_name)
    if mile_marker is None:
        return False
    return min_mile <= mile_marker <= max_mile

def safe_float(value):
    try:
        return float(value)
    except:
        return None

def read_and_filter_file(file_path, write_file_path, startmile, endmile):
    '''
    read original dat.gz file and select I-24 MOTION WB portion
    write rows into a new file
    | timestamp | milemarker | lane | speed | volume | occupancy |
    '''
    selected_fieldnames = ['timestamp', 'link_name', 'milemarker', 'lane', 'speed', 'volume', 'occupancy']
    open_func = gzip.open if file_path.endswith('.gz') else open
    with open_func(file_path, mode='rt') as file:
        reader = csv.DictReader(file)
        with open(write_file_path, mode='w', newline='') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=selected_fieldnames)
            writer.writeheader()
            for row in reader:
                if is_i24_westbound_milemarker(row[' link_name'], startmile, endmile): # 58-63
                    selected_row = {
                        'timestamp': row['timestamp'],
                        'link_name': row[' link_name'],
                        'milemarker': extract_mile_marker(row[' link_name']),
                        'lane': extract_lane_number(row[' lane_name']),
                        'speed': safe_float(row[' speed']),
                        'volume': safe_float(row[' volume']),
                        'occupancy': safe_float(row[' occupancy'])
                    }
                    writer.writerow(selected_row)

def interpolate_zeros(arr):
    arr = np.array(arr)
    interpolated_arr = arr.copy()
    
    for i, row in enumerate(arr):
        zero_indices = np.where(row < 4)[0]
        
        if len(zero_indices) > 0:
            # Define the x values for the valid (non-zero) data points
            x = np.arange(len(row))
            valid_indices = np.setdiff1d(x, zero_indices)
            
            if len(valid_indices) > 1:  # Ensure there are at least two points to interpolate
                # Create the interpolation function based on valid data points
                interp_func = interp1d(x[valid_indices], row[valid_indices], kind='linear', fill_value="extrapolate")
                
                # Replace the zero values with interpolated values
                interpolated_arr[i, zero_indices] = interp_func(zero_indices)
    
    return interpolated_arr

def rds_to_matrix(rds_file, det_locations ):
    '''
    Read RDS data from a CSV file and output a matrix of [N_dec, N_time] size,
    where N_dec is the number of detectors and N_time is the number of aggregated
    time intervals of 5 minutes.
    
    Parameters:
    - rds_file: Path to the RDS data CSV file.
    - det_locations: List of strings representing RDS sensor locations in the format "milemarker_lane", e.g., "56_7_3".
    
    Returns:
    - matrix: A numpy array of shape [N_dec, N_time].

    SUMO lane is 0-indexed, while RDS lanes are 1-index... ugh
    '''
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(rds_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    milemarkers = [round(float(".".join(location.split('_')[:2])),1) for location in det_locations]
    lanes = [int(location.split('_')[-1])+1 for location in det_locations]
    macro_data = {"speed": [], "volume": [], "occupancy": []}

    for milemarker, lane in zip(milemarkers, lanes):
        # Filter rows based on milemarker and lane
        filtered_df = df[(df['milemarker'] == milemarker) & (df['lane'] == lane)]
        
        # Aggregate by 5-minute intervals (assuming 'timestamp' is already in datetime format)
        if filtered_df.empty:
            print(f"No RDS data for milemarker {milemarker} lane {lane}")
        else:
            aggregated = filtered_df.groupby(pd.Grouper(key='timestamp', freq='5min')).agg({
                'speed': 'mean',
                'volume': 'sum',
                'occupancy': 'mean'
            }).reset_index()

            macro_data["speed"].append(aggregated["speed"].values)
            macro_data["volume"].append(aggregated["volume"].values * 12) # convert to vVeh/hr
            macro_data["occupancy"].append(aggregated["occupancy"].values)
            # Add identifier columns
            # aggregated['milemarker'] = milemarker
            # aggregated['lane'] = lane
            
            # # Append to aggregated_data list
            # aggregated_data.append(aggregated)
    
    # Concatenate all aggregated dataframes into a single dataframe
    # result_df = pd.concat(aggregated_data, ignore_index=True)
    # result_df.to_csv("out.csv")
    macro_data["speed"] = np.vstack(macro_data["speed"]) # [N_dec, N_time]
    macro_data["volume"] = np.vstack(macro_data["volume"]) # [N_dec, N_time]
    macro_data["occupancy"] = np.vstack(macro_data["occupancy"]) # [N_dec, N_time]

    # postprocessing
    macro_data["volume"] = interpolate_zeros(macro_data["volume"])
    macro_data["flow"] = macro_data["volume"]
    macro_data["density"] = macro_data["flow"]/macro_data["speed"]

    return macro_data

def extract_sim_meas(measurement_locations, file_dir = ""):
    """
    Extract simulated traffic measurements (Q, V, Occ) from SUMO detector output files (xxx.out.xml).
    Q/V/Occ: [N_dec x N_time]
    measurement_locations: a list of strings that map detector IDs
    """
    # Initialize an empty list to store the data for each detector
    detector_data = {"speed": [], "volume": [], "occupancy": []}

    for detector_id in measurement_locations:
        # Construct the filename for the detector's output XML file
        # print(f"reading {detector_id}...")
        filename = os.path.join(file_dir, f"det_{detector_id}.out.xml")
        
        # Check if the file exists
        if not os.path.isfile(filename):
            print(f"File {filename} does not exist. Skipping this detector.")
            continue
        
        # Parse the XML file
        tree = ET.parse(filename)
        root = tree.getroot()

        # Initialize a list to store the measurements for this detector
        speed = []
        volume = []
        occupancy = []

        # Iterate over each interval element in the XML
        for interval in root.findall('interval'):
            # Extract the entered attribute (number of vehicles entered in the interval)
            speed.append(float(interval.get('speed')) * 2.237) # convert m/s to mph
            volume.append(float(interval.get('flow')))
            occupancy.append(float(interval.get('occupancy')))
        
        # Append the measurements for this detector to the detector_data list
        detector_data["speed"].append(speed)
        detector_data["volume"].append(volume)
        detector_data["occupancy"].append(occupancy)
    
    for key, val in detector_data.items():
        detector_data[key] = np.array(val)
        # print(val.shape)
    
    return detector_data








def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = []
    for timestep in root.findall('timestep'):
        for vehicle in timestep.findall('vehicle'):
            vehicle_id = vehicle.get('id')
            time = timestep.get('time')
            lane_id = vehicle.get('lane')
            local_y = vehicle.get('x', '-1')
            mean_speed = vehicle.get('speed', '-1')
            mean_accel = vehicle.get('accel', '-1')  # Assuming accel is mean acceleration
            veh_length = vehicle.get('length', '-1')
            veh_class = vehicle.get('type', '-1')
            follower_id = vehicle.get('pos', '-1')  # Assuming pos is follower ID
            leader_id = vehicle.get('slope', '-1')  # Assuming slope is leader ID
            
            row = [vehicle_id, time, lane_id, local_y, mean_speed, mean_accel, veh_length, veh_class, follower_id, leader_id]
            # data.append([str(item) for item in row])
            data.append([" ".join(str(num) for num in row)])
    
    return data

# Function to write data to CSV
def write_csv(data, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['VehicleID', 'Time', 'LaneID', 'LocalY', 'MeanSpeed', 'MeanAccel', 'VehLength', 'VehClass', 'FollowerID', 'LeaderID'])
        # Write rows
        writer.writerows(data)


def det_to_csv(xml_file, suffix=""):

    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Open a CSV file for writing
    csv_file_name = xml_file.split(".")[-3]
    with open(f'{csv_file_name}{suffix}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        header = ["begin", "end", "id", "nVehContrib", "flow", "occupancy", "speed", "harmonicMeanSpeed", "length", "nVehEntered"]
        writer.writerow(header)
        
        # Write the data rows
        for interval in root.findall('interval'):
            row = [
                float(interval.get("begin")),
                float(interval.get("end")),
                interval.get("id"),
                int(interval.get("nVehContrib")),
                float(interval.get("flow")),
                float(interval.get("occupancy")),
                float(interval.get("speed")),
                float(interval.get("harmonicMeanSpeed")),
                float(interval.get("length")),
                int(interval.get("nVehEntered"))
            ]
            writer.writerow(row)

    return

def fcd_to_csv_byid(xml_file, csv_file):
    print(f"parsing {xml_file}...")
    data = parse_xml(xml_file)
    print(f"writing {csv_file}...")
    write_csv(data, csv_file)
    return

def vis_rds_lines(write_file_path):
    df = pd.read_csv(write_file_path)

    # Convert the 'speed' column to numeric, forcing errors to NaN (to handle missing values)
    df['occupancy'] = pd.to_numeric(df['speed'], errors='coerce')
    # Convert 'timestamp' to datetime for proper plotting
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')#.dt.strftime('%H:%M:%S')

    # Group by 'timestamp' and 'milemarker' and calculate the average speed, ignoring NaN values
    average_speeds = df.groupby(['timestamp', 'milemarker'])['speed'].mean().reset_index()

    # Pivot the table to have timestamps as index and milemarkers as columns
    pivot_df = average_speeds.pivot(index='timestamp', columns='milemarker', values='speed')
    # print(pivot_df.head())

    # Plot the time series of average speeds for each mile marker
    plt.figure(figsize=(12, 8))
    for milemarker in pivot_df.columns[:15]:
        # plt.plot(pivot_df.index, pivot_df[milemarker].dropna(), label=f'Mile Marker {milemarker}')
        # print(pivot_df.index[pivot_df[milemarker].notna()])
        plt.plot(pivot_df.index[pivot_df[milemarker].notna()], pivot_df[milemarker].dropna(), label=f'Mile Marker {milemarker}', alpha=0.5)
        

    # plt.title('Time Series of Average Speeds for Each Mile Marker')
    plt.xlabel('Timestamp')
    # plt.ylabel('Average Speed (mph)')
    plt.ylabel('Occupancy (%)')
    plt.legend(title='Mile Marker', loc='upper right')
    plt.grid(True)
    date_form = DateFormatter("%H:%M:%S")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

def vis_rds_color(write_file_path, lane_number=None):
    # Read the CSV file
    quantity = "volume"
    df = pd.read_csv(write_file_path)

    # Convert 'speed' to numeric, forcing errors to NaN (to handle missing values)
    df[quantity] = pd.to_numeric(df[quantity], errors='coerce')

    # Convert 'timestamp' to proper datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')
    if lane_number is None:
        df = df[(df['timestamp'].dt.hour >= 5) & (df['timestamp'].dt.hour <= 10)]
    else:
        df = df[(df['timestamp'].dt.hour >= 5) & (df['timestamp'].dt.hour <= 10) & (df['lane'] == lane_number)]

    # Create a pivot table to reshape the data for the colormap
    pivot_df = df.pivot_table(index='milemarker', columns='timestamp', values=quantity)
    pivot_df = pivot_df.fillna(method='ffill', axis=1) # fill previous timestamp values

    plt.figure(figsize=(15, 10))

    # Use pcolormesh to create the colormap
    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    plt.pcolormesh(X, Y, pivot_df.values, shading='auto', cmap='viridis')
    plt.colorbar(label=quantity)

    # Format the x-axis to show only HH:MM:SS
    plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.gca().set_yticks(pivot_df.index)
    plt.gca().set_yticklabels(pivot_df.index)

    # Set labels
    plt.title(f'{quantity} Colormap Across Time and Milemarkers')
    plt.xlabel('Time of the Day')
    plt.ylabel('Milemarkers')

    # Reverse the y-axis to have higher milemarkers at the bottom
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Show the plot
    plt.show()

    return

def plot_ramp_volumes(write_file_path):
    # Read the CSV file
    df = pd.read_csv(write_file_path)

    # Convert 'volume' to numeric, forcing errors to NaN (to handle missing values)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

    # Convert 'timestamp' to proper datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')

    # Filter data for on ramps and off ramps
    # on_ramps = df[df['link_name'].str.contains('On Ramp', case=False, na=False)]
    on_ramps = df[df['link_name'].str.contains('On Ramp|Auxiliary', case=False, na=False)]

    # off_ramps = df[df['link_name'].str.contains('Off Ramp', case=False, na=False)]
    # Group by "milemarker" and plot each group
    grouped = on_ramps.groupby('milemarker')

    plt.figure(figsize=(12, 8))
    for milemarker, group_data in grouped: 
        plt.plot(group_data['timestamp'], group_data['volume'], label=f'Milemarker {milemarker}', alpha=0.5)

    plt.title(f'Time Series of Volume for On Ramps')
    plt.xlabel('Timestamp')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)

    # Format the x-axis to show only HH:MM:SS
    date_form = DateFormatter("%H:%M:%S")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

    file_path = r'C:\Users\yanbing.wang\Documents\traffic\data\RDS\R3_TSS-11132023--1.dat.gz'
    write_file_path = r'C:\Users\yanbing.wang\Documents\traffic\data\RDS\I24_WB_52_60_11132023.csv'
    # read_and_filter_file(file_path, write_file_path, 52, 57.5)
    # vis_rds_lines(write_file_path=write_file_path)
    # vis_rds_color(write_file_path=write_file_path, lane_number=None)
    # plot_ramp_volumes(write_file_path)

    measurement_locations = [
                            # '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                             '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4', 
                             '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
                             '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                             '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                             '54_1_0', '54_1_1', '54_1_2', '54_1_3' ]
    
    matrix = rds_to_matrix(write_file_path, measurement_locations)
    