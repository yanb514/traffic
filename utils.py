"""
Bayesian calibration of car-following models
try IDM first
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import arviz as az
import matplotlib


def load_cf_data(data_file, pair_id_file, pair_id_idx):
    '''
    Arguments:
        data_file: NGSIM-like trajectory data
        pair_id_file: 
        pair_id_idx: 
    Return:
        cf_data: [dictionary]
            timestamps:
            foll_v:
            foll_a:
            lead_v:
            foll_p:
            lead_p:
            s_gap:
    '''
    pair_idx = 0
    # find the pair info
    with open(pair_id_file, mode='r') as file:
        for line in file:
            if pair_idx == pair_id_idx:
                columns = line.strip().split()
                foll_id = int(columns[0])
                lead_id = int(columns[1])
                start_time = float(columns[2]) 
                end_time = float(columns[3]) 
                break
            pair_idx += 1

    print(f"Getting CF data for foll:{foll_id}, lead:{lead_id}, time:{start_time}-{end_time}")
    # if end_time - start_time < 15:
    #     print("CF event less than 15 sec.")
    #     return
    
    # get the car-following data
    cf_data = defaultdict(list)

    with open(data_file, mode='r') as input_file:
        for line in input_file:
            # Split the line into columns
            columns = line.strip().split()

            # extract current info
            vehicle_id = int(columns[0])
            leader_id = int(columns[9])
            follower_id = int(columns[8])
            timestamp = float(columns[1]) * 0.1

            # get follower info
            if vehicle_id == foll_id and leader_id == lead_id \
                and timestamp >= start_time and timestamp <= end_time:
                
                foll_v_val = float(columns[4])
                foll_p_val = float(columns[3])
                foll_a_val = float(columns[5])
                cf_data["timestamps"].append(timestamp)
                cf_data["foll_v"].append(foll_v_val)
                cf_data["foll_p"].append(foll_p_val)
                cf_data["foll_a"].append(foll_a_val)

            # get leader info (leader could appear first)
            elif vehicle_id == lead_id and foll_id == follower_id \
                and timestamp >= start_time and timestamp <= end_time:
                cf_data["lead_v"].append(float(columns[4]))
                cf_data["lead_p"].append(float(columns[3]))
                cf_data["lead_a"].append(float(columns[5]))
                leader_length = float(columns[6])
             

    # Finally calculate the space gap in between
    cf_data["s_gap"] = [cf_data["lead_p"][i]-cf_data["foll_p"][i]-leader_length for i in range(len(cf_data["timestamps"]))]
    for key in cf_data:
        cf_data[key] = np.array(cf_data[key])

    cf_data["foll_id"] = foll_id
    cf_data["lead_id"] = lead_id
    print("CF event duration: {:.2f} sec, data size: {}".format(end_time-start_time, len(cf_data["timestamps"])))
    return cf_data


def plot_cf_data(cf_data):
    # Create a figure with two subplots
    font = {
        'size'   : 20}

    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot: timestamps vs. foll_p and timestamps vs. lead_p
    axs[0].plot(cf_data['timestamps'], cf_data['s_gap'], label='Space gap')
    # axs[0].plot(cf_data['timestamps'], cf_data['lead_p'], label='Leader Position')
    try:
        axs[0].plot(cf_data['timestamps'], cf_data['s_gap_sim'], label='Space gap (sim)')
    except:
        pass
    axs[0].set_xlabel('Time (sec)')
    axs[0].set_ylabel('Space gap (m)')
    axs[0].set_title('Space gap vs. Time')
    axs[0].legend()

    # Second subplot: timestamps vs. foll_v and timestamps vs. lead_v
    axs[1].plot(cf_data['timestamps'], cf_data['lead_v'], label='Leader Velocity')
    axs[1].plot(cf_data['timestamps'], cf_data['foll_v'], label='Follower Velocity')
    try:
        axs[1].plot(cf_data['timestamps'], cf_data['foll_v_sim'], label='Follower Velocity (sim)')
    except:
        pass
    axs[1].set_xlabel('Time (sec)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('Velocity vs. Time')
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()
    return




if __name__ == "__main__":

    data_path = "data/"
    file_name = "DATA (NO MOTORCYCLES).txt"
    data_file = os.path.join(data_path, file_name)
    pair_id_file = os.path.join(data_path, "ngsim_id_pair.txt")

    cf_data = load_cf_data(data_file, pair_id_file, pair_id_idx=3)
    plot_cf_data(cf_data)
    