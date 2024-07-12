"""
This file generates the training and validation results from macrosopic data stored in .pkl
"""
import pickle
import numpy as np
import os
import os
import numpy as np
import sys

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import utils_data_read as reader




pre = "macro_fcd_onramp"
suf  = "_byid.pkl"
exp = {
    "gt": pre + "_gt" + suf,
    "default": pre + "" + suf,
    "1a": pre + "_cf_q" + suf,
    "1b": pre + "_cf_v" + suf,
    "1c": pre + "_cf_rho" + suf,
    "2a": pre + "_lc_q" + suf,
    "2b": pre + "_lc_v" + suf,
    "2c": pre + "_lc_rho" + suf,
    "3a": pre + "_cflc_q" + suf,
    "3b": pre + "_cflc_v" + suf,
    "3c": pre + "_cflc_rho" + suf,
}

def training_rmse(exp_name):

    '''
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations: a list of detectors
    quantity: "volume", "speed" or "occupancy"
    '''
    # if "_q" in exp_name:
    #     quantity = "volume"
    # elif "_v" in exp_name:
    #     quantity = "speed"
    # elif "_rho" in exp_name:
    #     quantity = "occupancy"

    # density = (occupancy * 1000) / (vehicle_length + detector_length)

    # Read and extract data
    print("Training RMSE (detectors)")
    sim2_dict = reader.extract_sim_meas(measurement_locations=["trial_" + location for location in measurement_locations], file_dir=sumo_dir)

    size = sim1_dict["volume"].size

    arr_without_nan = np.nan_to_num((sim1_dict["volume"] - sim2_dict["volume"]), nan=0.0)
    norm = np.linalg.norm(arr_without_nan)/size
    print("Volume q: {:.2f}".format(norm))  #convert veh/s to veh/hr
          
    arr_without_nan = np.nan_to_num((sim1_dict["speed"] - sim2_dict["speed"]), nan=0.0)
    norm = np.linalg.norm(arr_without_nan)*3.6/size
    print("Speed v: {:.2f}".format(norm)) # km/hr 

    arr_without_nan = np.nan_to_num((sim1_dict["occupancy"] - sim2_dict["occupancy"]), nan=0.0)
    norm = np.linalg.norm(arr_without_nan) /size
    print("Occupancy occ: {:.2f}".format(norm)) # convert veh/m to veh/km

    return

def validation_rmse(exp_name):

    with open(exp[exp_name], 'rb') as file:
        macro_sim = pickle.load(file)

    size = macro_gt["flow"].size
    print("Validation RMSE (macro simulation data)")
    # print(macro_gt["speed"] - macro_sim["speed"])
    arr_without_nan = np.nan_to_num((macro_gt["flow"] - macro_sim["flow"])*3600, nan=0.0)
    norm = np.linalg.norm(arr_without_nan)/size
    print("Volume q: {:.2f}".format(norm))  #convert veh/s to veh/hr
          
    arr_without_nan = np.nan_to_num((macro_gt["speed"] - macro_sim["speed"]), nan=0.0)
    norm = np.linalg.norm(arr_without_nan)*3.6/size
    print("Speed v: {:.2f}".format(norm)) # km/hr 

    arr_without_nan = np.nan_to_num((macro_gt["density"] - macro_sim["density"]), nan=0.0)
    norm = np.linalg.norm(arr_without_nan) * 1000/size
    print("Density rho: {:.2f}".format(norm)) # convert veh/m to veh/km

    return


if __name__ == "__main__":

    sumo_dir = r'C:\Users\yanbing.wang\Documents\traffic\sumo\on_ramp'
    measurement_locations = ['upstream_0', 'upstream_1', 
                             'merge_0', 'merge_1', 'merge_2', 
                             'downstream_0', 'downstream_1']
    
    # ground truth detector data
    sim1_dict = reader.extract_sim_meas(measurement_locations=measurement_locations, file_dir=sumo_dir)
        
    with open(exp["gt"], 'rb') as file:
        macro_gt = pickle.load(file)

    training_rmse(None)
    validation_rmse("default")
