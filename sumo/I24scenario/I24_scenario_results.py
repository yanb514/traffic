"""
This file generates the training and validation results from macrosopic data stored in .pkl
"""
import pickle
import numpy as np
import os
import os
import numpy as np
import sys
import joblib

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import utils_data_read as reader
import i24_calibrate_de as i24


SCENARIO = "I24_scenario"
computer_name = os.environ.get('COMPUTERNAME', 'Unknown')
if "CSI" in computer_name:
    SUMO_DIR = r'C:\Users\yanbing.wang\Documents\traffic\sumo\I24scenario'
    RDS_DIR = r'C:\Users\yanbing.wang\Documents\traffic\data\RDS\I24_WB_52_60_11132023.csv'
    sumo_exe = 'sumo'
elif "VMS" in computer_name:
    SUMO_DIR = r'C:\Users\svcpsat\Documents\SUMO_studies\traffic\sumo\I24scenario'
    RDS_DIR = r'C:\Users\svcpsat\Documents\SUMO_studies\traffic\data\RDS\I24_WB_52_60_11132023.csv'
    sumo_exe = r'C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe'


measurement_locations = [
                        # '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                         '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4',
                         '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
                         '55_3_0', '55_3_1', '55_3_2', '55_3_3',
                         '54_6_0', '54_6_1', '54_6_2', '54_6_3',
                         '54_1_0', '54_1_1', '54_1_2', '54_1_3' ]


default_params =  {'maxSpeed': 34.91628705652602,
                    'minGap': 2.9288888706657783,
                    'accel': 1.0031145478483796,
                    'decel': 2.9618821510422406,
                    'tau': 1.3051261247487569,
                    'lcStrategic': 1.414,
                    'lcCooperative': 1.0,
                    'lcAssertive': 1.0,
                    'lcSpeedGain': 3.76,
                    'lcKeepRight': 0.0,
                    'lcOvertakeRight': 0.877}
i24.update_sumo_configuration(default_params)



def detector_rmse(exp_name):

    '''
    sumo_dir: directory for DETECTOR.out.xml files
    measurement_locations: a list of detectors
    quantity: "volume", "speed" or "occupancy"
    '''

    # Read and extract data
    print("Training RMSE (detectors)")
    sim2_dict = reader.extract_sim_meas(measurement_locations=["trial_" + location for location in measurement_locations], file_dir=SUMO_DIR)

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


def macro_rmse(exp_name):
    '''
    Compare simulated macro with RDS AMS in the selected temporal-spatial range
    '''

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

def read_opt_result(result_pkl):
    with open(result_pkl, 'rb') as file:
        result = pickle.load(file)
    print("Optimization result:", result)
    print("Best parameters found:", result.best_params)
    print("Objective function value at best parameters:", result.best_value)
    return

def read_de_result(result_pkl):
    # with open(result_pkl, 'rb') as file:
    result = joblib.load(result_pkl)
    print("Optimization result:", result)
    print("Best parameters found:", result.x)
    print("Objective function value at best parameters:", result.fun)

    return result.x


if __name__ == "__main__":

    
    EXP = "2b"

    if "1" in EXP:
        param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
        min_val = [25.0, 0.5, 1.0, 1.0, 0.5]  
        max_val = [40.0, 3.0, 4.0, 4.0, 2.0] 
    elif "2" in EXP:
        param_names = ['lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain']
        min_val = [0, 0, 0.0001, 0]  
        max_val = [5, 1, 5,      5] 
    elif "3" in EXP:
        param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau', 'lcStrategic', 'lcCooperative', 'lcAssertive', 'lcSpeedGain']
        min_val = [25.0, 0.5, 1.0, 1.0, 0.5, 0, 0, 0.0001, 0]  
        max_val = [40.0, 3.0, 4.0, 4.0, 2.0, 5, 1, 5,      5] 
    if "a" in EXP:
        MEAS = "volume"
    elif "b" in EXP:
        MEAS = "speed"
    elif "c" in EXP:
        MEAS = "occupancy"
    
    # ================== rerun with new parameters
    # i24.run_sumo()

    # # ================================ visualize time-space using best parameters
    result_pkl = rf'C:\Users\yanbing.wang\Documents\traffic\sumo\I24scenario\calibration_result\study_{EXP}.pkl'
    # result_pkl = os.path.join(SUMO_DIR, f'calibration_result\\study_{EXP}.pkl')
    print(result_pkl)
    best_params = read_opt_result(result_pkl)
    # i24.update_sumo_configuration(best_params)
    # i24.run_sumo(sim_config=SCENARIO+".sumocfg", fcd_output ="trajs_best.xml")

    # ============================== error at detectors
    vis.plot_rds_vs_sim(RDS_DIR, SUMO_DIR, measurement_locations, quantity="speed")
    detector_rmse(exp_name=EXP)
    macro_rmse(exp_name=EXP)


    # ============== compute & save macroscopic properties ==================
    # i24.update_sumo_configuration(best_params)
    # base_name = SCENARIO+""
    # fcd_name = "fcd_"+base_name+"_"+EXP
    # i24.run_sumo(sim_config = base_name+".sumocfg", fcd_output =fcd_name+".out.xml")
    # reader.fcd_to_csv_byid(xml_file=fcd_name+".out.xml", csv_file=fcd_name+".csv")
    # macro.reorder_by_id(fcd_name+".csv", bylane="mainline")
    # macro_data = macro.compute_macro(fcd_name+"_mainline.csv", dx=160.934, dt=30, save=True, plot=True)

    # with open('macro_fcd_I24_scenario_1b_byid.pkl', 'rb') as file:
    #     macro_data = pickle.load(file)
    # macro.plot_macro(macro_data, dx=160.934, dt=30)


    
    # asm_file = r"C:\Users\yanbing.wang\Documents\traffic\data\2023-11-13-ASM.csv"
    # vis.read_asm(asm_file)
    # vis.scatter_fcd_i24(fcd_name+".out.xml")
    # ground truth detector data
    # sim1_dict = reader.extract_sim_meas(measurement_locations=measurement_locations, file_dir=sumo_dir)
        
    # with open(exp["gt"], 'rb') as file:
    #     macro_gt = pickle.load(file)

    # training_rmse(None)
    # validation_rmse("default")

    