import traci
import optuna
import subprocess
import os
import os
import xml.etree.ElementTree as ET
import numpy as np
import sys
import matplotlib.pyplot as plt
import shutil

main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # two levels up
sys.path.insert(0, main_path)
import utils_vis as vis
import utils_data_read as reader




def run_sumo(sim_config, tripinfo_output=None, fcd_output=None):
    """Run a SUMO simulation with the given configuration."""
    # command = ['sumo', '-c', sim_config, '--tripinfo-output', tripinfo_output, '--fcd-output', fcd_output]

    command = ['sumo', '-c', sim_config]
    if tripinfo_output is not None:
        command.extend(['--tripinfo-output', tripinfo_output])
        
    if fcd_output is not None:
        command.extend([ '--fcd-output', fcd_output])
        
    subprocess.run(command, check=True)



def update_sumo_configuration(param):
    """
    Update the SUMO configuration file with the given parameters.
    
    Parameters:
        param (dict): List of parameter values [maxSpeed, minGap, accel, decel, tau]
    """
    
    # Define the path to your rou.xml file
    file_path = SCENARIO+'.rou.xml'

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the vType element with id="trial"
    for vtype in root.findall('vType'):
        if vtype.get('id') == 'trial':
            # Update the attributes with the provided parameters
            for key, val in param.items():
                vtype.set(key, str(val))
            break

    # Write the updated XML content back to the file
    tree.write(file_path, encoding='UTF-8', xml_declaration=True)
    return

def create_temp_config(param, trial_number):
    """
    Update the SUMO configuration file with the given parameters and save it as a new file.
    create new .rou.xml and .sumocfg files for each trial
    
    Parameters:
        param (dict): List of parameter values [maxSpeed, minGap, accel, decel, tau]
        trial_number (int): The trial number to be used for naming the new file.
    """
    
    # Define the path to your original rou.xml and sumocfg files
    original_rou_file_path = SCENARIO + '.rou.xml'
    original_net_file_path = SCENARIO + '.net.xml'
    original_sumocfg_file_path = SCENARIO + '.sumocfg'
    original_add_file_path = 'I24_RDS.add.xml'
    
    # Create the directory for the new files if it doesn't exist
    output_dir = os.path.join('temp', str(trial_number))
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== .Parse the original rou.xml file ==========================
    rou_tree = ET.parse(original_rou_file_path)
    rou_root = rou_tree.getroot()

    # Find the vType element with id="trial"
    for vtype in rou_root.findall('vType'):
        if vtype.get('id') == 'trial':
            # Update the attributes with the provided parameters
            for key, val in param.items():
                vtype.set(key, str(val))
            break

    new_rou_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.rou.xml")
    rou_tree.write(new_rou_file_path, encoding='UTF-8', xml_declaration=True)

    # ==================== copy original net.xml file ==========================
    shutil.copy(original_net_file_path, os.path.join(output_dir, f"{trial_number}_{SCENARIO}.net.xml"))

    # ==================== copy original add.xml file ==========================
    new_add_file_path = os.path.join(output_dir, f"{trial_number}_{original_add_file_path}")
    shutil.copy(original_add_file_path, new_add_file_path)
    
    #  ==================== parse original sumocfg.xml file ==========================
    sumocfg_tree = ET.parse(original_sumocfg_file_path)
    sumocfg_root = sumocfg_tree.getroot()
    input_element = sumocfg_root.find('input')
    if input_element is not None:
        input_element.find('route-files').set('value', f"{trial_number}_{SCENARIO}.rou.xml")
        input_element.find('net-file').set('value', f"{trial_number}_{SCENARIO}.net.xml")
        input_element.find('additional-files').set('value',  f"{trial_number}_{original_add_file_path}")

    new_sumocfg_file_path = os.path.join(output_dir, f"{trial_number}_{SCENARIO}.sumocfg")
    sumocfg_tree.write(new_sumocfg_file_path, encoding='UTF-8', xml_declaration=True)
    
    return new_sumocfg_file_path, output_dir


def objective(trial):
    """Objective function for optimization."""
    # Define the parameters to be optimized
    driver_param = {
        param_name: trial.suggest_uniform(param_name, min_val[i], max_val[i])
        for i, param_name in enumerate(param_names)
    }
    # print(driver_param, trial.number)
    
    # Update SUMO configuration or route files with these parameters
    temp_config_path, temp_path = create_temp_config(driver_param, trial.number)

    # Run SUMO simulation
    run_sumo(temp_config_path)
    
    # Extract simulated traffic volumes
    simulated_output = reader.extract_sim_meas(measurement_locations=measurement_locations, file_dir=temp_path)
    
    # Align time
    # TODO: SIMULATED_OUTPUT starts at 5AM-8AM, while measured_output is 0-24, both in 5min intervals
    start_idx = int(5*60/5)
    end_idx = start_idx + simulated_output["volume"].shape[1]
    
    # Calculate the objective function value
    error = np.linalg.norm(simulated_output["volume"][:,:-1] - measured_output["volume"][:, start_idx: end_idx-1])
    clear_directory(os.path.join("temp", str(trial.number)))
    
    return error

def clear_directory(directory_path):
    """
    Clear all files within the specified directory.
    
    Parameters:
        directory_path (str): The path to the directory to be cleared.
    """
    try:
        shutil.rmtree(directory_path)
        print(f"Directory {directory_path} and all its contents have been removed.")
    except FileNotFoundError:
        print(f"Directory {directory_path} does not exist.")
    except Exception as e:
        print(f"Error removing directory {directory_path}: {e}")


if __name__ == "__main__":

    # Measurement locations
    # ================ I24 scenario ====================
    SCENARIO = "I24_scenario"
   
    # measurement_locations = [
    #                         # '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
    #                          '56_3_0', '56_3_1', '56_3_2', '56_3_3', '56_3_4',
    #                          '56_0_0', '56_0_1', '56_0_2', '56_0_3', '56_0_4',
    #                          '55_3_0', '55_3_1', '55_3_2', '55_3_3',
    #                          '54_6_0', '54_6_1', '54_6_2', '54_6_3',
    #                          '54_1_0', '54_1_1', '54_1_2', '54_1_3' ]
    measurement_locations = [
                            # '56_7_0', '56_7_1', '56_7_2', '56_7_3', '56_7_4', 
                             '56_3_0',  '56_3_4',
                             '56_0_0', '56_0_4',
                             '55_3_0',  '55_3_3'
                             ]
    
    # param_names = ['maxSpeed', 'minGap', 'accel', 'decel', 'tau']
    # min_val = [30.0, 1.0, 1.0, 1.0, 0.5]  
    # max_val = [35.0, 3.0, 4.0, 3.0, 2.0]  
    param_names = ['maxSpeed']
    min_val = [30]  
    max_val = [40] 

    # ================================= get RDS data
    # run_sumo(sim_config=SCENARIO+"_gt.sumocfg", fcd_output ="trajs_gt.xml")
    # vis.visualize_fcd("trajs_gt.xml") # lanes=["E0_0", "E0_1", "E1_0", "E1_1", "E2_0", "E2_1", "E2_2", "E4_0", "E4_1"]
    # rds_dir = r'C:\Users\yanbing.wang\Documents\traffic\data\RDS\I24_WB_52_60_11132023.csv'
    # measured_output = rds_dict = reader.rds_to_matrix(rds_file=rds_dir, det_locations=measurement_locations)

    # # ================================= Create a study object and optimize the objective function
    # clear_directory("temp")
    # sampler = optuna.samplers.TPESampler(seed=10)
    # study = optuna.create_study(direction='minimize', sampler=sampler)
    # study.optimize(objective, n_trials=5000, n_jobs=16)
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()

    # # # Get the best parameters
    # best_params = study.best_params
    # print('Best parameters:', best_params)

    # # ================================ visualize time-space using best parameters
    # best_params =  {'maxSpeed': 31.857824622051734, 'minGap': 2.439592868726828, 'accel': 1.307791098346906, 'decel': 2.217700569222072, 'tau': 1.4450057428115095}
    # update_sumo_configuration(best_params)
    # run_sumo(sim_config=SCENARIO+".sumocfg", fcd_output ="trajs_best.xml")
    # vis.visualize_fcd("trajs_best.xml") # lanes=["E0_0", "E0_1", "E1_0", "E1_1", "E2_0", "E2_1", "E2_2", "E4_0", "E4_1"]

    run_sumo(sim_config=SCENARIO+".sumocfg")

    