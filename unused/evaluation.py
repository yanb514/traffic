import utils
import pickle
import numpy as np
import unused.model_bank as mb
import os
import multiprocessing as mp

def simulate_each_driver_worker(args):
    data_file, pair_id_file, result_path, save_path, idx = args

    cf_data = utils.load_cf_data(data_file, pair_id_file, pair_id_idx=idx)
    if cf_data is not None:
        file_name = f"ovrv_s_tr_{cf_data['foll_id']}_{cf_data['lead_id']}.pkl"
        try:
            with open(os.path.join(result_path, file_name), 'rb') as file:
                res = pickle.load(file)
                cf_data = mb.solve_ivp_cf("ovrv", cf_data, res)
                path_name = os.path.join(save_path, file_name)
                with open(path_name, 'wb') as file:
                    pickle.dump(cf_data, file)
               
        except Exception as e:
            print("** ", str(e))
            pass

    return

def calculate_mse_driver_specific(data_file, pair_id_file, result_path, save_path, training):
    with open(pair_id_file, 'r') as file:
        line_count = sum(1 for _ in file)
    # line_count = 10
    pool = mp.Pool()
    pool.map(simulate_each_driver_worker, [(data_file, pair_id_file, result_path, save_path, idx) for idx in range(line_count)])
    pool.close()
    pool.join()
    print("completed saving all simulation files")
    # List all files in the directory
    files = os.listdir(save_path)

    # Filter out only the pickle files
    pickle_files = [file for file in files if file.endswith(".pkl")]

    # Read each pickle file
    mse_train_values, mse_validation_values, totalN = [],[],0
    for pickle_file in pickle_files:
        file_path = os.path.join(save_path, pickle_file)
        with open(file_path, 'rb') as file:
            cf_data = pickle.load(file)
            N = len(cf_data["timestamps"])
            trainN = int(N*training)
            valiN = N-trainN
            mse_train = np.mean((cf_data["s_gap"][:trainN] - cf_data["s_gap_sim"][:trainN])**2)
            mse_val = np.mean((cf_data["s_gap"][trainN:] - cf_data["s_gap_sim"][trainN:])**2)
            
            mse_train_values.append(mse_train*trainN)
            mse_validation_values.append(mse_val*valiN)

            totalN += N


    print("MSE training: {:.2f}".format(np.sum(mse_train_values) / (totalN * training)))
    print("MSE validation: {:.2f}".format(np.sum(mse_validation_values) / (totalN * (1 - training))))

    return



def simulate_each_driver_fixed_param(args):
    data_file, pair_id_file, save_path, idx, global_theta = args

    cf_data = utils.load_cf_data(data_file, pair_id_file, pair_id_idx=idx)
    if cf_data is not None:
        file_name = f"ovrv_s_tr_{cf_data['foll_id']}_{cf_data['lead_id']}.pkl"
        try:
            cf_data = mb.solve_ivp_cf("ovrv", cf_data, global_theta)
            path_name = os.path.join(save_path, file_name)
            with open(path_name, 'wb') as file:
                pickle.dump(cf_data, file)
               
        except Exception as e:
            print("** ", str(e))
            pass

    return


def calculate_mse_global(data_file, pair_id_file, save_path, training):
    global_ovrv = [9.543e-01,  1.298e+01,  1.404e+01,  2.153e+01]
    with open(pair_id_file, 'r') as file:
        line_count = sum(1 for _ in file)
    # line_count = 10
    pool = mp.Pool()
    pool.map(simulate_each_driver_fixed_param, [(data_file, pair_id_file, save_path, idx, global_ovrv) for idx in range(line_count)])
    pool.close()
    pool.join()
    print("completed saving all simulation files")
    # List all files in the directory

    files = os.listdir(save_path)

    # Filter out only the pickle files
    pickle_files = [file for file in files if file.endswith(".pkl")]

    # Read each pickle file
    mse_train_values, mse_validation_values, totalN = [],[],0
    for pickle_file in pickle_files:
        file_path = os.path.join(save_path, pickle_file)
        with open(file_path, 'rb') as file:
            cf_data = pickle.load(file)
            N = len(cf_data["timestamps"])
            trainN = int(N*training)
            valiN = N-trainN
            mse_train = np.mean((cf_data["s_gap"][:trainN] - cf_data["s_gap_sim"][:trainN])**2)
            mse_val = np.mean((cf_data["s_gap"][trainN:] - cf_data["s_gap_sim"][trainN:])**2)
            
            mse_train_values.append(mse_train*trainN)
            mse_validation_values.append(mse_val*valiN)

            totalN += N


    print("MSE training: {:.2f}".format(np.sum(mse_train_values) / (totalN * training)))
    print("MSE validation: {:.2f}".format(np.sum(mse_validation_values) / (totalN * (1 - training))))


import matplotlib.pyplot as plt
import matplotlib
def plot_optimal_velocity_curves(result_path):
    font = {
        # 'family' : 'times',
        # 'weight' : 'bold',
        'size'   : 16}

    matplotlib.rc('font', **font)

    def V(s, theta):
        return theta[1] * (np.tanh((s-theta[2])/theta[3]) + np.tanh(theta[2]/theta[3]))
    
    files = os.listdir(result_path)
    # Read each pickle file
    
    plt.figure()
    s_array = np.linspace(0, 100, 1000)
    cnt=0
    for file in files:
        cnt+=1
        if cnt > 1000:
            break
        file_path = os.path.join(result_path, file)
        with open(file_path, 'rb') as file:
            res = pickle.load(file)
            if res:
                # print(res.x)
                plt.plot(s_array, V(s_array, res.x), linewidth=0.1, c="grey")
            
    global_ovrv = [9.543e-01,  1.298e+01,  1.404e+01,  2.153e+01]
    plt.plot(s_array, V(s_array, global_ovrv), linewidth=2, c="red", label="global best-fit param")
    plt.ylim([0,70])
    plt.legend()
    plt.xlabel("Space gap (m)")
    plt.ylabel("Optimal velocity (m/s)")
    return

if __name__ == "__main__":
    data_path = "data/"
    file_name = "DATA (NO MOTORCYCLES).txt"
    data_file = os.path.join(data_path, file_name)
    # pair_id_file = os.path.join(data_path, "ngsim_id_pair.txt")
    # result_path = os.path.join("calibration_result","optimize_ovrv_direct_train0.75")

    # calculate_mse(data_file, pair_id_file, result_path, training=0.75)
    
    total_lines = sum(1 for line in open(data_file))
    print("Total number of lines:", total_lines)