import utils
import os
import numpy as np
import unused.model_bank as mb
import pickle
import json
import arviz as az
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from multiprocessing import Pool
import matplotlib
# import aesara.tensor as tt

import warnings
warnings.filterwarnings("ignore")


def calibrate_single_cf(model_name, method, cf_data, config, training=0.75, fix=None, save_path=False):
    print(f"Calibrating {model_name} for foll: {cf_data['foll_id']} lead: {cf_data['lead_id']} using {method}...")
    model_name = model_name.lower()
    if method.lower() == "bayesian":
        result_path = os.path.join(config["result_path"],"bayesian_"+model_name)
        if model_name == "cthrv":
            tr, _ = mb.Bayes_CTHRV(cf_data=cf_data)
        elif model_name == "idm":
            tr, _ = mb.Bayes_IDM(cf_data=cf_data)


    elif method.lower() == "optimization":
        if fix:
            # add="_"+"_".join([config["model_param"][model_name]["name"][i] for i in range(len(config["model_param"][model_name]["name"])) if i == fix])
            add = "_HM_B_ALPHA"
        else: add=""
        result_path = os.path.join(config["result_path"], f"optimize_{model_name}_direct_train{training}{add}")
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print(f"Directory '{result_path}' created.")
        try:
            tr = mb.find_theta(model_name, cf_data, training=training, fix=fix) # output optimization result
        except Exception as e:
            raise Exception(f"** Error in optimizaiton method find_theta, message: {str(e)}")
    else:
        print(f"{model_name} not implemented.")
        raise NotImplementedError

    if save_path:     
        file_name = f"{model_name}_s_tr_{cf_data['foll_id']}_{cf_data['lead_id']}.pkl" 
        data_path = os.path.join(result_path, file_name)
        with open(data_path, "wb") as f:
            try:
                pickle.dump(tr, f)
            except AttributeError:
                delattr(tr, "obj_func")
                pickle.dump(tr, f)
            print(f"{file_name} saved to {result_path}")
    return

def calibrate_population_parallel(args):
    model_name, method, config, training, fix, save_path, idx = args
 
    data_path = config["ngsim"]["data_path"]
    file_name = config["ngsim"]["file_name"]

    data_file = os.path.join(data_path, file_name)
    pair_id_file = os.path.join(data_path, "ngsim_id_pair.txt")
    
    cf_data = utils.load_cf_data(data_file, pair_id_file, pair_id_idx=idx)
    if cf_data is not None:
        try:
            calibrate_single_cf(model_name=model_name, method=method, cf_data=cf_data, config=config, training=training, fix=fix, save_path=save_path)
        except:
            pass

def calibrate_population(model_name, method, config, training=0.75, fix=None, save_path=False):
    data_path = config["ngsim"]["data_path"]
    file_name = config["ngsim"]["file_name"]

    data_file = os.path.join(data_path, file_name)
    pair_id_file = os.path.join(data_path, "ngsim_id_pair.txt")
    
    #TODO Super inefficient way to get all the cf data
    with open(pair_id_file, 'r') as file:
        line_count = sum(1 for _ in file)
        
    pool = Pool()
    pool.map(calibrate_population_parallel, [(model_name, method, config, training, fix, save_path, idx) for idx in range(line_count)])
    pool.close()
    pool.join()


    return


def get_empirical_pdf(model_name, method, config, result_path):

    file_name = config["ngsim"]["file_name"]
    method = method.lower()
    model_name = model_name.lower()
    mu = []
    all_files = os.listdir(result_path)
    pickle_files = [file for file in all_files if file.endswith('.pkl') and file.startswith(f"{model_name}_s")]
    
    for i, file_name in enumerate(pickle_files):
        with open(os.path.join(result_path, file_name), 'rb') as file:
            data = pickle.load(file)
            # if i == 0:
            # print(data)
            if isinstance(data, tuple):
                val = data[0].reshape(-1,1)
            elif hasattr(data, "x"):
                val = data.x.reshape(-1,1)
                # print(val)
            else: # data is of 
                # get mean values of each driver
                # summary = az.summary(data)
                # mean_val = summary.loc[["mu[0]", "mu[1]", "mu[2]", "mu[3]", "mu[4]"], ["mean"]].values
                if not data:
                    continue
                # get MAP value of each driver parameter
                posterior_samples = data.posterior["mu"][0,:,:].values
                map_index = np.argmax(posterior_samples, axis=0)
                val = posterior_samples[map_index,:].diagonal().reshape(-1,1)

            mu.append(val)

    mu = np.concatenate(mu, axis=1) # size (n_theta, N_population)
    return mu

from scipy.stats import multivariate_normal
def draw_from_empirical_pdf(mu, sample_num = 100, dist_name=None):
    if dist_name is None:
        # simply draw random samples from mu -> closest to empirical pdf
        random_indices = np.random.choice(mu.shape[1], sample_num)
        random_samples = mu[:, random_indices]

    elif dist_name == "gaussian":
        mean = np.mean(mu, axis=1)
        covariance_matrix = np.cov(mu)
        dist = multivariate_normal(mean=mean, cov=covariance_matrix)
        random_samples = dist.rvs(size=sample_num)

    else:
        raise NotImplementedError
    return random_samples

def plot_empirical_pdf(model_name, mu_array, config):
    row, col = mu_array.shape
    param_names = config["model_param"][model_name]['name']
    upper = config["model_param"][model_name]["upper"]
    lower = config["model_param"][model_name]["lower"]

    assert row == len(param_names)

    datadict = {param_names[i]: mu_array[i] for i in range(row)}
    idata = az.convert_to_inference_data(datadict)

    # Get the variable names

    # Plot pairwise joint distributions as heatmaps
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    
    for i in range(row):
        for j in range(row):
            if i > j:
                # az.plot_kde(idata.posterior[param_names[j]], idata.posterior[param_names[i]], ax=axs[i, j], contour=True)
                try:
                    az.plot_pair(idata.posterior, var_names=[param_names[j], param_names[i]], kind='kde', marginals=False,ax=axs[i, j])
                    axs[i, j].set_xlabel(param_names[j])
                    axs[i, j].set_ylabel(param_names[i])
                    # axs[i, j].set_xlim([lower[j], upper[j]])
                    # axs[i, j].set_ylim([lower[i], upper[i]])
                except: pass
            if i == j:
                # az.plot_posterior(idata.posterior[param_names[j]],ax=axs[i, j])
                axs[i, j].hist(idata.posterior[param_names[j]], bins=30)


    [axs[i, j].axis('off') for i in range(5) for j in range (5) if not axs[i, j].has_data()]
    # # Plot the correlation matrix
    autocorr_matrix = np.corrcoef(mu_array)
    # Specify the position for the autocorrelation plot
    left, bottom, width, height = 0.65, 0.65, 0.35, 0.35
    ax_autocorr = fig.add_axes([left, bottom, width, height])
    im = ax_autocorr.imshow(autocorr_matrix, cmap='binary', interpolation='nearest')
    ax_autocorr.set_title('Autocorrelation Matrix')
    ax_autocorr.set_xticks(np.arange(len(param_names)))
    ax_autocorr.set_yticks(np.arange(len(param_names)))
    ax_autocorr.set_xticklabels(param_names)
    ax_autocorr.set_yticklabels(param_names)

    cbar = plt.colorbar(im, ax=ax_autocorr)
    cbar.set_label('Autocorrelation')
    plt.tight_layout()
    return


def plot_empirical_pdf_flat(model_name, mu_array, config, plot_idx):
    font = {
        # 'family' : 'times',
        # 'weight' : 'bold',
        'size'   : 20}

    matplotlib.rc('font', **font)
    row, col = mu_array.shape
    param_names = config["model_param"][model_name]['name']
    upper = config["model_param"][model_name]["upper"]
    lower = config["model_param"][model_name]["lower"]

    assert row == len(param_names)

    # Plot pairwise joint distributions as heatmaps
    fig, axs = plt.subplots(1, len(plot_idx), figsize=(len(plot_idx)*5,5))
    if len(plot_idx)==1:
        idx = plot_idx[0]
        axs.hist(mu_array[idx], bins=30)
        axs.set_xlabel(param_names[idx])
        axs.set_ylabel("Count")
        # return
    else:
        idx = 0
        for i in plot_idx:
            axs[idx].hist(mu_array[i], bins=30)
            axs[idx].set_xlabel(param_names[i])
            axs[idx].set_ylabel("Count")
            idx += 1

    plt.tight_layout()
    return

  

if __name__ == "__main__":

    with open(r"C:\Users\yanbing.wang\Documents\traffic\micro-calibration\config.json") as f:
        config = json.load(f)

    # calibrate_population("idm", "s", config, save_path=True)
    mu = get_empirical_pdf("idm", "s", config, save_path=False)
