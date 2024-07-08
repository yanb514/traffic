import json
import numpy as np
import pymc as pm
from scipy.integrate import solve_ivp,odeint
from scipy.interpolate import interp1d
import pytensor.tensor as pt
from scipy.optimize import differential_evolution, fmin_tnc, fmin_slsqp, fmin_l_bfgs_b, basinhopping, direct
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
# import os
# print(os.getcwd())
# Read the config file
with open(r"C:\Users\yanbing.wang\Documents\traffic\config.json") as f:
    config = json.load(f)
dt = config["ngsim"]["dt"]


# ======================================= CF models ODE ===================================================
def dxdt_cthrv(t, x, theta, fitu):
    return np.array([-x[1] + fitu(t), 
                     theta[0]*x[0] - (theta[0]*theta[2]+theta[1])* x[1] + theta[1]* fitu(t)])

def dxdt_idm(t, x, theta, fitu):
    sn = theta[1] + x[1] * theta[2] + x[1] * (fitu(t)-x[1]) / (2 * np.sqrt(theta[3] * theta[4]))
    a = theta[3] * (1 - (x[1] / theta[0]) ** 4 - (sn / x[0]) ** 2)
    # theta = [VMAX, DSAFE, TSAFE, AMAX, AMIN]
    return np.array([-x[1] + fitu(t), 
                     a]) # use max(a, -10) when using scipy.optimize()

def dxdt_ovrv(t,x, theta, fitu):
    # theta = [ALPHA, A, HM, B]
    ov = theta[1] * (np.tanh((x[0]-theta[2])/theta[3]) + np.tanh(theta[2]/theta[3]))
    return np.array([-x[1] + fitu(t), 
                     theta[0]*(ov-x[1])])

def solve_ivp_cf(model_name, cf_data, theta):
    if isinstance(theta, tuple):
        theta = theta[0]
    elif hasattr(theta, "x"):
        theta = theta.x
        
    model_name = model_name.lower()
    timestamps = cf_data["timestamps"]
    t_span = (timestamps[0], timestamps[-1])
    u = cf_data["lead_v"]
    fitu = interp1d(timestamps, u, kind='linear', fill_value='extrapolate')

    # Define the initial state vector x0
    x0 = np.array([cf_data["s_gap"][0], 
                cf_data["foll_v"][0]])
    if model_name == "cthrv":
        _ode = dxdt_cthrv
    elif model_name == "idm":
        def _ode(t, x, theta, fitu):
            sn = theta[1] + x[1] * theta[2] + x[1] * (fitu(t)-x[1]) / (2 * np.sqrt(theta[3] * theta[4]))
            a = theta[3] * (1 - (x[1] / theta[0]) ** 4 - (sn / x[0]) ** 2)
            # theta = [VMAX, DSAFE, TSAFE, AMAX, AMIN]
            return np.array([-x[1] + fitu(t), 
                            max(a, -10)]) # have to bound accel to simulate
    elif model_name == "ovrv":
        _ode = dxdt_ovrv
        
    sol = solve_ivp(lambda t, x: _ode(t, x, theta, fitu), t_span, x0, t_eval=timestamps)

    if not sol.success:
        raise Exception(f"***** unsuccessful simulation: foll_id {cf_data['foll_id']}. {sol.message} ")
    
    if np.any(sol.y[0] < 0) or np.any(sol.y[1] < 0):
        raise Exception(f"** encounter negative spacing or speed in simulation: foll_id {cf_data['foll_id']}")
    
    cf_data["s_gap_sim"] = sol.y[0]
    cf_data["foll_v_sim"] = sol.y[1]
    
    # integrate velocity to get position
    
    cf_data["foll_p_sim"] = cumtrapz(cf_data["foll_v_sim"], timestamps, initial=0) + cf_data["foll_p"][0]
    return cf_data


# ======================================= For optimization solver ===================================================
def find_theta(model_name, cf_data, training=1, fix=None):
    global_ovrv = [9.543e-01,  1.298e+01,  1.404e+01,  2.153e+01]
    # get training part
    N = len(cf_data["timestamps"])
    N_train = int(training * N)
    if N_train < 10:
        print(f"trajectory too short.")
        return
    
    for key in cf_data.keys():
        if key not in ["foll_id", "lead_id"]:
            cf_data[key] = cf_data[key][:N_train]

    model_name = model_name.lower()
    timestamps = cf_data["timestamps"]
    u = cf_data["lead_v"]
    fitu = interp1d(timestamps, u, kind='linear', fill_value='extrapolate')
    t_span = (timestamps[0], timestamps[-1])

    # Define the initial state vector x0
    x0 = np.array([cf_data["s_gap"][0], 
                cf_data["foll_v"][0]])
    bounds = [(config["model_param"][model_name]["lower"][i], config["model_param"][model_name]["upper"][i]) for i in range(len(config["model_param"][model_name]["upper"]))]
    
    if model_name == "cthrv":
        ode = dxdt_cthrv
        init_guess = [0.1,0.1,1.2]
    elif model_name == "idm":
        bounds = [(10,40), (0.01,10), (0.01,5), (0.01,5), (0.01,5)]
        def ode(t, x, theta, fitu):
            sn = theta[1] + x[1] * theta[2] + x[1] * (fitu(t)-x[1]) / (2 * np.sqrt(theta[3] * theta[4]))
            a = theta[3] * (1 - (x[1] / theta[0]) ** 4 - (sn / x[0]) ** 2)
            # theta = [VMAX, DSAFE, TSAFE, AMAX, AMIN]
            return np.array([-x[1] + fitu(t), 
                            max(a, -10)]) # acceleration has to be bounded, otherwise causes overflow
        init_guess = [33,2,1.6,0.73,1.67]
    elif model_name == "ovrv":
        if fix==2:
            bounds = [bounds[i] for i in [1]]
            def dxdt_ovrv_red(t,x, theta, fitu):
                # theta = [ALPHA, A, HM*, B]
                ov = theta[0] * (np.tanh((x[0]-global_ovrv[2])/global_ovrv[3]) + np.tanh(global_ovrv[2]/global_ovrv[3]))
                return np.array([-x[1] + fitu(t), 
                                global_ovrv[0]*(ov-x[1])])
            ode = dxdt_ovrv_red
        else:
            ode = dxdt_ovrv
        
    def obj_function(theta):
        sol = solve_ivp(lambda t, x: ode(t, x, theta, fitu), t_span, x0, t_eval=timestamps)
        mse = np.mean((sol.y[0] - cf_data["s_gap"])**2)
        # mse = np.mean((sol.y[1] - cf_data["foll_v"])**2)
        return mse 


    # res = differential_evolution(func=obj_function, x0=init_guess, bounds=bounds) # slower then direct, but result is pretty good
    # res = basinhopping(func=obj_function, x0=init_guess)
    res = direct(func=obj_function, bounds=bounds) # fast, result pretty good
    if fix == 2 and model_name == "ovrv":
        # res.x=np.insert(res.x, [2,2], global_ovrv[2:3])
        res.x = np.array([global_ovrv[0], res.x[0] ,global_ovrv[2],global_ovrv[3]])
        print(res.x)
    res.obj_func = obj_function
    res.bounds = bounds
    res.param_names = config["model_param"][model_name]['name']
    # res.ode = ode
    # res.fitu = fitu
    # res.t_span = t_span
    # res.x0 = x0
    # res.t_eval=timestamps
    # res.cf_data = cf_data
    return res


import matplotlib.ticker as ticker
def plot_cross_sec_obj(res, eps=1, n=10, step=3):
    theta_opt = res.x
    row = len(theta_opt)
    bounds = res.bounds
    param_names = res.param_names
    f_opt = res.obj_func(theta_opt)
    theta_center = np.array([(l+u)/2 for (l,u) in bounds])
    print(theta_center)
    # n = 10 # discretization in parameter space
    # step = 3 # skip x/y labels every step numbers
    thetas = np.vstack([np.linspace(bound[0],bound[1],n) for bound in bounds]) # N x 10 grid

    best_idx = []
    for i in range(row):
        diffi = np.absolute(thetas[i]-theta_opt[i])
        idxi = diffi.argmin()
        best_idx.append(idxi)
    
    assert row == len(res.param_names)

    # Plot pairwise cross-sectional obj func values as heatmaps
    fig, axs = plt.subplots(2, 3, figsize=(10,7))
    axs = axs.flatten()
    ax_idx = 0
    theta_subopt = []
    for i in range(row):
        for j in range(row):
            if i > j:
                # compute obj func values in this cross section
                obj_f_2d = np.zeros((n,n))
                subopt_i, subopt_j = [],[] # indices of suboptimal parameters

                for ii in range(n): 
                    for jj in range(n):
                        theta = theta_opt.copy()
                        theta[i] = thetas[i][ii]
                        theta[j] = thetas[j][jj]
                        # print(theta)
                        obj_val = res.obj_func(theta)
                        # print(theta, obj_val)
                        obj_f_2d[ii][jj] = obj_val
                        if obj_val <= f_opt + eps:
                            subopt_i.append(ii)
                            subopt_j.append(jj)
                            theta_subopt.append(theta)
                        

                # az.plot_pair(obj_f_2d, var_names=[param_names[j], param_names[i]], kind='kde', marginals=False,ax=axs[i, j])
                axs[ax_idx].imshow(obj_f_2d)
                axs[ax_idx].scatter(subopt_j, subopt_i, s=15, marker='^', c="red")
                axs[ax_idx].scatter(best_idx[j], best_idx[i], s=20, marker='o', c="green")

                axs[ax_idx].set_xlabel(param_names[j])
                axs[ax_idx].set_ylabel(param_names[i])

                # Set the x and y tick positions
                x_ticks = np.arange(0, n, step)  # Skip every other 5 labels
                y_ticks = np.arange(0, n, step)  # Skip every other 5 labels
                axs[ax_idx].set_xticks(x_ticks, labels=["%.2f" % elem for elem in thetas[j][::step]])
                axs[ax_idx].set_yticks(y_ticks, labels=["%.2f" % elem for elem in thetas[i][::step]])

                ax_idx += 1
           

    # [axs[i, j].axis('off') for i in range(row) for j in range (row) if not axs[i, j].has_data()]
    # [axs[i, j].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}")) for i in range(row) for j in range (row)]
    # [axs[i, j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}")) for i in range(row) for j in range (row)]

    # cbar = plt.colorbar(im, ax=ax_autocorr)
    # cbar.set_label('Autocorrelation')

    # find the theta from theta_subopt that's closest to the center
    theta_subopt = np.array(theta_subopt).T
    distances = np.linalg.norm(theta_subopt - theta_center[:, np.newaxis], axis=0)
    closest_index = np.argmin(distances)
    closest_vector = theta_subopt[:, closest_index]
    print(closest_vector)
                
    plt.tight_layout()
    return closest_vector
# ======================================= CF models for Bayesian calibration ===================================================

def CTHRV_v(K1, K2, TAU, s_gap, foll_v, lead_v): # not for simulation
    a = K1*s_gap - (K1*TAU + K1)*foll_v + K2*lead_v
    return foll_v + a * dt

def CTHRV_s(K1, K2, TAU, s_gap, foll_v, lead_v): # not for simulation
    N = len(s_gap)
    a = K1*s_gap[:N-2] - (K1*TAU + K1)*foll_v[:N-2] + K2*lead_v[:N-2] #[0:N-2]
    v = foll_v[:N-2] + dt * a # [1:N-1]
    s = s_gap[1:N-1] + dt* (lead_v[1:N-1] - v) # [2:N]
    return s

def IDM_v(VMAX, DSAFE, TSAFE, AMAX, AMIN, DELTA, s_gap, foll_v, lead_v):
    sn = DSAFE + foll_v * TSAFE + foll_v * (lead_v-foll_v) / (2 * np.sqrt(AMAX * AMIN))
    a = AMAX * (1 - (foll_v / VMAX) ** DELTA - (sn / s_gap) ** 2)
    return foll_v + a * dt # [1, N]

def IDM_s(VMAX, DSAFE, TSAFE, AMAX, AMIN, DELTA, s_gap, foll_v, lead_v):
    N = len(s_gap)
    sn = DSAFE + foll_v[:N-2] * TSAFE + foll_v[:N-2] * (lead_v[:N-2]-foll_v[:N-2]) / (2 * np.sqrt(AMAX * AMIN)) # [0:N-2]
    a = AMAX * (1 - (foll_v[:N-2] / VMAX) ** DELTA - (sn / s_gap[:N-2]) ** 2) # [0: N-2]
    v = foll_v[:N-2] + dt * a # [1:N-1]
    s = s_gap[1:N-1] + dt * (lead_v[1:N-1] - v) # [2:N+1]
    return s

# ======================================= Bayesian Calibration ==================================================

from pytensor.compile.ops import as_op
def Bayes_ivp(model_name, cf_data):
    model_name = model_name.lower()
    timestamps = cf_data["timestamps"]
    u = cf_data["lead_v"]
    fitu = interp1d(timestamps, u, kind='linear', fill_value='extrapolate')
    t_span = (timestamps[0], timestamps[-1])
    x0 = np.array([cf_data["s_gap"][0], 
                cf_data["foll_v"][0]])
    
    if model_name.lower() == "cthrv":
        _ode = dxdt_cthrv
    elif model_name.lower() == "idm":
        _ode = dxdt_idm
    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def pytensor_forward_model_matrix(theta):
        return solve_ivp(lambda t, x: _ode(t, x, theta, fitu), t_span, x0, t_eval=timestamps, method='RK23', vectorized=True).y[0].flatten()
    
    model = pm.Model()
    # Variable list to give to the sample step parameter
    
    with model:
    
        # define Prior distributions of the CF model parameters
        s_a = pm.Exponential('s_a', lam=1e3) # error of the likelihood
        if model_name.lower() == "cthrv":
            # K1 = pm.Normal("K1", mu=0.1, sigma=0.1)
            K1 = pm.Uniform("K1", lower=0, upper=5)
            K2 = pm.Uniform("K2", lower=0, upper=5)
            TAU = pm.Uniform("TAU", lower=0, upper=5)
            ode_solution = pytensor_forward_model_matrix(
                pm.math.stack([K1, K2, TAU])
            )
           
        elif model_name.lower() == "idm":
            DELTA = 4
            # Define the prior mean for the MvNormal distribution
            prior_mean = pm.floatX(np.array([0, 0, 0, 0, 0]))

            # Define the MvNormal distribution for the normalized parameters
            parameters_normalized = pm.MvNormal('mu_normalized', mu=prior_mean, chol=np.eye(5))

            # Define the deterministic transformation to get the log parameters
            log_parameters = pm.Deterministic('log_mu', parameters_normalized * np.array([.3, 1., 1., .5, .5])
                                            + np.array([2., 0.69, 0.47, -.3, .51]))

            # Define the deterministic transformation to get the parameters
            parameters = pm.Deterministic('mu', np.exp(log_parameters))

            ode_solution = pytensor_forward_model_matrix(parameters)
            
        # sample prior parameters and make forward prediction
        tr = pm.sample_prior_predictive(samples=50)
        
        s_obs = pm.Normal("obs", mu=ode_solution,
                                    sigma = s_a*dt, 
                                    observed = cf_data["s_gap"])
        
        # model intererence init='jitter+adapt_diag_grad',, random_seed=16,  chains=1,cores=8, discard_tuned_samples=True, return_inferencedata=True, target_accept=0.90)
        vars_list = list(model.values_to_rvs.keys())[:-1]
        print(vars_list)
        tr.extend(pm.sample(step=[pm.DEMetropolisZ(vars_list)], draws=5000, tune=5000, chains=1))
        # tr.extend(pm.sample(draws=10000, tune=10000, init='jitter+adapt_diag_grad', random_seed=16,  chains=1,cores=8, discard_tuned_samples=True, return_inferencedata=True, target_accept=0.90))
        
        # sample posterior parameters and make prediction
        pm.sample_posterior_predictive(tr, extend_inferencedata=True)#, var_names=["obs"], predictions=True)
        return tr, model
    


def Bayes_CTHRV(cf_data, obs="s"):
    model = pm.Model()
    with model:
    
        # define Prior distributions of the CF model parameters
        # K1 = pm.Normal("K1", mu=0.1, sigma=0.1)
        K1 = pm.Uniform("K1", lower=0, upper=10)
        K2 = pm.Uniform("K2", lower=0, upper=10)
        TAU = pm.Uniform("TAU", lower=0, upper=10)
        
        if obs == "v":
        # define likelihood distribution
            s_a = pm.Exponential('s_a', lam=1e3) # error of the likelihood
            v_obs = pm.Normal('obs', mu=CTHRV_v(K1, K2, TAU,
                                            cf_data["s_gap"][:-1], 
                                            cf_data["foll_v"][:-1], 
                                            cf_data["lead_v"][:-1]), 
                                            sigma=s_a*dt, 
                                            observed=cf_data["foll_v"][1:])
        elif obs == "s":
            s_a = pm.Exponential('s_a', lam=1e2) # error of the likelihood
            s_obs = pm.Normal("obs", mu=CTHRV_s(K1, K2, TAU, 
                                            s_gap = cf_data["s_gap"],
                                            foll_v = cf_data["foll_v"],
                                            lead_v = cf_data["lead_v"]),
                                            sigma = s_a*dt, 
                                            observed = cf_data["s_gap"][2:])
        else:
            print(f"obs = {obs} not implemented")
            raise NotImplementedError
        
        # sample prior parameters and make forward prediction
        tr = pm.sample_prior_predictive(samples=50)

        # model intererence
        tr.extend(pm.sample(draws=1000, tune=500, random_seed=16, init='jitter+adapt_diag_grad', chains=1,
                        cores=8, discard_tuned_samples=True, return_inferencedata=True, target_accept=0.90))
        
        # sample posterior parameters and make prediction
        pm.sample_posterior_predictive(tr, extend_inferencedata=True)#, var_names=["obs"], predictions=True)
        return tr, model
    


def Bayes_IDM(cf_data, obs="s"):
    model = pm.Model()
    with model:
    
        # define Prior distributions of the CF model parameters
        # K1 = pm.Normal("K1", mu=0.1, sigma=0.1)

        DELTA = 4
        # Define the prior mean for the MvNormal distribution
        # Define the prior mean for the MvNormal distribution
        prior_mean = pm.floatX(np.array([0, 0, 0, 0, 0]))

        # Define the MvNormal distribution for the normalized parameters
        parameters_normalized = pm.MvNormal('mu_normalized', mu=prior_mean, chol=np.eye(5))

        # Define the deterministic transformation to get the log parameters
        log_parameters = pm.Deterministic('log_mu', parameters_normalized * np.array([.3, 1., 1., .5, .5])
                                        + np.array([2., 0.69, 0.47, -.3, .51]))

        # Define the deterministic transformation to get the parameters
        parameters = pm.Deterministic('mu', np.exp(log_parameters))
        
        if obs == "v":
        # define likelihood distribution
            s_a = pm.Exponential('s_a', lam=1e4)
            v_obs = pm.Normal('obs', mu=IDM_v(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], DELTA, 
                                              s_gap = cf_data["s_gap"][:-1], 
                                              foll_v = cf_data["foll_v"][:-1], 
                                              lead_v = cf_data["lead_v"][:-1]), 
                                              sigma = s_a*dt, 
                                              observed = cf_data["foll_v"][1:])
        elif obs == "s":
            s_a = pm.Exponential('s_a', lam=1e2) # error of the likelihood
            s_obs = pm.Normal('obs', mu=IDM_s(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], DELTA, 
                                              s_gap = cf_data["s_gap"], 
                                              foll_v = cf_data["foll_v"], 
                                              lead_v = cf_data["lead_v"]), 
                                              sigma = s_a*dt, 
                                              observed = cf_data["s_gap"][2:])
        else:
            print(f"obs = {obs} not implemented")
            raise NotImplementedError
        
        # sample prior parameters and make forward prediction
        # tr = pm.sample_prior_predictive(samples=50)

        # model intererence
        # tr.extend(pm.sample(draws=1000, tune=500, random_seed=16, init='jitter+adapt_diag_grad', chains=1,
        #                 cores=8, discard_tuned_samples=True, return_inferencedata=True, target_accept=0.90))
        tr = pm.sample(draws=1000, tune=500, random_seed=16, init='jitter+adapt_diag_grad', chains=1,
                        cores=8, discard_tuned_samples=True, return_inferencedata=True, target_accept=0.90)
        # sample posterior parameters and make prediction
        pm.sample_posterior_predictive(tr, extend_inferencedata=True)#, var_names=["obs"], predictions=True)
        return tr, model
    
