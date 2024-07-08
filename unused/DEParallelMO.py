from __future__ import division, print_function, absolute_import
import numpy as np
from scipy._lib._util import check_random_state

# pip from scipy._lib.six import xrange
import multiprocessing
import pickle
from unused.crowdingDistanceSelection import CrowdingDistanceSelection
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import random
import math
import copy
from functools import partial
import unused.model_bank as mb
import json
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os
import utils
TIMEOUT = 300

with open(r"C:\Users\yanbing.wang\Documents\traffic\config.json") as f:
    config = json.load(f)
dt = config["ngsim"]["dt"]

class NoneClass:
    def __init__(self):
        pass

    def close(self):
        pass

    def terminate(self):
        pass


class DEParallelMO(DifferentialEvolutionSolver):
    def __init__(
        self,
        P,
        func,
        # bounds,
        args=(),
        strategy="best1bin",
        maxiter=500,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=None,
        maxfun=np.inf,
        callback=None,
        disp=False,
        polish=False,
        init="latinhypercube",
        atol=0,
        load_file=None,
        model_name="idm",
        cf_data=None,
    ):

        # ========================== CF related =============================
        # model_name = model_name.lower()
        timestamps = cf_data["timestamps"]
        u = cf_data["lead_v"]
        self._fitu = interp1d(timestamps, u, kind='linear', fill_value='extrapolate')
        self._t_span = (timestamps[0], timestamps[-1])
        self._timestamps = timestamps
        self._cf_data = cf_data

        # Define the initial state vector x0
        self._x0 = np.array([cf_data["s_gap"][0], 
                    cf_data["foll_v"][0]])
        # ========================== CF related =============================
            

        self.call_list = []

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        self.polish = polish

        # relative and absolute tolerances for convergence
        self.tol, self.atol = tol, atol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (
            not np.all(np.isfinite(mutation))
            or np.any(np.array(mutation) >= 2)
            or np.any(np.array(mutation) < 0)
        ):
            raise ValueError(
                "The mutation constant must be a float in "
                "U[0, 2), or specified as a tuple(min, max)"
                " where min < max and min, max are in U[0, 2)."
            )

        self.dither = None
        if hasattr(mutation, "__iter__") and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        bounds = [(config["model_param"][model_name]["lower"][i], config["model_param"][model_name]["upper"][i]) for i in range(len(config["model_param"][model_name]["upper"]))]
        self.limits = np.array(bounds, dtype="float").T
        if np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits)):
            raise ValueError(
                "bounds should be a sequence containing "
                "real valued (min, max) pairs for each value"
                " in x"
            )

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)
        self.random_number_generator = check_random_state(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        self.num_population_members = popsize * self.parameter_count
        self.population_shape = (self.num_population_members, self.parameter_count)
        self._nfev = 0

        self.P = P
        self.population_energies = np.zeros((self.num_population_members, self.P))
        self.output_file = None
        self.polish = False

        if load_file is None:
            if init == "latinhypercube":
                self.init_population_lhs()
            elif init == "random":
                self.init_population_random()
            else:
                raise ValueError(
                    "The population initialization method must be one"
                    "of 'latinhypercube' or 'random'"
                )
            self._calculate_population_energies()
        else:
            self.load_from_file(load_file)

        self.disp = disp
        self.strategy = "rand1bin"
        self.it = 0
        self._mapwrapper = NoneClass()
        self._wrapped_constraints = None
        self.population_history = []
        self.energies_history = []

        

    def set_output_file(self, fname):
        self.output_file = fname

    def load_from_file(self, fname):
        f = open(fname, "rb")
        data = pickle.load(f)
        f.close()
        self.population_energies = data["population_energies"]
        self.population = data["population"]

    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """

        param_list = []
        trial_list = []

        for index, candidate in enumerate(self.population):
            if self._nfev > self.maxfun:
                break
            trial_list.append(candidate)

            parameters = self._scale_parameters(candidate)

            param_list.append(parameters)

        p = multiprocessing.Pool(multiprocessing.cpu_count())
        # ode, u, t_span, x0, timestamps, s_data, v_data
        result = p.map(partial(self.func, 
                               u=self._fitu, t_span=self._t_span, x0=self._x0, timestamps=self._timestamps,
                               s_data=self._cf_data["s_gap"], v_data=self._cf_data["foll_v"]), 
                               param_list)
        p.close()

        for i, cd in enumerate(self.population):
            self.call_list.append((copy.copy(cd), copy.copy(result[i])))

        self.population_energies = np.zeros((self.num_population_members, self.P))

        for index, candidate in enumerate(self.population):
            self.population_energies[index, :] = result[index]
            self._nfev += 1

        if self.output_file is not None:
            self.dump_data()
            print("update")
        print("iteration finished")

    def dump_data(self):
        dic = {
            "population_energies": self.population_energies,
            "population": self.population,
            "original_population": self.__scale_arg1 + (self.population - 0.5) * self.__scale_arg2 # scale back
        }
        f = open(self.output_file, "wb")
        pickle.dump(dic, f)
        f.close()

    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        self.it = self.it + 1

        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        if self.dither is not None:
            self.scale = (
                self.random_number_generator.rand() * (self.dither[1] - self.dither[0])
                + self.dither[0]
            )

        param_list = []
        trials = []

        p = multiprocessing.Pool(multiprocessing.cpu_count())

        for candidate in range(self.num_population_members):
            if self._nfev > self.maxfun:
                raise StopIteration

            # create a trial solution
            trial = self._mutate(candidate)

            # ensuring that it's in the range [0, 1)
            self._ensure_constraint(trial)

            # scale from [0, 1) to the actual parameter value
            parameters = self._scale_parameters(trial)
            trials.append(trial)

            param_list.append(parameters)

        # result = p.map(self.func, param_list)
        result = p.map(partial(self.func, 
                        u=self._fitu, t_span=self._t_span, x0=self._x0, timestamps=self._timestamps,
                        s_data=self._cf_data["s_gap"], v_data=self._cf_data["foll_v"]), 
                        param_list)

        p.close()

        N = self.population_energies.shape[0]
        trial_population = np.zeros((2 * N, self.population.shape[1]))
        trial_energy = np.zeros((2 * N, len(result[0])))

        trial_population[0:N, :] = self.population[0:N, :]

        trial_energy[0:N, :] = self.population_energies[0:N, :]

        for i in range(len(result)):
            trial_population[N + i, :] = copy.copy(trials[i])
            trial_energy[N + i, :] = copy.copy(result[i])

            self.call_list.append((copy.copy(trials[i]), copy.copy(result[i])))

        distance_selection = CrowdingDistanceSelection(trial_energy)

        rankings = distance_selection.get_rankings()

        indices = []
        current_rank = 0

        while True:

            if len(indices) + len(rankings[current_rank]) <= N:
                indices.extend(rankings[current_rank])
            else:
                distances, ordered = distance_selection.get_crowding_distance(
                    rankings[current_rank]
                )
                to_get = N - len(indices)
                indices.extend(ordered[0:to_get])

            current_rank += 1

            if len(indices) == N:
                break

        random.shuffle(indices)

        for j in range(len(indices)):
            pop_ind = indices[j]

            self.population[j, :] = copy.copy(trial_population[pop_ind, :])
            self.population_energies[j, :] = copy.copy(trial_energy[pop_ind, :])

            # print("j", j, list(self.population[j,:]),self.population_energies[j,:] )
            # print("pop_ind", pop_ind, list(trial_population[pop_ind, :]), trial_energy[pop_ind, :])

        self.population_history.append(copy.copy(self.population))

        self.energies_history.append(copy.copy(self.population_energies))

        if self.output_file is not None:
            self.dump_data()

        return self.x, self.population_energies[0]

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def dump_result(self):
        f = open("out.txt", "a")

        f.write("---------------------- ---\r\n")
        f.write(
            "solution " + str(self.x) + "--" + str(self.population_energies[0]) + "\r\n"
        )

    def dump_data(self):
        dic = {
            "population_energies": self.population_energies,
            "population": self.population,
            "pop_history": self.population_history,
            "ener_history": self.energies_history,
        }
        f = open(self.output_file, "wb")
        pickle.dump(dic, f)
        f.close()

    def check_calls(self):

        for j in range(self.num_population_members):
            member = self.population[j, :]

            for trial, result in self.call_list:
                # print("member is", member)
                # print("trial is", trial)
                abs_error = 0
                for k in range(len(member)):
                    abs_error += abs(trial[k] - member[k])

                if abs_error < 1e-7:
                    print("member is", member)
                    print("trial is", trial)
                    print("recored", result)
                    break

            true_value = self._scale_parameters(trial)
            value = self.func(true_value)

            print(
                "recored",
                result,
                "computed",
                value,
                "energies",
                self.population_energies[j, :],
            )



        
def obj_function_idm(theta, u, t_span, x0, timestamps, s_data, v_data):
    def _ode(t, x, theta, fitu):
        sn = theta[1] + x[1] * theta[2] + x[1] * (fitu(t)-x[1]) / (2 * np.sqrt(theta[3] * theta[4]))
        a = theta[3] * (1 - (x[1] / theta[0]) ** 4 - (sn / x[0]) ** 2)
        # theta = [VMAX, DSAFE, TSAFE, AMAX, AMIN]
        return np.array([-x[1] + fitu(t), 
                        max(a, -10)]) # acceleration has to be bounded, otherwise causes overflow

    sol = solve_ivp(lambda t, x: _ode(t, x, theta, u), t_span, x0, t_eval=timestamps)
    mses = np.mean((sol.y[0] - s_data)**2)
    msev = np.mean((sol.y[1] - v_data)**2)
    # print("returning", mses, msev)
    return mses, msev
    
def obj_function_ovrv(theta, u, t_span, x0, timestamps, s_data, v_data):
    sol = solve_ivp(lambda t, x: mb.dxdt_ovrv(t, x, theta, u), t_span, x0, t_eval=timestamps)
    mses = np.mean((sol.y[0] - s_data)**2)
    msev = np.mean((sol.y[1] - v_data)**2)
    # maes = np.mean(np.abs(sol.y[0] - s_data))
    # speed_var = np.var()
    return mses, msev



if __name__ == "__main__":
    data_path = "data/"
    file_name = "DATA (NO MOTORCYCLES).txt"
    data_file = os.path.join(data_path, file_name)
    pair_id_file = os.path.join(data_path, "ngsim_id_pair.txt")

    cf_data = utils.load_cf_data(data_file, pair_id_file, pair_id_idx=20)
    for key in cf_data:
        cf_data[key] = np.array(cf_data[key])

    MODEL_NAME = "ovrv"
    obj_func_dict = {
        "idm": obj_function_idm,
        "ovrv": obj_function_ovrv
    }
    solver_parallel = DEParallelMO(
        2, obj_func_dict[MODEL_NAME], popsize=10, maxiter=20, load_file=None, model_name=MODEL_NAME,
        cf_data=cf_data,
    )
    output_file = f"calibration_result/DEMO_{MODEL_NAME}/{MODEL_NAME}_{cf_data['foll_id']}_{cf_data['lead_id']}.pkl"
    solver_parallel.set_output_file(output_file)
    solver_parallel.solve()
