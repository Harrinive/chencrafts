from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (
    minimize,
    differential_evolution,
    shgo,
    brute,
    basinhopping,
    dual_annealing,
    NonlinearConstraint,
)
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Union

from tqdm.notebook import tqdm

from chencrafts.bsqubits.error_rates import manual_constr
from chencrafts.toolbox.save import path_decorator, save_variable_list_dict, load_variable_list_dict
from chencrafts.toolbox.plot import IntCmap, filter

# ##############################################################################
TARGET_NORMALIZE = 1e-7


def nan_2_flat_val(full_variables, possible_nan_value):
    """
    The full_variables should contain "n_bar" and "kappa_s"
    """
    if np.isnan(possible_nan_value):
        return full_variables["n_bar"] * full_variables["kappa_s"]
    else:
        return possible_nan_value


def nan_2_constr(full_variables, possible_nan_value):
    """
    The full_variables should contain "n_bar", "kappa_s", "g_sa", 
    "min_detuning", "detuning_lower_bound", "constr_amp"
    """
    if np.isnan(possible_nan_value):
        base_val = full_variables["n_bar"] * full_variables["kappa_s"]
        val = base_val + manual_constr(**full_variables)
        return val
    else:
        return possible_nan_value

# ##############################################################################
class OptTraj():
    def __init__(
        self,
        para_name: List[str],
        para_traj: np.ndarray,
        target_traj: np.ndarray,
        constr_traj: np.ndarray,
    ):
        self.para_name = para_name
        self.para_traj = para_traj
        self.target_traj = target_traj
        self.constr_traj = constr_traj

        self.length = self.para_traj.shape[0]

    @classmethod
    def from_file(cls, file_name):
        traj_dict = load_variable_list_dict(file_name, throw_nan=False)

        para_name = [name for name in traj_dict.keys() if name not in [
            "target", "constr"]]

        para_shape = [len(traj_dict[para_name[0]]), len(para_name)]
        para_traj = np.zeros(para_shape)
        for idx, name in enumerate(para_name):
            para_traj[:, idx] = traj_dict[name]

        instance = cls(
            para_name,
            para_traj,
            traj_dict["target"],
            traj_dict["constr"],
        )

        return instance

    def __getitem__(self, name):
        idx = self.para_name.index(name)
        return self.para_traj[:, idx]

    def _x_arr_2_dict(self, x: np.ndarray | List):
        return OrderedDict(zip(self.para_name, x))

    def _x_dict_2_arr(self, x: dict):
        return [x[name] for name in self.para_name]

    @property
    def final_para(self):
        return self._x_arr_2_dict(self.para_traj[-1, :])

    @property
    def init_para(self):
        return self._x_arr_2_dict(self.para_traj[0, :])

    @property
    def final_target(self):
        return self.target_traj[-1]

    @property
    def init_target(self):
        return self.target_traj[0]

    def copy(self):
        new_result = OptTraj(
            self.para_traj.copy(),
            self.target_traj.copy(),
            self.constr_traj.copy(),
        )
        return new_result

    def append(self, para_dict: dict, target: float, constr: float):
        para_arr = self._x_dict_2_arr(para_dict)
        self.para_traj = np.append(self.para_traj, [para_arr], axis=0)
        self.target_traj = np.append(self.target_traj, target)
        self.constr_traj = np.append(self.constr_traj, constr)
        self.length += 1

    def to_dict(self) -> OrderedDict:
        traj_dict = OrderedDict({})
        for idx, key in enumerate(self.para_name):
            traj_dict[key] = self.para_traj[:, idx]
        traj_dict["target"] = self.target_traj
        traj_dict["constr"] = self.constr_traj
        return traj_dict

    def _normalize_para(self, para_range_dict: dict = {}):
        new_var = self.para_traj.copy()

        for var, (low, high) in para_range_dict.items():
            idx = self.para_name.index(var)
            new_var[:, idx] = (new_var[:, idx] - low) / (high - low)

        return new_var

    def plot(self, para_range_dict: dict = {}):
        # need further updating: use twin y axis for the target_traj
        normalized_para = self._normalize_para(para_range_dict)
        max_target = np.max(self.target_traj)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=300)

        ax.plot(range(self.length), normalized_para, label=self.para_name)
        ax.plot(range(self.length), self.target_traj /
                 max_target, label="normed_target")
        ax.plot(range(self.length), self.constr_traj /
                 max_target, label="normed_constr")
        ax.legend()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Normalized Parameters")

        if ax is None:
            plt.show()

    def plot_2d(
        self, 
        ax, 
        x_name,
        y_name,
        c: str = "white",
        destination_only: bool = True, 
        background_interp: Callable = None,
    ):
        x = self[x_name]
        y = self[y_name]

        if not destination_only:
            ax.plot(x, y, c=c, alpha=0.3)
        ax.scatter(x[-1], y[-1], c=c, s=8)

        if background_interp is not None:
            val = background_interp(x[-1], y[-1])
            if np.abs(val) >= 1e-2 and np.abs(val) < 1e2: 
                text = f"  {val:.3f}"
            else:
                text = f"  {val:.1e}"
            ax.text(x[-1], y[-1], text, ha="left", va="center", c=c, fontsize=7)

    def save(self, file_name):
        save_variable_list_dict(file_name, self.to_dict())


class MultiTraj():
    def __init__(
        self,
    ):
        self.traj_list: List[OptTraj] = []
        self.length = 0

    @classmethod
    def from_list(
        cls,
        traj_list: List[OptTraj],
    ):
        new_list = cls()
        for traj in traj_list:
            new_list.append(traj)
        return new_list

    @classmethod
    def from_folder(
        cls,
        path,
    ):
        multi_traj = cls()

        path = path_decorator(path)

        idx = 0
        while True:
            try:
                traj = OptTraj.from_file(f"{path}{idx}.csv")
                multi_traj.append(traj)
                idx += 1
            except FileNotFoundError:
                return multi_traj

    def __getitem__(
        self,
        idx,
    ) -> OptTraj | MultiTraj:
        if isinstance(idx, int):
            return self.traj_list[idx]
        elif isinstance(idx, slice):
            return MultiTraj.from_list(self.traj_list[idx])
        else:
            raise TypeError(f"Only accept int and slice as index")

    def _target_list(self):
        target_list = []
        for traj in self.traj_list:
            target_list.append(traj.final_target)

        return target_list

    def append(
        self,
        traj: OptTraj,
    ):
        self.traj_list.append(traj)
        self.length += 1

    def save(
        self,
        path,
    ):
        path = path_decorator(path)
        for idx in range(self.length):
            self[idx].save(f"{path}{idx}.csv")

    def best_traj(self, select_num=1) -> OptTraj | MultiTraj:
        sort = np.argsort(self._target_list())
        new_traj = MultiTraj()
        for sorted_idx in range(select_num):
            idx = int(sort[sorted_idx])
            new_traj.append(self[idx])

        if select_num == 1:
            return new_traj[0]
        else:
            return new_traj

    def plot_target(self, ax=None, ylim=(1e-7, 6e-6)):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=300)

        best = self.best_traj()
        cmap = IntCmap(self.length)
        for idx, traj in enumerate(self.traj_list):
            if traj == best:
                filter_name = "emph"
            else:
                filter_name = "trans"

            ax.plot(
                range(traj.length),
                traj.target_traj,
                label=f"traj {idx}",
                color=filter(cmap(idx), filter_name),
                zorder=-1
            )
            ax.scatter(
                [traj.length - 1],
                [traj.target_traj[-1]],
                color=filter(cmap(idx), filter_name),
                zorder=-1
            )

        ax.set_ylim(*ylim)
        # ax.set_title("error rates")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Total Error Rate / GHz")
        # ax.set_legend()
        ax.grid()

        if ax is None:
            # plt.savefig("./figures/C2QA slides/error rates w iteration small.png")
            plt.tight_layout()
            plt.show()

# ##############################################################################
class Optimization():
    def __init__(
        self,
        fixed_variables: OrderedDict[str, float],
        free_variable_ranges: OrderedDict[str, List[float]],
        target_func: Callable,
        target_kwargs: dict = {},
        optimizer: str = "L-BFGS-B",
    ):
        """
        The target function should be like: 
        target_func(full_variable_dict, **kwargs)  
        Supported optimizers: L-BFGS-B, Nelder-Mead, Powell, shgo, differential evolution
        """
        self.fixed_variables = fixed_variables
        self.free_variables = free_variable_ranges
        self._update_free_name_list()

        # the value stored in self.default_variables is dynamic - it is always
        # the same as the one stored in self.fixed_variables
        self.default_variables = fixed_variables.copy()
        for key, (low, high) in self.free_variables.items():
            self.default_variables[key] = (low + high) / 2

        self.target_func = target_func
        self.target_kwargs = target_kwargs

        self.optimizer = optimizer
        assert self.optimizer in ["L-BFGS-B", "Nelder-Mead", "Powell",
                                  "shgo", "differential evolution"]

    def _update_free_name_list(self):
        self.free_name_list = list(self.free_variables.keys())
        print(f"Current order of input: {self.free_name_list}")

    def _check_exist(
        self,
        variable: str,
    ):
        if variable not in self.default_variables.keys():
            raise KeyError(f"{variable} is not in the default variable dict. "
                           "Please consider re-initializing an optimize object including this variable.")

    def _fix(
        self,
        variable: str,
        value: float = None,
    ):
        self._check_exist(variable)

        if value is not None:
            if value != self.default_variables[variable]:
                self.default_variables[variable] = value
                # print(f"Fixing the value of {variable} different from the "
                # "default value leads to the changing of default value.")
        else:
            value = self.default_variables[variable]

        if variable in self.fixed_variables():
            self.fixed_variables[variable] = value
        elif variable in self.free_variables():
            self.fixed_variables[variable] = value
            del self.free_variables[variable]

    def fix(
        self,
        variables=None,
        **kwargs,
    ):
        """
        The method accpets:
        Optimize.fix("var"), 
        Optimize.fix(["var_1", "var_2"]), 
        Optimize.fix({"var_1": 1, "var_2": 2}), 
        Optimize.fix(var_1 = 1, var_2 = 2)
        """
        if variables is None:
            variables = kwargs

        if isinstance(variables, str):
            self._fix(variables)
        elif isinstance(variables, list):
            for var in variables:
                self._fix(var)
        elif isinstance(variables, dict):
            for key, val in variables.items():
                self._fix(key, val)
        else:
            raise ValueError(f"Only accept str, list, dict as the input.")

        self._update_free_name_list()

    def _free(
        self,
        variable: str,
        range: List[float],
    ):
        self._check_exist(variable)

        if variable in self.free_variables():
            self.free_variables[variable] = range
        elif variable in self.fixed_variables():
            self.free_variables[variable] = range
            del self.fixed_variables[variable]

    def free(
        self,
        variables: dict = None,
        fix_rest: bool = False,
        **kwargs,
    ):
        """
        The method accpets: 
        Optimize.fix({"var_1": (0, 1), "var_2": (0, 2)}), 
        Optimize.fix(var_1 = (0, 1), var_2 = (0, 2))
        """

        if variables is None:
            variables = kwargs

        if not isinstance(variables, dict):
            raise ValueError(f"Only accept dict as the input.")

        for key, val in variables.items():
            self._free(key, val)

        if fix_rest:
            remaining_var = [var for var in self.free_variables.keys()
                             if var not in variables.keys()]
            self.fix(remaining_var)

        self._update_free_name_list()

    def _normalize_input(self, variables: dict):
        new_var = variables.copy()

        for var, range in self.free_variables.items():
            low, high = range
            new_var[var] = (new_var[var] - low) / (high - low)

        return new_var

    def _denormalize_input(self, variables: dict):
        new_var = variables.copy()

        for var, range in self.free_variables.items():
            low, high = range
            new_var[var] = new_var[var] * (high - low) + low

        return new_var

    def _normalize_output(self, output):
        return output / TARGET_NORMALIZE

    def _denormalize_output(self, output):
        return output * TARGET_NORMALIZE

    def _x_arr_2_dict(self, x: np.ndarray | List):
        return OrderedDict(zip(self.free_name_list, x))

    def _x_dict_2_arr(self, x: dict):
        return [x[name] for name in self.free_name_list]

    def target(self, free_var: dict | List):
        """
        Input: free variable dict
        """
        return self.target_func(self.fixed_variables | free_var, **self.target_kwargs)

    def _opt_func(self, x):
        """
        Input should be a LIST of free variable in the order of self.free_name_list. 
        But this is totally implicit for the user. 
        """
        x_dict = self._x_arr_2_dict(x)
        denorm_x = self._denormalize_input(x_dict)

        target = self._normalize_output(self.target(denorm_x))

        return target

    def opt_init(
        self,
        x_dict: dict = {},
        check_func: Callable = lambda *args, **kwargs: True,
        check_kwargs: dict = {}
    ) -> OptTraj:
        """
        check legal initialization func: check_func(full_dict, **check_kwargs)
        """
        if x_dict == {}:
            while True:
                norm_init = np.random.uniform(
                    low=0,
                    high=1,
                    size=len(self.free_name_list)
                )
                norm_init_dict = OrderedDict(
                    zip(self.free_name_list, norm_init))
                denorm_init_dict = self._denormalize_input(norm_init_dict)
                if check_func(self.fixed_variables | denorm_init_dict, **check_kwargs):
                    return denorm_init_dict
        else:
            denorm_init_dict = self._denormalize_input(norm_init_dict)
            if not check_func(self.fixed_variables | denorm_init_dict, **check_kwargs):
                raise ValueError(f"invalid")
            return denorm_init_dict

    def run(
        self,
        init_x: dict = {},
        call_back: Callable = None,
        check_func: Callable = lambda x: True,
        check_kwargs: dict = {}
    ):
        """
        If not specifying the initial x, a random x within range will be used.  
        Call back function: call_back(full_dict, target, constr)  
        Check legal initialization func: check_func(full_dict, **check_kwargs)
        """
        if init_x == {}:
            init_x = self.opt_init(check_func=check_func,
                                   check_kwargs=check_kwargs)

        init_x_arr = self._x_dict_2_arr(self._normalize_input(init_x))

        def evaluate_record(x):
            x_dict = self._x_arr_2_dict(x)
            denorm_x = self._denormalize_input(x_dict)

            target = self.target_func(
                self.fixed_variables | denorm_x, **self.target_kwargs)
            constr = 0

            return denorm_x, target, constr

        init_denorm_x, init_target, init_constr = evaluate_record(init_x_arr)
        result = OptTraj(
            self.free_name_list,
            np.array([self._x_dict_2_arr(init_denorm_x)]),
            np.array([init_target]),
            np.array([init_constr]),
        )

        def opt_call_back(x, convergence=None):
            denorm_x, target, constr = evaluate_record(x)

            result.append(
                denorm_x,
                target,
                constr
            )

            if call_back is not None:
                call_back(
                    denorm_x.copy(),
                    target,
                    constr,
                )

        opt_bounds = [[0.0, 1.0]] * len(self.free_name_list)
        if self.optimizer in ("L-BFGS-B", "Nelder-Mead", "Powell"):
            scipy_res = minimize(
                self._opt_func,
                x0=init_x_arr,
                bounds=opt_bounds,
                callback=opt_call_back,
                method=self.optimizer,
                # options={"maxls": 20}
            )
        elif self.optimizer == "shgo":
            scipy_res = shgo(
                self._opt_func,
                bounds=opt_bounds,
                callback=opt_call_back,
            )
        elif self.optimizer == "differential evolution":
            scipy_res = differential_evolution(
                self._opt_func,
                bounds=opt_bounds,
                callback=opt_call_back,
            )

        return result


class MultiOpt():
    def __init__(
        self,
        optimize: Optimization,
    ):
        self.optimize = optimize

    def run(
        self,
        run_num,
        call_back: Callable = None,
        check_func: Callable = lambda x: True,
        check_kwargs: dict = {},
    ):
        multi_result = MultiTraj()
        for _ in tqdm(range(run_num)):
            result = self.optimize.run(
                init_x={},
                call_back=call_back,
                check_func=check_func,
                check_kwargs=check_kwargs,
            )
            multi_result.append(result)

        return multi_result
