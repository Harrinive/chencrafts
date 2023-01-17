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
# from robo.fmin import bayesian_optimization

from typing import Callable, Dict, List

from tqdm.notebook import tqdm
import os

# from chencrafts.bsqubits.error_rates import manual_constr
from chencrafts.toolbox.save import (
    save_variable_list_dict, 
    load_variable_list_dict, 
    save_variable_dict,
    load_variable_dict,
)
from chencrafts.toolbox.plot import Cmap, filter


# ##############################################################################
def sample_from_range(range_dict: Dict) -> Dict[float]:
    new_dict = {}
    for key, (low, high) in range_dict.items():
        new_dict[key] = np.random.uniform(low, high)
    return new_dict

# ##############################################################################
TARGET_NORMALIZE = 1e-7

def nan_2_flat_val(full_variables, possible_nan_value):
    """
    The full_variables should contain "kappa_s" and "disp"
    """
    if np.isnan(possible_nan_value):
        return (full_variables["disp"])**2 * full_variables["kappa_s"]
    else:
        return possible_nan_value


# def nan_2_constr(full_variables, possible_nan_value):
#     """
#     The full_variables should contain "disp", "kappa_s", "g_sa", 
#     "min_detuning", "detuning_lower_bound", "constr_amp"
#     """
#     if np.isnan(possible_nan_value):
#         base_val = (full_variables["disp"])**2 * full_variables["kappa_s"]
#         val = base_val + manual_constr(**full_variables)
#         return val
#     else:
#         return possible_nan_value

# ##############################################################################
class OptTraj():
    def __init__(
        self,
        para_name: List[str],
        para_traj: np.ndarray,
        target_traj: np.ndarray,
        constr_traj: np.ndarray,
        fixed_para: Dict[str, float] = {},
    ):
        self.para_name = para_name
        self.para_traj = para_traj
        self.target_traj = target_traj
        self.constr_traj = constr_traj

        self.length = self.para_traj.shape[0]

        self.fixed_para = fixed_para

    @classmethod
    def from_file(cls, file_name, fixed_para_file_name = None):
        traj_dict = load_variable_list_dict(file_name, throw_nan=False)

        para_name = [name for name in traj_dict.keys() if name not in [
            "target", "constr"]]

        para_shape = [len(traj_dict[para_name[0]]), len(para_name)]
        para_traj = np.zeros(para_shape)
        for idx, name in enumerate(para_name):
            para_traj[:, idx] = traj_dict[name]

        if fixed_para_file_name is not None:
            fixed_para = load_variable_dict(fixed_para_file_name)
        else:
            fixed_para = {}

        instance = cls(
            para_name,
            para_traj,
            traj_dict["target"],
            traj_dict["constr"],
            fixed_para
        )

        return instance

    def __getitem__(self, name) -> np.ndarray:
        idx = self.para_name.index(name)
        return self.para_traj[:, idx]

    def _x_arr_2_dict(self, x: np.ndarray | List):
        return dict(zip(self.para_name, x))

    def _x_dict_2_arr(self, x: dict):
        return [x[name] for name in self.para_name]

    @property
    def final_para(self, full=False) -> Dict[str, float]:
        return self._x_arr_2_dict(self.para_traj[-1, :])

    @property
    def final_full_para(self) -> Dict[str, float]:
        return self._x_arr_2_dict(self.para_traj[-1, :]) | self.fixed_para

    @property
    def init_para(self, full=False) -> Dict[str, float]:
        return self._x_arr_2_dict(self.para_traj[0, :])

    @property
    def init_full_para(self) -> Dict[str, float]:
        return self._x_arr_2_dict(self.para_traj[0, :]) | self.fixed_para

    @property
    def final_target(self) -> float:
        return self.target_traj[-1]

    @property
    def init_target(self) -> float:
        return self.target_traj[0]

    def copy(self) -> "OptTraj":
        new_result = OptTraj(
            self.para_name,
            self.para_traj.copy(),
            self.target_traj.copy(),
            self.constr_traj.copy(),
            self.fixed_para.copy(),
        )
        return new_result

    def append(self, para_dict: dict, target: float, constr: float) -> None:
        para_arr = self._x_dict_2_arr(para_dict)
        self.para_traj = np.append(self.para_traj, [para_arr], axis=0)
        self.target_traj = np.append(self.target_traj, target)
        self.constr_traj = np.append(self.constr_traj, constr)
        self.length += 1

    def to_dict(self) -> Dict:
        traj_dict = {}
        for idx, key in enumerate(self.para_name):
            traj_dict[key] = self.para_traj[:, idx]
        traj_dict["target"] = self.target_traj
        traj_dict["constr"] = self.constr_traj
        return traj_dict

    def _normalize_para(self, para_range_dict: dict = {}) -> np.ndarray:
        new_var = self.para_traj.copy()

        for var, (low, high) in para_range_dict.items():
            idx = self.para_name.index(var)
            new_var[:, idx] = (new_var[:, idx] - low) / (high - low)

        return new_var

    def plot(self, para_range_dict: dict = {}, ax = None) -> None:
        # need further updating: use twin y axis for the target_traj
        normalized_para = self._normalize_para(para_range_dict)
        max_target = np.max(self.target_traj)

        need_show = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=150)
            need_show = True

        ax.plot(range(self.length), normalized_para, label=self.para_name)
        ax.plot(range(self.length), self.target_traj /
            max_target, label="normed_target")
        ax.plot(range(self.length), self.constr_traj /
            max_target, label="normed_constr")
        ax.legend()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Normalized Parameters")

        if need_show:
            plt.show()

    def plot_2d(
        self, 
        ax, 
        x_name,
        y_name,
        c: str = "white",
        destination_only: bool = True, 
        background_interp: Callable = None,
    ) -> None:
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

    def save(self, file_name, fixed_para_file_name = None):
        save_variable_list_dict(file_name, self.to_dict())
        if fixed_para_file_name is not None:
            save_variable_dict(fixed_para_file_name, self.fixed_para)


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
    ) -> "MultiTraj":
        new_list = cls()
        for traj in traj_list:
            new_list.append(traj)
        return new_list

    @classmethod
    def from_folder(
        cls,
        path,
        with_fixed = True,
    ) -> "MultiTraj":
        multi_traj = cls()

        path = os.path.normpath(path)
        if with_fixed:
            fixed_path = f"{path}/fixed.csv"
        else:
            fixed_path = None

        idx = 0
        while True:
            try:
                traj_path = f"{path}/{idx}.csv"

                traj = OptTraj.from_file(traj_path, fixed_path)
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

    def _target_list(self) -> List[float]:
        target_list = []
        for traj in self.traj_list:
            target_list.append(traj.final_target)

        return target_list

    def append(
        self,
        traj: OptTraj,
    ) -> None:
        self.traj_list.append(traj)
        self.length += 1

    def save(
        self,
        path: str,
    ) -> None:
        """
        Assume all of the OptTraj have the same fixed_para
        """
        path = os.path.normpath(path)
        for idx in range(self.length):
            self[idx].save(
                f"{path}/{idx}.csv", 
                fixed_para_file_name=f"{path}/fixed.csv"
            )

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

    def plot_target(self, ax=None, ylim=()):
        need_show = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.5), dpi=150)
            need_show = True

        best = self.best_traj()
        cmap = Cmap(self.length)
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
        ax.set_ylabel("Cost function")
        # ax.set_legend()
        ax.grid()

        if need_show:
            # plt.savefig("./figures/C2QA slides/error rates w iteration small.png")
            plt.tight_layout()
            plt.show()


# ##############################################################################
class Optimization():
    def __init__(
        self,
        fixed_variables: Dict[str, float],
        free_variable_ranges: Dict[str, List[float]],
        target_func: Callable,
        target_kwargs: dict = {},
        optimizer: str = "L-BFGS-B",
    ):
        """
        The target function should be like: 
        target_func(full_variable_dict, **kwargs)  
        Supported optimizers: L-BFGS-B, Nelder-Mead, Powell, shgo, differential evolution, bayesian optimization
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
                                  "shgo", "differential evolution", "bayesian optimization"]

    def _update_free_name_list(self):
        self.free_name_list = list(self.free_variables.keys())
        # print(f"Current order of input: {self.free_name_list}")

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

        if variable in self.fixed_variables:
            self.fixed_variables[variable] = value
        elif variable in self.free_variables:
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
        return dict(zip(self.free_name_list, x))

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
                norm_init_dict = dict(
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
        check_kwargs: dict = {},
        opt_options: dict = {},
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
            fixed_para = self.fixed_variables
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
                options=opt_options,
            )
        elif self.optimizer == "shgo":
            scipy_res = shgo(
                self._opt_func,
                bounds=opt_bounds,
                callback=opt_call_back,
                # options=opt_options,
            )
        elif self.optimizer == "differential evolution":
            scipy_res = differential_evolution(
                self._opt_func,
                bounds=opt_bounds,
                callback=opt_call_back,
                # options=opt_options,
            )
        # elif self.optimizer == "bayesian optimization":
        #     bo_res = bayesian_optimization(
        #         self._opt_func, 
        #         lower=opt_bounds[:, 0], 
        #         upper=opt_bounds[:, 1],
        #         num_iterations=len(self.free_name_list)
        #     )
        #     result = OptTraj(
        #         self.free_name_list,
        #         np.array(bo_res["X"]),
        #         np.array(bo_res["y"]),
        #         np.ones_like(bo_res["y"]) * np.nan,
        #         fixed_para = self.fixed_variables,
        #     )

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
        opt_options: dict = {},
        save_path: str = None,
    ):
        multi_result = MultiTraj()
        for _ in tqdm(range(run_num)):

            try: 
                result = self.optimize.run(
                    init_x={},
                    call_back=call_back,
                    check_func=check_func,
                    check_kwargs=check_kwargs,
                    opt_options=opt_options,
                )
            except ValueError as e:
                print(f"Capture a ValueError from optimization: {e}")
                continue

            multi_result.append(result)
            if save_path is not None:
                multi_result.save(save_path)

        return multi_result
