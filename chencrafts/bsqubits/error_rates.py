import numpy as np
import scqubits as scq

from scipy.constants import h, k
from chencrafts.bsqubits import single_sweep_tmon, sweep_tmon, generate_variable_meshgrids, generate_sweep_lists, state_sets, initialize_joint_system_tmon

from typing import Callable, List

class error_channel:
    def __init__(
        self, 
        name: str, 
        input_var_names: List[str], 
        expression: Callable,
        input_mode: str = "original",
        full_var_names: List[str] = None,
    ):
        self.name = name
        self.input_var_names = input_var_names
        self.expression = expression

        if input_mode in ["original", "full", "dict"]:
            self.input_mode = input_mode
        else:
            raise ValueError(f"Input mode {input_mode} is invalid, only use "
            "\"original\", \"full\" and \"dict\".")

        self.variable_full_idx = None
        if full_var_names is not None:
            self.variable_full_idx = (
                [full_var_names.index(variable) for variable in self.input_var_name]
            )
        elif self.input_mode == "full":
            raise ValueError(f"For using the full variable list as an input, "
            "you should specify the full_var_names")

    def __call__(self, args, input_mode=None):
        # try the assigned input_mode first
        if input_mode == "original":
            return self._error_rate(args)
        elif input_mode == "full":
            return self._er_from_full(args)
        elif input_mode == "dict":
            return self._er_from_dict(args)

        if self.input_mode is None:
            return self._error_rate(args)
        elif self.input_mode == "full":
            return self._er_from_full(args)
        elif self.input_mode == "dict":
            return self._er_from_dict(args)

    def _error_rate(self, args):
        # input should be an array/list
        if not (isinstance(args, list) or isinstance(args, np.ndarray)):
            raise ValueError("Please input an array")

        return self.expression(*args)

    def _er_from_full(self, args):
        # input should be an array/list
        if not (isinstance(args, list) or isinstance(args, np.ndarray)):
            raise ValueError("Please input an array")
        if self.variable_full_idx is None:
            raise ValueError(f"For using the full variable list as an input, "
            "you should specify the full_var_names")

        params = args[self.variable_full_idx]
        return self.expression(*params)

    def _er_from_dict(self, arg_dict):
        # input should be a dict
        if not isinstance(arg_dict, dict):
            raise ValueError("Please input a dictionary")

        params = [arg_dict[arg] for arg in self.input_var_names]
        return self.expression(*params)

class ErrorRateBase:
    # error channels
    # 
    pass
