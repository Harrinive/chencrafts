from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List

# Outdated error channel class, accept list/dict as inputs 
# ##############################################################################
class ErrorChannelArrIpt:
    def __init__(
        self, 
        name: str, 
        input_var_names: List[str], 
        expression: Callable,
        default_input_mode: str = "original",
        full_var_names: List[str] = None,
    ):
        self.name = name
        self.input_var_names = input_var_names
        self.expression = expression

        if default_input_mode in ["original", "full", "dict"]:
            self.default_input_mode = default_input_mode
        else:
            raise ValueError(f"Input mode {default_input_mode} is invalid, only use "
            "\"original\", \"full\" and \"dict\".")

        if full_var_names is not None:
            self.variable_full_idx = (
                [full_var_names.index(variable) for variable in self.input_var_names]
            )
        else:
            self.variable_full_idx = None

    def __call__(self, args, input_mode=None):
        # try the assigned input_mode first
        if input_mode == "original":
            return self._error_rate(args)
        elif input_mode == "full":
            return self._er_from_full(args)
        elif input_mode == "dict":
            return self._er_from_dict(args)
        elif input_mode is None:
            if self.default_input_mode == "original":
                return self._error_rate(args)
            elif self.default_input_mode == "full":
                return self._er_from_full(args)
            elif self.default_input_mode == "dict":
                return self._er_from_dict(args)
        else:
            raise ValueError(f"Input mode {input_mode} is invalid, should use "
            "None, \"original\", \"full\" and \"dict\".")

    def update_full_var_names(
        self,
        full_var_names: List[str],
    ):
        self.variable_full_idx = (
            [full_var_names.index(variable) for variable in self.input_var_names]
        )

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

        args = np.array(args)
        params = args[self.variable_full_idx]
        return self.expression(*params)

    def _er_from_dict(self, arg_dict):
        # input should be a dict
        if not isinstance(arg_dict, dict):
            raise ValueError("Please input a dictionary")

        params = [arg_dict[arg] for arg in self.input_var_names]
        return self.expression(*params)

class ErrorRateArrIpt:
    def __init__(
        self, 
        full_var_names: List[str],
        default_input_mode: str = "full",
    ):
        self.full_var_names = full_var_names

        if default_input_mode in ["original", "full", "dict"]:
            self.default_input_mode = default_input_mode
        else:
            raise ValueError(f"Input mode {default_input_mode} is invalid, only use "
            "\"original\", \"full\" and \"dict\".")

        self.error_channels = OrderedDict({})
        self.channel_enable_info = OrderedDict({})

    def __call__(
        self,
        args,
        return_dict: bool = False,
        input_mode: str = None,
    ):
        """returns the total enabled error rate"""
        if return_dict:
            error_dict = OrderedDict({})
        else:
            total_error = 0

        for name, error_channel in self.error_channels.items():
            error_rate = (
                error_channel(args, input_mode=input_mode)
                * self.channel_enable_info[name]
            )
            
            if return_dict:
                error_dict[name] = error_rate
            else:
                total_error += error_rate

        if return_dict:
            return error_dict
        else:
            return total_error

    def __getitem__(
        self,
        error_name
    ):
        """calculate the error rate from a single channel"""
        return self.error_channels[error_name]

    def add_channel(
        self,
        name: str, 
        input_var_names: List[str], 
        expression: Callable,
    ):
        channel = ErrorChannel(
            name,
            input_var_names, 
            expression,
            self.default_input_mode,
            self.full_var_names,
        )
        self.error_channels[name] = channel
        self.channel_enable_info[name] = True

    def add_existed_channel(
        self,
        channel: ErrorChannel,
    ):  
        name = channel.name

        channel.default_input_mode = self.default_input_mode
        channel.update_full_var_names(self.full_var_names)

        self.error_channels[name] = channel
        self.channel_enable_info[name] = True

    def disable_channel(
        self,
        name: str
    ):
        self.channel_enable_info[name] = False

    def enable_channel(
        self,
        name: str
    ):
        self.channel_enable_info[name] = True
        