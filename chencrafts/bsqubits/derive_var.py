import numpy as np
from scipy.constants import h, k
import scqubits as scq

from collections import OrderedDict
from typing import List, Dict, Callable, Tuple

from chencrafts.toolbox.data_processing import (
    NSArray,
    DimensionModify
)
from chencrafts.bsqubits.ec_systems import (
    CavityTmonSys,
)
from chencrafts.bsqubits.basis_n_states import cat
from chencrafts.bsqubits.sweeps import tmon_sweep_dict

PI2 = np.pi * 2

def _var_dict_2_shape_dict(var_dict):
    shape_dict = OrderedDict({})
    for key, val in var_dict.items():
        shape_dict[key] = len(val)
    return shape_dict

def _n_th(omega, temp):
        """omega is in the unit of GHz"""
        return 1 / (np.exp(omega * h * 1e9 / temp / k) - 1)

class DerivedVariableBase():
    scq_available_var: List[str] = []
    def __init__(
        self,
        para: dict[str, float], 
        sim_para: dict[str, float],
        swept_para_dict: dict[str, List | np.ndarray] = {},
    ):
        # independent parameters: fixed + simulation + varied
        self.para_dict = para
        self.sim_para = sim_para
        self.sweep_para_dict = OrderedDict([(key, np.array(val)) 
        for key, val in swept_para_dict.items()])

        # output
        if self.sweep_para_dict != {}:
            # self.para_dict_to_use is a meshgrid if the user want to sweep 
            self.para_dict_to_use = self._meshgrid()
        else:
            self.para_dict_to_use = OrderedDict([(key, NSArray(val)) 
                for key, val in self.para_dict.items()])
        self.derived_dict = OrderedDict({})

        # dimension modify
        self._scq_sweep_shape = self._init_scq_sweep_shape()
        self._target_shape = _var_dict_2_shape_dict(self.sweep_para_dict)
        self._dim_modify = DimensionModify(
            self._scq_sweep_shape,
            self._target_shape
        )

    def __getitem__(
        self,
        name: str,
    ):
        try:
            return self.para_dict_to_use[name]
        except KeyError:
            pass

        try:
            return self.derived_dict[name]
        except KeyError:
            raise KeyError(f"{name} not found in the parameters including the derived one. "
            "If you didn't call use `evaluate()`, try it.")

    def _init_scq_sweep_shape(self) -> OrderedDict:
        """
        available_scq_sweep_name is a class constant, 
        for example, it can be ["omega_s", "g_sa", "EJ", "EC"]
        """

        scq_sweep_shape = OrderedDict({})
        for key in self.scq_available_var:
            if key in self.sweep_para_dict.keys():
                scq_sweep_shape[key] = len(self.sweep_para_dict[key])
            else:
                scq_sweep_shape[key] = 1

        return scq_sweep_shape

    def _meshgrid(self):
        variable_mesh_dict = OrderedDict(zip(
            self.sweep_para_dict.keys(),
            np.meshgrid(*self.sweep_para_dict.values(), indexing="ij")
        ))
        
        full_para_mesh = OrderedDict({})
        shape = list(variable_mesh_dict.values())[0].shape
        for key, val in self.para_dict.items():
            if key in self.sweep_para_dict.keys():
                mesh = variable_mesh_dict[key]
            else:
                mesh = np.ones(shape) * val

            full_para_mesh[key] = NSArray(mesh, self.sweep_para_dict)

        return full_para_mesh

    def _sweep_wrapper(
        self, 
        nsarray: NSArray, 
        from_scq_sweep: bool = True
    ):
        if from_scq_sweep:
            shaped_array = self._dim_modify(nsarray)
        else:
            dim_modify = DimensionModify(
                _var_dict_2_shape_dict(nsarray.param_info),
                self._target_shape
            )
            shaped_array = dim_modify(nsarray)

        return NSArray(
            shaped_array,
            self.sweep_para_dict
        )
    
    @property
    def full_para(self):
        return self.para_dict_to_use | self.derived_dict

    def keys(self):
        return self.full_para.keys()
    
    def values(self):
        return self.full_para.values()

    def items(self):
        return self.full_para.items()

    def _extra_sweep(
        self, 
        func: Callable, 
        extra_input_dict: Dict[str, np.ndarray] = {},
        output_elem_shape: float = 1,
        kwargs: Dict[str, np.ndarray] = {},
    ) -> NSArray | List[NSArray]:
        """
        _sweep_wrapper() is NOT automatically applied, so it should be applied externally

        func should have the form:  
        func(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs)
        """
        try:
            sweep: scq.ParameterSweep = self.sweep
        except NameError:
            raise RuntimeError("Run sweep before using add_sweep!")

        # if there is no extra dimension
        if extra_input_dict == {}:
            sweep.add_sweep(func, "tmporary_derived_variable", **kwargs)
            return sweep["tmporary_derived_variable"].copy()

        # shape
        overall_sweep_dict = sweep.param_info | extra_input_dict
        shape = tuple(_var_dict_2_shape_dict(overall_sweep_dict).values())
        extra_shape = tuple(_var_dict_2_shape_dict(extra_input_dict).values())

        # initialize an empty nsarray
        data = NSArray(
            np.zeros(shape + output_elem_shape),
            overall_sweep_dict
        )

        # iterating on extra dimensions
        select_all_scq_dim = (slice(None),) * len(sweep.param_info)

        for idx_tuple in np.ndindex(extra_shape):
            var_dict = dict([
                (key, val_list[idx_tuple[idx]]) for idx, (key, val_list) in enumerate(extra_input_dict.items())
            ])

            sweep.add_sweep(
                func, 
                "tmporary_derived_variable", 
                **var_dict,
                **kwargs
            )

            full_idx = select_all_scq_dim + idx_tuple + (slice(None),) * len(output_elem_shape)
            data[full_idx] = sweep["tmporary_derived_variable"].copy()

        return data


    def _evaluate_extra_sweep_from_dict(
        self, 
        sweep_dict: Dict[str, Tuple[Callable, Tuple[str]]], 
        kwargs: Dict
    ):
        """
        dictionary key is a str or a tuple: output_name or (output_names)  
        dict values is a tuple: (function, input_names)  
        the function should return a np.array object
        """
        for out_var_name, (func, in_var_name) in sweep_dict.items():
            sweep_dict = {}
            for key in in_var_name:
                if key in self.sweep_para_dict.keys():
                    sweep_dict[key] = self.sweep_para_dict[key]
                else:
                    sweep_dict[key] = [self.para_dict[key]]

            if isinstance(out_var_name, str):
                data = self._extra_sweep(
                    func, 
                    sweep_dict,
                    kwargs=kwargs,
                    output_elem_shape=tuple()
                )
                self.derived_dict[out_var_name] = self._sweep_wrapper(
                    data, from_scq_sweep=(sweep_dict == {})
                )
            elif isinstance(out_var_name, tuple | list):
                data = self._extra_sweep(
                    func, 
                    sweep_dict,
                    kwargs=kwargs,
                    output_elem_shape=(len(out_var_name),)
                )
                for idx, key in enumerate(out_var_name):
                    self.derived_dict[key] = self._sweep_wrapper(
                        data[..., idx], from_scq_sweep=(sweep_dict == {})
                    )


class DerivedVariableTmon(DerivedVariableBase):
    scq_available_var = CavityTmonSys.sweep_available_name     # Order is important!!

    def __init__(
        self, 
        para: OrderedDict, 
        sim_para: OrderedDict, 
        swept_para_dict: dict = {},
    ):
        super().__init__(
            para, 
            sim_para, 
            swept_para_dict,
        )

    def evaluate(
        self,
        convergence_range = (1e-8, 1e-4),
        update_ncut = True,
        return_full_para = True,
    ):

        # evaluate eigensystem using scq.ParameterSweep
        if np.allclose(list(self._scq_sweep_shape.values()), 1):
            self.system = CavityTmonSys(
                self.para_dict,
                self.sim_para,
                {},
                convergence_range = convergence_range,
                update_ncut = update_ncut,
            )

        else:
            self.system = CavityTmonSys(
                self.para_dict,
                self.sim_para,
                self.sweep_para_dict,
                convergence_range = None,
                update_ncut = False,
            )
        self.sweep = self.system.sweep()

        # Store the data that directly come from the sweep
        self.derived_dict.update(dict(
            chi_sa = PI2 * self._sweep_wrapper(
                self.sweep["chi"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
            K_s = PI2 * self._sweep_wrapper(
                self.sweep["kerr"]["subsys1": 0, "subsys2": 0], 
            ), 
            chi_prime = PI2 * self._sweep_wrapper(
                self.sweep["chi_prime"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
        ))

        # Evaluate extra sweep over parameter outside of the self.scq_available_var
        a_s = self.system.a_s()
        a_dag_a = a_s.dag() * a_s
        sig_p_sig_m = self.system.proj_a(1, 1)
        self._evaluate_extra_sweep_from_dict(
            tmon_sweep_dict, 
            kwargs={
                "a_dag_a": a_dag_a,
                "sig_p_sig_m": sig_p_sig_m,
            },
        )

        # Evaluate the derived variables that can be simply calculated by elementary functions
        self.derived_dict.update(dict(
            Gamma_phi = self["Gamma_phi_ng"] + self["Gamma_phi_cc"],
            Gamma_up_ro = self["Gamma_up"].copy(),
            Gamma_down_ro = self["Gamma_down"].copy(),
            kappa_s = PI2 * self["omega_s"] / self["Q_s"],
            n_th = _n_th(self["omega_s"], self["temp_s"]), 
            T_M = self["T_W"] + self["tau_FD"] + self["tau_m"] 
                + np.pi / np.abs(self["chi_sa"]) + 12 * self["sigma"], 
        ))
        self.derived_dict.update(dict(
            cavity_loss = self["kappa_s"] * self["n_bar"] 
                + self["Gamma_down"] * self["anc_excitation"]
        ))
        
        if not return_full_para:
            return self.derived_dict
        else:
            full_dict = self.para_dict_to_use.copy()
            full_dict.update(self.derived_dict)
            return full_dict

