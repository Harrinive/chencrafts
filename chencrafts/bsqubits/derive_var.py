import numpy as np
from scipy.constants import h, k
import scqubits as scq

from collections import OrderedDict
from typing import List, Dict

from chencrafts.toolbox.data_processing import (
    NSArray,
    DimensionModify
)
from chencrafts.bsqubits.ec_systems import (
    JointSystemTmon,
)

PI2 = np.pi * 2

class DerivedVariableBase():
    scq_available_var: List[str] = []
    def __init__(
        self,
        para: dict[str, float], 
        sim_para: dict[str, float],
        swept_para_dict: dict[str, List | np.ndarray] = {},
    ):
        self.para_dict = para
        self.sim_para = sim_para
        
        self.sweep_para_dict = OrderedDict([(key, np.array(val)) 
        for key, val in swept_para_dict.items()])

        # output
        if self.sweep_para_dict != {}:
            # self.para_dict_to_use is a meshgrid if the user want to sweep 
            self.para_dict_to_use = self._meshgrid()
        else:
            self.para_dict_to_use = self.para_dict.copy()
        self.derived_dict = OrderedDict({})

        # dimension modify
        self._scq_sweep_shape = self._init_scq_sweep_shape()
        target_shape = OrderedDict()
        for key, val in self.sweep_para_dict.items():
            target_shape[key] = len(val)
        self._dim_modify = DimensionModify(
            self._scq_sweep_shape,
            target_shape
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

    def _sweep_wrapper(self, nsarray: NSArray):
        array = self._dim_modify(nsarray)
        return NSArray(
            array,
            self.sweep_para_dict
        )

    def _n_th(self, omega, temp):
        """omega is in the unit of GHz"""
        return 1 / (np.exp(omega * h * 1e9 / temp / k) - 1)
    
    @property
    def full_para(self):
        return self.para_dict_to_use | self.derived_dict

    def keys(self):
        return self.full_para.keys()
    
    def values(self):
        return self.full_para.values()

    def items(self):
        return self.full_para.items()


class DerivedVariableTmon(DerivedVariableBase):
    scq_available_var = JointSystemTmon.sweep_available_name     # Order is important!!

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

        if np.allclose(list(self._scq_sweep_shape.values()), 1):
            self.system = JointSystemTmon(
                self.para_dict,
                self.sim_para,
                {},
                convergence_range = convergence_range,
                update_ncut = update_ncut,
            )

        else:
            self.system = JointSystemTmon(
                self.para_dict,
                self.sim_para,
                self.sweep_para_dict,
                convergence_range = None,
                update_ncut = False,
            )

        self.sweep = self.system.sweep()

        self.derived_dict.update({
            "chi_sa": PI2 * self._sweep_wrapper(
                self.sweep["chi"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
            "K_s": PI2 * self._sweep_wrapper(
                self.sweep["kerr"]["subsys1": 0, "subsys2": 0], 
            ), 
            "chi_prime": PI2 * self._sweep_wrapper(
                self.sweep["chi_prime"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
            "Gamma_up": self._sweep_wrapper(self.sweep["gamma_up"]), 
            "Gamma_down": self._sweep_wrapper(self.sweep["gamma_down"]), 
            "Gamma_phi": self._sweep_wrapper(self.sweep["gamma_phi"]), 
            "Gamma_up_ro": self._sweep_wrapper(self.sweep["gamma_up"]), 
            "Gamma_down_ro": self._sweep_wrapper(self.sweep["gamma_down"]), 
            "min_detuning": PI2 * self._sweep_wrapper(self.sweep["min_detuning"]),
        })

        self.derived_dict.update({
            "n_bar": self["disp"],
            "n_th": self._n_th(self["omega_s"], self["temp_s"]), 
            "kappa_s": PI2 * self["omega_s"] / self["Q_s"]
                + self["Gamma_down"] * (PI2 * self["g_sa"] / self["min_detuning"])**2, 
            "T_M": self["T_W"] + self["tau_FD"] + self["tau_m"] 
                + np.pi / np.abs(self["chi_sa"]) + 12 * self["sigma"], 
        })

        self._sweep_jump_rate()

        if not return_full_para:
            return self.derived_dict
        else:
            full_dict = self.para_dict_to_use.copy()
            full_dict.update(self.derived_dict)
            return full_dict

    def _sweep_jump_rate(self):
        sweep_dict = self.sweep.param_info

