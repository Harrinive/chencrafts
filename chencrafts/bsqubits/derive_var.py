import numpy as np
from scipy.constants import h, k
from scipy.special import erfc
import scqubits as scq

import warnings
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
    shape_dict = {}
    for key, val in var_dict.items():
        shape_dict[key] = len(val)
    return shape_dict

def _n_th(freq, temp):
    """freq is in the unit of GHz, temp is in the unit of K"""
    return 1 / (np.exp(freq * h * 1e9 / temp / k) - 1)

def _readout_error(disp, relax_rate, int_time) -> NSArray:
    SNR = 2 * np.abs(disp) * np.sqrt(relax_rate * int_time)
    return 0.5 * erfc(SNR / 2)

def _addit_rate_ro(kappa_down, n_ro, n_crit, lambda_2, kappa_r, kappa_phi) -> Tuple[NSArray]:
    k_down_ro = (kappa_down * (1 - (n_ro + 0.5) / 2 / n_crit) 
        + lambda_2 * kappa_r + 2 * lambda_2 * kappa_phi * (n_ro + 1))
    k_up_ro = 2 * lambda_2 * kappa_phi * n_ro
    return k_down_ro, k_up_ro

def _shot_noise(kappa_r, chi_ar, n_th_r) -> NSArray:
    return kappa_r / 2 * (np.sqrt(
        (1 + 1j * chi_ar / kappa_r)**2 + 4j * chi_ar * n_th_r / kappa_r
    ) - 1).real

class DerivedVariableBase():
    scq_available_var: List[str] = []
    default_para: Dict[str, float] = {}
    def __init__(
        self,
        para: dict[str, float], 
        sim_para: dict[str, float],
        swept_para_dict: dict[str, List | np.ndarray] = {},
    ):
        # independent parameters: fixed + simulation + varied
        self.para_dict = para
        self.sim_para = sim_para
        self.sweep_para_dict = dict([(key, np.array(val)) 
            for key, val in swept_para_dict.items()])

        # output
        if self.sweep_para_dict != {}:
            # self.para_dict_to_use is a meshgrid if the user want to sweep 
            self.para_dict_to_use = self._meshgrid(self._merge_default())
        else:
            self.para_dict_to_use = dict([(key, NSArray(val)) 
                for key, val in self._merge_default().items()])
        self.derived_dict = {}

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
    ) -> NSArray:
        try:
            return self.para_dict_to_use[name]
        except KeyError:
            pass

        try:
            return self.derived_dict[name]
        except KeyError:
            raise KeyError(f"{name} not found in the parameters including the derived one. "
            "If you didn't call use `evaluate()`, try it.")

    def _init_scq_sweep_shape(self) -> Dict:
        """
        available_scq_sweep_name is a class constant, 
        for example, it can be ["omega_s", "g_sa", "EJ", "EC"]
        """

        scq_sweep_shape = {}
        for key in self.scq_available_var:
            if key in self.sweep_para_dict.keys():
                scq_sweep_shape[key] = len(self.sweep_para_dict[key])
            else:
                scq_sweep_shape[key] = 1

        return scq_sweep_shape

    def _merge_default(self):
        return self.default_para | self.para_dict

    def _meshgrid(self, var_dict):
        variable_mesh_dict = dict(zip(
            self.sweep_para_dict.keys(),
            np.meshgrid(*self.sweep_para_dict.values(), indexing="ij")
        ))
        
        full_para_mesh = {}
        shape = list(variable_mesh_dict.values())[0].shape

        for key, val in var_dict.items():
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
        sweep_func_dict: Dict[str, Tuple[Callable, Tuple[str]]], 
        kwargs: Dict
    ):
        """
        dictionary key is a str or a tuple: output_name or (output_names)  
        dict values is a tuple: (function, input_names)  
        the function should return a np.array object
        """
        for out_var_name, (func, in_var_name) in sweep_func_dict.items():
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
    default_para: Dict[str, float] = dict(
        n_th_base = 0.0,
    )

    def __init__(
        self, 
        para: Dict[str, float], 
        sim_para: Dict[str, float], 
        swept_para_dict: Dict = {},
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
        """
        At this level, every energy should be in the angular frequency unit except 
        especial statement. 
        """
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
            omega_a_GHz = self._sweep_wrapper(
                self.sweep["bare_evals"][1][..., 1] 
                - self.sweep["bare_evals"][1][..., 0]
            ),
            det_01_GHz = self._sweep_wrapper(
                self.sweep["bare_evals"][1][..., 1] 
                - self.sweep["bare_evals"][1][..., 0]
            ) - self["omega_s"],
            det_12_GHz = self._sweep_wrapper(
                self.sweep["bare_evals"][1][..., 2] 
                - self.sweep["bare_evals"][1][..., 1]
            ) - self["omega_s"],
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
        self.derived_dict["2*K_a"] = PI2 * (self["det_12_GHz"] - self["det_01_GHz"])

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
        # 1st level
        self.derived_dict.update(dict(
            # bare decoherence rate
            kappa_s = PI2 * self["omega_s"] / self["Q_s"],
            kappa_a = self["kappa_cap"],
            kappa_phi = self["kappa_phi_ng"] + self["kappa_phi_cc"],
            n_th_s = _n_th(self["omega_s"], self["temp_s"]) + self["n_th_base"], 
            n_th_a = _n_th(self["omega_a_GHz"], self["temp_a"]) + self["n_th_base"], 
            # readout
            chi_ar = self["chi_ar/kappa_r"] * self["kappa_r"],
            sigma = self["sigma*2*K_a"] / np.abs(self["2*K_a"]),
        ))

        # #  Override the chi_ar/kappa_r ratio when using Ofek parameters
        # warnings.warn("Warning: Overriding the chi_ar calculation!")
        # self.derived_dict["chi_ar"] = 1e-3      

        # 2nd level
        lambda_2 = np.abs(self["chi_ar"] / self["2*K_a"])
        n_crit = (1 / 4 / lambda_2)
        n_ro = self["n_ro/n_crit"] * n_crit
        kappa_down_ro, kappa_up_ro = _addit_rate_ro(
            self["kappa_a"], n_ro, n_crit, lambda_2, self["kappa_r"], self["kappa_phi"]
        )
        self.derived_dict.update(dict(
            # readout
            lambda_ro = lambda_2,
            n_crit = n_crit,
            n_ro = n_ro,
            kappa_down_ro = kappa_down_ro,
            kappa_up_ro = kappa_up_ro,
            # additional coherence rates
            kappa_phi_r = _shot_noise(self["kappa_r"], self["chi_ar"], self["n_th_r"]),
            kappa_a_r = lambda_2 * self["kappa_r"],
            # pulse
            tau_p = self["sigma"] * np.abs(self["tau_p/sigma"]),
            tau_p_eff = self["sigma"] * np.abs(self["tau_p_eff/sigma"]),
        ))
        # 3rd level
        M_ge = _readout_error(
            np.sqrt(self["n_ro"]), 
            self["kappa_r"], 
            self["tau_m"]
        )
        self.derived_dict.update(dict(
            # readout
            M_ge = M_ge,
            M_eg = M_ge.copy(),
            # total decoherence rate
            gamma_down = self["kappa_s"] * self["n_bar_s"] * (1 + self["n_th_s"]) 
                + self["kappa_a"] * self["n_bar_a"],
            gamma_01_down = self["kappa_s"] * self["n_fock1_s"] * (1 + self["n_th_s"]) 
                + self["kappa_a"] * self["n_fock1_a"],
            gamma_up = self["kappa_s"] * (self["n_bar_s"] + 1) * self["n_th_s"] 
                + self["kappa_a"] * self["n_bar_a"] * self["n_th_a"],
            Gamma_down = self["kappa_a"] + self["kappa_a_r"],
            Gamma_up = (self["kappa_a"] + self["kappa_a_r"]) * self["n_th_a"],
            Gamma_phi = self["kappa_phi"] + self["kappa_phi_r"],
            Gamma_down_ro = self["kappa_a"] + self["kappa_down_ro"],
            Gamma_up_ro = self["kappa_a"] * self["n_th_a"] + self["kappa_up_ro"],
            # other
            T_M = self["T_W"] + self["tau_FD"] + self["tau_m"] 
                + np.pi / np.abs(self["chi_sa"]) + 3 * self["tau_p"], 
        ))

        if not return_full_para:
            return self.derived_dict
        else:
            full_dict = self.para_dict_to_use.copy()
            full_dict.update(self.derived_dict)
            return full_dict

