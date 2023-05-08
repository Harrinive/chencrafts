import numpy as np
from scipy.constants import h, k
from scipy.special import erfc
import scqubits as scq

from typing import List, Dict, Callable, Tuple, Any
import os

from chencrafts.toolbox import (
    NSArray,
    DimensionModify,
    save_variable_dict,
    load_variable_dict,
    save_variable_list_dict,
    load_variable_list_dict,
    dill_dump,
    dill_load,
)
from chencrafts.bsqubits.ec_systems import (
    CavityAncSystem,
    CavityTmonSys,
    CavityFlxnSys,
)
from chencrafts.bsqubits.sweeps import (
    tmon_sweep_dict, 
    flxn_sweep_dict,
    sweep_flxn_down,
    sweep_flxn_up,
)

PI2 = np.pi * 2

def _var_dict_2_shape_dict(var_dict):
    shape_dict = {}
    for key, val in var_dict.items():
        shape_dict[key] = len(val)
    return shape_dict

def _n_th(freq, temp, n_th_base=0.0):
    """freq is in the unit of GHz, temp is in the unit of K"""
    return 1 / (np.exp(freq * h * 1e9 / temp / k) - 1) + n_th_base

def _readout_error(disp, relax_rate, int_time) -> NSArray:
    SNR = 2 * np.abs(disp) * np.sqrt(relax_rate * int_time)
    return 0.5 * erfc(SNR / 2)

def _addit_rate_ro(kappa_down, n_ro, n_crit, lambda_2, kappa_r, kappa_phi) -> Tuple[NSArray]:
    k_down_ro = kappa_down * (- (n_ro + 0.5) / 2 / n_crit) \
        + lambda_2 * kappa_r + 2 * lambda_2 * kappa_phi * (n_ro + 1)
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
        para: Dict[str, float], 
        sim_para: Dict[str, float],
        swept_para: Dict[str, range | List | np.ndarray] = {},
    ):
        # independent parameters: fixed + simulation + varied
        self.para = para
        self.sim_para = sim_para
        self.swept_para: Dict[str, np.ndarray] = dict([(key, np.array(val)) 
            for key, val in swept_para.items()])

        # output
        if self.swept_para != {}:
            # self.para_dict_to_use is a meshgrid if the user want to sweep 
            self.para_dict_to_use = self._meshgrid(self._merge_default())
        else:
            self.para_dict_to_use = dict([(key, NSArray(val)) 
                for key, val in self._merge_default().items()])
        self.derived_dict = {}

        # dimension modify
        self._scq_sweep_shape = self._init_scq_sweep_shape()
        self._target_shape = _var_dict_2_shape_dict(self.swept_para)
        self._default_dim_modify = DimensionModify(
            self._scq_sweep_shape,
            self._target_shape
        )

        self.system: CavityAncSystem
        self.sweep: scq.ParameterSweep

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

    @classmethod
    # def from_file(cls, filename: str) -> "DerivedVariableBase":
    #     return dill_load(filename)

    @classmethod
    def from_export_folder(cls, path: str) -> "DerivedVariableBase":
        path = os.path.normpath(path)

        para = load_variable_dict(
            f"{path}/para.csv",
        )
        sim_para = load_variable_dict(
            f"{path}/sim_para.csv",
        )
        swept_para = load_variable_list_dict(
            f"{path}/sweep_para.csv",
            orient="index",
        )
        shape = [len(val) for val in swept_para.values()]
        
        new_der_para = cls(
            para, 
            sim_para,
            swept_para,
        )

        derived_dict = load_variable_list_dict(
            f"{path}/derived_para.csv",
            orient="columns",
            throw_nan=False
        )
        ns_derived_dict = dict([
            (key, NSArray(val.reshape(shape), swept_para)) for key, val in derived_dict.items()
        ])

        new_der_para.derived_dict = ns_derived_dict

        try: 
            new_der_para.system = dill_load(f"{path}/system.dill")
        except FileNotFoundError:
            pass
        try:
            new_der_para.sweep = dill_load(f"{path}/sweep.dill")
            # new_der_para.sweep = scq.read(f"{path}/sweep.h5")
        except FileNotFoundError:
            pass


        return new_der_para
    
    # def save(self, filename: str) -> None:
    #     dill_dump(self, filename)

    def export(self, path: str) -> None:
        path = os.path.normpath(path)

        save_variable_dict(
            f"{path}/para.csv",
            self.para,
        )
        save_variable_dict(
            f"{path}/sim_para.csv",
            self.sim_para,
        )
        save_variable_list_dict(
            f"{path}/sweep_para.csv",
            self.swept_para,
            orient="index",
        )

        flattened_derived_para = {}
        for key, val in self.derived_dict.items():
            try: 
                flattened_derived_para[key] = val.reshape(-1)
            except AttributeError as e:
                print(f"Data {key} is not saved due to AttributeError: {e}")

        save_variable_list_dict(
            f"{path}/derived_para.csv",
            flattened_derived_para,
            orient="columns",
        )

        try:
            dill_dump(self.system, f"{path}/system.dill")
        except NameError:
            pass
        try:
            dill_dump(self.sweep, f"{path}/sweep.dill")
            # self.sweep.filewrite(f"{path}/sweep.h5")
        except NameError:
            pass

    def _init_scq_sweep_shape(self) -> Dict:
        """
        available_scq_sweep_name is a class constant, 
        for example, it can be ["omega_s_GHz", "g_sa_GHz", "EJ_GHz", "EC_GHz"]
        """

        scq_sweep_shape = {}
        for key in self.scq_available_var:
            if key in self.swept_para.keys():
                scq_sweep_shape[key] = len(self.swept_para[key])
            else:
                scq_sweep_shape[key] = 1

        return scq_sweep_shape

    def _merge_default(self):
        return self.default_para | self.para

    def _meshgrid(self, var_dict):
        variable_mesh_dict = dict(zip(
            self.swept_para.keys(),
            np.meshgrid(*self.swept_para.values(), indexing="ij")
        ))
        
        full_para_mesh = {}
        shape = list(variable_mesh_dict.values())[0].shape

        for key, val in var_dict.items():
            if key in self.swept_para.keys():
                mesh = variable_mesh_dict[key]
            else:
                mesh = np.ones(shape) * val

            full_para_mesh[key] = NSArray(mesh, self.swept_para)

        return full_para_mesh
    
    @property
    def full_para(self):
        para = self.para_dict_to_use | self.derived_dict
        return dict(sorted(para.items()))

    def keys(self):
        return self.full_para.keys()

    def values(self):
        return self.full_para.values()

    def items(self):
        return self.full_para.items()
    
    def _dim_modify(
        self,
        nsarray: NSArray,
        target_shape: Dict[str, int],
    ) -> NSArray:
        dim_modify = DimensionModify(
            _var_dict_2_shape_dict(nsarray.param_info),
            target_shape
        )
        return dim_modify(nsarray)

    def scq_sweep_wrapper(
        self, 
        scq_nsarray: NSArray, 
        output_val_lists_dict: Dict[str, List | np.ndarray | range | None] = {},
        from_scq_sweep: bool = False,
    ):
        """ 
        wrapping the data directly coming from the scq.ParameterSweep

        extra_shape_dict should be non-empty when then OUTPUT data have extra dimension
        which is not included in self.swept_para and not named in scqubits output. 
        For example, for the evals sweep, the data has one more dimension to store a list of evals. 
        """
        if from_scq_sweep and output_val_lists_dict == {}:
            shaped_array = self._default_dim_modify(scq_nsarray)

            return NSArray(
                shaped_array,
                self.swept_para
            )
        
        # check the extra dimension is actually the missing part
        current_dim = len(scq_nsarray.shape)
        extra_dim = len(output_val_lists_dict)
        named_dim = len(scq_nsarray.param_info)
        if current_dim != extra_dim + named_dim:
            raise ValueError(f"extra_dim_name's length ({extra_dim}) should match the dimension "
                f"with missing names ({current_dim - named_dim})")
        
        # find the missing dimensions' length
        output_shape = {}
        for idx, (name, val_list) in enumerate(output_val_lists_dict.items()):
            if val_list is None:
                output_val_lists_dict[name] = range(scq_nsarray.shape[named_dim + idx])
            output_shape[name] = scq_nsarray.shape[named_dim + idx]

        # add the output dimensions
        scq_nsarray = NSArray(
            scq_nsarray,
            scq_nsarray.param_info | output_val_lists_dict
        )

        target_shape = self._target_shape | output_shape

        result = self._dim_modify(
            scq_nsarray,
            target_shape
        )
        return NSArray(
            result,
            self.swept_para | output_val_lists_dict
        )

    def extra_sweep(
        self, 
        func: Callable, 
        extra_dependencies: List[str] = [],
        output_val_lists_dict: Dict = {},
        kwargs: Dict[str, Any] = {},
    ) -> NSArray:
        """
        sweep_wrapper() is NOT automatically applied, so it should be applied externally

        func should have the form:  
        func(paramsweep, paramindex_tuple, paramvals_tuple, **kwargs)

        extra_dependencies: a list of variables that is not in cls.scq_available_var

        output_val_lists_dict is a dictionary describing the output elements in each calculation. Each dimension has a name and a variable list (or None)
        """
        try:
            sweep: scq.ParameterSweep = self.sweep
        except NameError:
            raise RuntimeError("Run sweep before using add_sweep!")

        # if there is no extra dimension
        if extra_dependencies == []:
            sweep.add_sweep(func, "tmporary_derived_variable", **kwargs)
            return sweep["tmporary_derived_variable"].copy()
        # if there is, cook up a new swept variable dictionary
        extra_input_dict = {}
        for key in extra_dependencies:
            if key in self.swept_para.keys():
                extra_input_dict[key] = np.array(self.swept_para[key])
            else:
                extra_input_dict[key] = np.array([self.para[key]])

        # shape
        overall_input_dict = sweep.param_info | extra_input_dict
        overall_shape = tuple(_var_dict_2_shape_dict(overall_input_dict).values())
        extra_input_shape = tuple(_var_dict_2_shape_dict(extra_input_dict).values())

        # iterating on extra dimensions
        select_all_scq_dim = (slice(None),) * len(sweep.param_info)

        for count, idx_tuple in enumerate(np.ndindex(extra_input_shape)):
            var_dict = dict([
                (key, val_list[idx_tuple[idx]]) for idx, (key, val_list) in enumerate(extra_input_dict.items())
            ])

            sweep.add_sweep(
                func, 
                "tmporary_derived_variable", 
                **var_dict,
                **kwargs
            )
            scq_result = sweep["tmporary_derived_variable"].copy()

            # detect the shape and then initialize the storage array
            if count == 0:
                named_dim = len(scq_result.param_info)
                for idx, (name, val_list) in enumerate(output_val_lists_dict.items()):
                    if val_list is None:
                        output_val_lists_dict[name] = range(scq_result.shape[named_dim + idx])

                output_elem_shape = tuple(_var_dict_2_shape_dict(output_val_lists_dict).values())

                # initialize an empty nsarray
                data = NSArray(
                    np.zeros(overall_shape + output_elem_shape),
                    overall_input_dict | output_val_lists_dict
                )

            full_idx = select_all_scq_dim + idx_tuple + (slice(None),) * len(output_val_lists_dict)
            data[full_idx] = scq_result

        data = self._dim_modify(
            data,
            _var_dict_2_shape_dict(self.swept_para | output_val_lists_dict)
        )

        return NSArray(
            data,
            self.swept_para | output_val_lists_dict,
        )

    # def evaluate_extra_sweep_from_dict(
    #     self, 
    #     sweep_func_dict: Dict[str | Tuple[str,...], Tuple[Callable, Tuple[str, ...]]], 
    #     kwargs: Dict
    # ):
    #     """
    #     dictionary key is a str or a tuple: output_name or (output_names)  
    #     dict values is a tuple: (function, input_names)  
    #     the function should return a np.array object
    #     """
    #     result_dict = {}

    #     for out_var_name, (func, in_var_name) in sweep_func_dict.items():
    #         sweep_dict = {}
    #         for key in in_var_name:
    #             if key in self.swept_para.keys():
    #                 sweep_dict[key] = self.swept_para[key]
    #             else:
    #                 sweep_dict[key] = [self.para[key]]

    #         if isinstance(out_var_name, str):
    #             data = self.extra_sweep(
    #                 func, 
    #                 sweep_dict,
    #                 kwargs=kwargs,
    #                 output_vals_dict={},
    #             )
    #             result_dict[out_var_name] = self.sweep_wrapper(
    #                 data, from_scq_sweep=(sweep_dict == {})
    #             )
    #         elif isinstance(out_var_name, tuple | list):
    #             data = self.extra_sweep(
    #                 func, 
    #                 sweep_dict,
    #                 kwargs=kwargs,
    #                 output_vals_dict={},
    #                 output_elem_shape=(len(out_var_name),)
    #             )
    #             for idx, key in enumerate(out_var_name):
    #                 result_dict[key] = self.sweep_wrapper(
    #                     data[..., idx], from_scq_sweep=(sweep_dict == {})
    #                 )

    #     return result_dict


class DerivedVariableTmon(DerivedVariableBase):
    scq_available_var = CavityTmonSys.sweep_available_name     # Order is important!!
    default_para: Dict[str, float] = dict(
        n_th_base = 0.0,
    )

    def __init__(
        self, 
        para: Dict[str, float], 
        sim_para: Dict[str, float], 
        swept_para: Dict = {},
    ):
        super().__init__(
            para, 
            sim_para, 
            swept_para,
        )

    def evaluate(
        self,
        convergence_range = (1e-8, 1e-4),
        update_ncut = True,
        store_intermediate_result = True,
    ):
        """
        At this level, every energy should be in the angular frequency unit except 
        especial statement. 
        """
        # evaluate eigensystem using scq.ParameterSweep
        if np.allclose(list(self._scq_sweep_shape.values()), 1):
            self.system = CavityTmonSys(
                self.para,
                self.sim_para,
                {},
                convergence_range = convergence_range,
                update_ncut = update_ncut,
            )

        else:
            self.system = CavityTmonSys(
                self.para,
                self.sim_para,
                self.swept_para,
                convergence_range = None,
                update_ncut = False,
            )

        self.sweep = self.system.sweep()

        # Store the data that directly come from the sweep
        self.derived_dict.update(dict(
            omega_a_GHz = self.scq_sweep_wrapper(
                self.sweep["bare_evals"][1][..., 1] 
                - self.sweep["bare_evals"][1][..., 0]
            ),
            chi_sa = PI2 * self.scq_sweep_wrapper(
                self.sweep["chi"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
            K_s = PI2 * self.scq_sweep_wrapper(
                self.sweep["kerr"]["subsys1": 0, "subsys2": 0], 
            ), 
            chi_prime = PI2 * self.scq_sweep_wrapper(
                self.sweep["chi_prime"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
        ))
        det_01_GHz = self.scq_sweep_wrapper(
            self.sweep["bare_evals"][1][..., 1] 
            - self.sweep["bare_evals"][1][..., 0]
        ) - self["omega_s_GHz"]
        det_12_GHz = self.scq_sweep_wrapper(
            self.sweep["bare_evals"][1][..., 2] 
            - self.sweep["bare_evals"][1][..., 1]
        ) - self["omega_s_GHz"]
        non_linr_a = PI2 * (det_12_GHz - det_01_GHz)   # "2*K_a" in note

        # Evaluate extra sweep over parameter outside of the self.scq_available_var
        a_s = self.system.a_s()
        a_dag_a = a_s.dag() * a_s
        sig_p_sig_m = self.system.proj_a(1, 1)

        ancilla_copy = scq.Transmon(
            EJ = self.system.ancilla.EJ,
            EC = self.system.ancilla.EC,
            ng = self.system.ancilla.ng,
            ncut = self.system.ancilla.ncut,
            truncated_dim = self.system.ancilla.truncated_dim,
        )

        kwargs = {
            "ancilla_copy": ancilla_copy,
            "a_dag_a": a_dag_a,
            "sig_p_sig_m": sig_p_sig_m,
            
        }

        func, extra_input_name, output_var_dict = tmon_sweep_dict["kappa_a"]
        n_bar = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )
        kappa_a_down, kappa_phi_ng, kappa_phi_cc = [n_bar[..., idx] for idx in range(3)]

        func, extra_input_name, output_var_dict = tmon_sweep_dict["conv"]
        conv = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )

        func, extra_input_name, output_var_dict = tmon_sweep_dict["n_bar"]
        n_bar = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )
        n_bar_s, n_bar_a, n_fock1_s, n_fock1_a = [n_bar[..., idx] for idx in range(4)]

        # Evaluate the derived variables that can be simply calculated by elementary functions
        # bare decoherence rate
        kappa_s = PI2 * self["omega_s_GHz"] / self["Q_s"]
        kappa_a = kappa_a_down
        kappa_phi = kappa_phi_ng + kappa_phi_cc
        n_th_s = _n_th(self["omega_s_GHz"], self["temp_s"], self["n_th_base"])
        n_th_a = _n_th(self["omega_a_GHz"], self["temp_a"], self["n_th_base"])

        # readout
        chi_ar = self["chi_ar/kappa_r"] * self["kappa_r"]
        sigma = self["sigma*2*K_a"] / np.abs(non_linr_a)

        lambda_2 = np.abs(chi_ar / non_linr_a)
        n_crit = (1 / 4 / lambda_2)
        n_ro = self["n_ro/n_crit"] * n_crit
        kappa_down_ro, kappa_up_ro = _addit_rate_ro(
            kappa_a, n_ro, n_crit, lambda_2, self["kappa_r"], kappa_phi
        )

        M_ge = _readout_error(
            np.sqrt(n_ro), 
            self["kappa_r"], 
            self["tau_m"],
        )
        M_eg = M_ge.copy()

        # additional coherence rates
        kappa_phi_r = _shot_noise(self["kappa_r"], chi_ar, self["n_th_r"])
        kappa_a_r = lambda_2 * self["kappa_r"]

        # pulse
        tau_p = sigma * np.abs(self["tau_p/sigma"])
        tau_p_eff = sigma * np.abs(self["tau_p_eff/sigma"])

        # total decoherence rate
        gamma_down = kappa_s * n_bar_s * (1 + n_th_s) \
            + kappa_a * n_bar_a
        gamma_01_down = kappa_s * n_fock1_s * (1 + n_th_s) \
            + kappa_a * n_fock1_a
        gamma_up = kappa_s * (n_bar_s + 1) * n_th_s \
            + kappa_a * n_bar_a * n_th_a
        Gamma_down = kappa_a + kappa_a_r
        Gamma_up = (kappa_a + kappa_a_r) * n_th_a
        Gamma_phi = kappa_phi + kappa_phi_r
        Gamma_down_ro = kappa_a + kappa_down_ro
        Gamma_up_ro = kappa_a * n_th_a + kappa_up_ro

        T1_a = 1 / Gamma_down
        T2_a = 1 / (Gamma_phi + Gamma_down / 2)
        T_s = 1 / gamma_01_down

        # other
        T_M = self["T_W"] + self["tau_FD"] + self["tau_m"] \
            + np.pi / np.abs(self["chi_sa"]) + 3 * tau_p

        # store the data into dictionaries
        intermediate_result = {}
        intermediate_result.update(dict(
            conv = conv, 
            
            n_bar_a = n_bar_a, 
            n_fock1_s = n_fock1_s, 
            n_fock1_a = n_fock1_a,

            det_01_GHz = det_01_GHz,
            det_12_GHz = det_12_GHz,
            non_linr_a = non_linr_a,

            kappa_s = kappa_s,
            kappa_a = kappa_a,
            kappa_phi = kappa_phi,
            n_th_s = n_th_s,
            n_th_a = n_th_a,

            chi_ar = chi_ar,
            sigma = sigma,

            lambda_2 = lambda_2,
            n_ro = n_ro,
            n_crit = n_crit,

            kappa_phi_r = kappa_phi_r,
            kappa_a_r = kappa_a_r,
        ))
        if store_intermediate_result:
            self.derived_dict.update(intermediate_result)

        self.derived_dict.update(dict(
            n_bar_s = n_bar_s,

            kappa_down_ro = kappa_down_ro,
            kappa_up_ro = kappa_up_ro,

            M_ge = M_ge,
            M_eg = M_eg,

            tau_p = tau_p,
            tau_p_eff = tau_p_eff,

            gamma_down = gamma_down,
            gamma_01_down = gamma_01_down,
            gamma_up = gamma_up,
            Gamma_down = Gamma_down, 
            Gamma_up = Gamma_up,
            Gamma_phi = Gamma_phi,
            Gamma_down_ro = Gamma_down_ro,
            Gamma_up_ro = Gamma_up_ro,

            T_M = T_M,

            T1_a = T1_a,
            T2_a = T2_a,
            T_s = T_s,
        ))

        return self.full_para


class DerivedVariableFlxn(DerivedVariableBase):
    scq_available_var = CavityFlxnSys.sweep_available_name     # Order is important!!
    default_para: Dict[str, float] = dict(
        n_th_base = 0.0,
    )

    def __init__(
        self, 
        para: Dict[str, float], 
        sim_para: Dict[str, float], 
        swept_para: Dict = {},
    ):
        super().__init__(
            para, 
            sim_para, 
            swept_para,
        )

    def evaluate(
        self,
        convergence_range = (1e-8, 1e-4),
        update_cutoff = True,
        store_intermediate_result = True,
    ):
        """
        At this level, every energy should be in the angular frequency unit except 
        especial statement. 
        """
        # evaluate eigensystem using scq.ParameterSweep
        if np.allclose(list(self._scq_sweep_shape.values()), 1):
            self.system = CavityFlxnSys(
                self.para,
                self.sim_para,
                {},
                convergence_range = convergence_range,
                update_cutoff = update_cutoff,
            )

        else:
            self.system = CavityFlxnSys(
                self.para,
                self.sim_para,
                self.swept_para,
                convergence_range = None,
                update_cutoff = False,
            )
        self.sweep = self.system.sweep()

        # Store the data that directly come from the sweep

        self.derived_dict.update(dict(
            chi_sa = PI2 * self.scq_sweep_wrapper(
                self.sweep["chi"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
            K_s = PI2 * self.scq_sweep_wrapper(
                self.sweep["kerr"]["subsys1": 0, "subsys2": 0], 
            ), 
            chi_prime = PI2 * self.scq_sweep_wrapper(
                self.sweep["chi_prime"]["subsys1": 0, "subsys2": 1][..., 1], 
            ), 
        ))

        # Evaluate extra sweep over parameter outside of the self.scq_available_var
        a_s = self.system.a_s()
        a_dag_a = a_s.dag() * a_s
        sig_p_sig_m = self.system.proj_a(1, 1)
        
        ancilla_copy = scq.Fluxonium(
            EJ = self.system.ancilla.EJ,
            EC = self.system.ancilla.EC,
            EL = self.system.ancilla.EL,
            flux = self.system.ancilla.flux,
            cutoff = self.system.ancilla.cutoff,
            truncated_dim = self.system.ancilla.truncated_dim,
        )

        kwargs = {
            "ancilla_copy": ancilla_copy,
            "a_dag_a": a_dag_a,
            "sig_p_sig_m": sig_p_sig_m,
            
        }

        func, extra_input_name, output_var_dict = flxn_sweep_dict["kappa_a_down"]
        kappa_a_down_channels = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )

        func, extra_input_name, output_var_dict = flxn_sweep_dict["kappa_a_up"]
        kappa_a_up_channels = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )

        func, extra_input_name, output_var_dict = flxn_sweep_dict["kappa_a_phi"]
        kappa_a_phi_channels = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )

        func, extra_input_name, output_var_dict = flxn_sweep_dict["conv"]
        conv = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )

        func, extra_input_name, output_var_dict = flxn_sweep_dict["n_bar"]
        n_bar = self.extra_sweep(
            func = func, 
            extra_dependencies = extra_input_name,
            output_val_lists_dict = output_var_dict,
            kwargs = kwargs,
        )
        n_bar_s, n_bar_a, n_fock1_s, n_fock1_a = [n_bar[..., idx] for idx in range(4)]

        # Evaluate the derived variables that can be simply calculated by elementary functions
        # bare decoherence rate
        kappa_s = PI2 * self["omega_s_GHz"] / self["Q_s"]
        kappa_a_down = np.sum(kappa_a_down_channels, axis=(-2, -1))
        kappa_a_up = np.sum(kappa_a_up_channels, axis=(-2, -1))
        kappa_phi = np.sum(kappa_a_phi_channels, axis=-1)
        n_th_s = _n_th(self["omega_s_GHz"], self["temp_s"], self["n_th_base"])
        # n_th_a = _n_th(self["omega_a_GHz"], self["temp_a"], self["n_th_base"])

        # readout
        chi_ar = self["chi_ar/kappa_r"] * self["kappa_r"]
        # sigma = self["sigma*2*K_a"] / np.abs(non_linr_a)  # set to be fixed
        sigma = self["sigma"]

        # lambda_2 = np.abs(chi_ar / non_linr_a)        # no non-lin for a flxn
        # n_crit = (1 / 4 / lambda_2)                   # no non-lin for a flxn
        # n_ro = self["n_ro/n_crit"] * n_crit           # set to be fixed
        n_ro = self["n_ro"]
        # kappa_down_ro, kappa_up_ro = _addit_rate_ro(
        #     kappa_a, n_ro, n_crit, lambda_2, self["kappa_r"], kappa_phi
        # )                                             # temporarily set to 0
        kappa_down_ro, kappa_up_ro = np.zeros_like(self["omega_s_GHz"]), np.zeros_like(self["omega_s_GHz"])

        M_ge = _readout_error(
            np.sqrt(n_ro), 
            self["kappa_r"], 
            self["tau_m"],
        )
        M_eg = M_ge.copy()

        # additional coherence rates
        kappa_phi_r = _shot_noise(self["kappa_r"], chi_ar, self["n_th_r"])
        # kappa_a_r = lambda_2 * self["kappa_r"]        # temporarily set to 0
        kappa_a_r = np.zeros_like(self["omega_s_GHz"])

        # pulse
        tau_p = sigma * np.abs(self["tau_p/sigma"])
        tau_p_eff = sigma * np.abs(self["tau_p_eff/sigma"])

        # total decoherence rate
        gamma_down = kappa_s * n_bar_s * (1 + n_th_s) \
            + kappa_a_down * n_bar_a
        gamma_01_down = kappa_s * n_fock1_s * (1 + n_th_s) \
            + kappa_a_down * n_fock1_a
        gamma_up = kappa_s * (n_bar_s + 1) * n_th_s \
            + kappa_a_up * n_bar_a
        Gamma_down = kappa_a_down + kappa_a_r
        Gamma_up = kappa_a_up       # readout induced excitation rate is not added
        # Gamma_up = (kappa_a + kappa_a_r) * n_th_a
        Gamma_phi = kappa_phi + kappa_phi_r
        Gamma_down_ro = kappa_a_down + kappa_down_ro
        Gamma_up_ro = kappa_a_up + kappa_up_ro

        T1_a = 1 / Gamma_down
        T2_a = 1 / (Gamma_phi + Gamma_down / 2)
        T_s = 1 / gamma_01_down

        # other
        T_M = self["T_W"] + self["tau_FD"] + self["tau_m"] \
            + np.pi / np.abs(self["chi_sa"]) + 3 * tau_p

        # store the data into dictionaries
        intermediate_result = {}

        # eval_dict = {}
        # for idx in range(int(self.sim_para["anc_dim"])):
        #     eval_dict[f"omega_a_{idx}_GHz"] = self.scq_sweep_wrapper(
        #         self.sweep["bare_evals"][1][..., idx] 
        #         - self.sweep["bare_evals"][1][..., 0]
        #     )
        # intermediate_result.update(eval_dict)

        intermediate_result.update(dict(
            conv = conv,
            # det_01_GHz = det_01_GHz,
            # det_02_GHz = det_02_GHz,
            # det_12_GHz = det_12_GHz,
            # non_linr_a = non_linr_a,

            kappa_s = kappa_s,
            kappa_a_down = kappa_a_down,
            kappa_a_up = kappa_a_up,
            kappa_phi = kappa_phi,
            n_th_s = n_th_s,
            # n_th_a = n_th_a,

            chi_ar = chi_ar,
            # sigma = sigma,

            # lambda_2 = lambda_2,
            # n_ro = n_ro,
            # n_crit = n_crit,

            kappa_phi_r = kappa_phi_r,
            kappa_a_r = kappa_a_r,
        ))
        if store_intermediate_result:
            self.derived_dict.update(intermediate_result)

        self.derived_dict.update(dict(
            n_bar_s = n_bar_s,
            n_bar_a = n_bar_a,
            n_fock1_s = n_fock1_s,
            n_fock1_a = n_fock1_a,

            kappa_down_ro = kappa_down_ro,
            kappa_up_ro = kappa_up_ro,

            M_ge = M_ge,
            M_eg = M_eg,

            tau_p = tau_p,
            tau_p_eff = tau_p_eff,

            gamma_down = gamma_down,
            gamma_01_down = gamma_01_down,
            gamma_up = gamma_up,
            Gamma_down = Gamma_down, 
            Gamma_up = Gamma_up,
            Gamma_phi = Gamma_phi,
            Gamma_down_ro = Gamma_down_ro,
            Gamma_up_ro = Gamma_up_ro,

            T_M = T_M,

            T1_a = T1_a,
            T2_a = T2_a,
            T_s = T_s,
        ))

        return self.full_para
