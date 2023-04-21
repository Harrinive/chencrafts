import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray, Parameters

import numpy as np
import qutip as qt

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Dict, List, Tuple, Callable
import copy

class FlexibleSweep(

):
    """
    A flexible sweep class for supporting scq.ParameterSweep. 
    By defining fixed and swept parameters, ParameterSweep object can be autometically generated
    """
    def __init__(
        self,
        hilbertspace: HilbertSpace,
        para: Dict[str, float],
        sim_para: Dict[str, float | int],
        swept_para: Dict[str, List[float] | np.ndarray] = {},
        update_hilbertspace_by_keyword: Callable | None = None,
        subsys_update_info: Dict | None = None,
    ):
        """
        Parameters
        ----------
        update_hilbertspace_by_keyword:
            A function that takes the signature 
            function(HilbertSpace, <keyword 1>, <keyword 2>, ...) 
            It's not necessary to include all of the parameters in para and swept_para.
        subsys_update_info: 
            If a parameter name is not included in the dictionary, then it is assumed that 
            the parameter DO NOT affect the HilbertSpace at all.
        """
        # Parameters
        self.para = para
        self.sim_para = sim_para
        self.swept_para = dict([(key, np.array(val)) 
            for key, val in swept_para.items()])
        
        # Meshgrids and shape of the sweep
        if swept_para == {}:
            _parameters = None
            self._swept_para_meshgrids = {}
            self.dims = tuple()
        _parameters = Parameters(self.swept_para)
        self._swept_para_meshgrids = _parameters.meshgrid_by_name()
        self.dims = _parameters.counts

        # ParameterSweep
        self._complete_param_dict = self._get_complete_param_dict()
        self._update_hilbertspace_by_keyword = update_hilbertspace_by_keyword
        self.sweep = ParameterSweep(
            hilbertspace=hilbertspace,
            paramvals_by_name=self._complete_param_dict,
            update_hilbertspace=self._update_hilbertspace,
            evals_count=int(sim_para["evals_count"]),
            subsys_update_info=self._all_subsys_update_info(subsys_update_info),
            deepcopy=True,
            num_cpus=int(sim_para["num_cpus"]),
            override_update_func_check=True,
        )   

    def _get_complete_param_dict(self) -> Dict[str, np.ndarray]:
        param_by_name = {}
        for key, value in self.para.items():
            if key not in param_by_name:
                param_by_name[key] = np.array([value])

        param_by_name.update(self.swept_para)

        return param_by_name

    def _all_subsys_update_info(self, subsys_update_info) -> Dict:
        subsys_update_info = copy.deepcopy(subsys_update_info)
        for key in self.para.keys():
            if key not in subsys_update_info.keys():
                subsys_update_info[key] = None
        for key in self.swept_para.keys():
            if key not in subsys_update_info.keys():
                subsys_update_info[key] = None
        
        return subsys_update_info

    def _update_hilbertspace(self, ps: ParameterSweep, *args):
        """
        args are given by scqubits ordered as the keys of self._complete_param_list.
        Should select and give the correct parameters to the update_hilbertspace_by_keyword function.
        """

        param_dict = dict(zip(self._complete_param_dict.keys(), args))
        
        if self._update_hilbertspace_by_keyword is None:
            return 

        return self._update_hilbertspace_by_keyword(ps, **param_dict)
        
    @property
    def fixed_dim_slice(self) -> Tuple[slice]:
        slc_list = []
        for key in self.para.keys():
            if key in self.swept_para.keys():
                continue
            slc_list.append(slice(key, 0))

        return tuple(slc_list)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            # scq.ParameterSweep usually requires multiple slicing
            para_name = key[0]
            further_slice = key[1:]
            return self.sweep[para_name][further_slice][self.fixed_dim_slice]

        if key in self.sweep.keys():
            return self.sweep[key][self.fixed_dim_slice]
        
        elif key in self.swept_para.keys():
            return self._swept_para_meshgrids[key]
        
        elif key in self.para.keys():
            return NamedSlotsNdarray(
                np.ones(self.dims) * self.para[key],
                self.swept_para
            )
        
        else:
            raise KeyError(f"Key {key} is not found in the sweep.")

    def keys(self) -> List[str]:
        key_set = set(self.para.keys())
        key_set.update(self.swept_para.keys())
        key_set.update(self.sweep.keys())

        return list(key_set)
    
    def values(self) -> List[NamedSlotsNdarray]:
        return [self[key] for key in self.keys()]
        
    def items(self) -> List[Tuple[str, NamedSlotsNdarray]]:
        return [(key, self[key]) for key in self.keys()]
