import scqubits as scq

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep, StateLabel
from scqubits.core.namedslots_array import NamedSlotsNdarray, Parameters
from scqubits.core.storage import SpectrumData

from chencrafts.cqed.scq_helper import meshgrid_by_name
from chencrafts.cqed.mode_assignment import branch_analysis

import numpy as np
from numpy.typing import NDArray
from warnings import warn

from typing import Dict, List, Tuple, Callable, Any, Literal, Optional, overload

class FlexibleSweep():   
    """
    FlexibleSweep is a wrapper of scq.ParameterSweep. 
        - It allows for flexible parameter sweeping by defining fixed and swept 
            parameters. 
        - It will take `update_hilbertspace_by_keyword` as an input of the sweep
    """
    def __init__(
        self,
        hilbertspace: HilbertSpace,
        para: Dict[str, float] = {"dummy": 0.0},    # a dummy parameter
        swept_para: Dict[str, List[float] | NDArray[Any]] = {}, 
        update_hilbertspace_by_keyword: Callable | None = None,
        evals_count: Optional[int] = None,
        num_cpus: int = 1,
        subsys_update_info: Dict[str, Any] = {},
        default_update_info: List | Literal["all"] | None = "all",
        **kwargs,
    ):
        """
        FlexibleSweep is a wrapper of scq.ParameterSweep. 
            - It allows for flexible parameter sweeping by defining fixed and swept 
            parameters. When a parameter appears in both `para` and `swept_para`,
            the value in `swept_para` will be given priority.
            - It will take `update_hilbertspace_by_keyword` as an input of the sweep

        Parameters
        ----------
        hilbertspace: HilbertSpace
            scq.HilbertSpace object
        para: Dict[str, float]
            A dictionary of default parameters when not swept over. It's not necessary 
            neither required to include the parameters in swept_para. By default, it's set 
            to {"x": 0.0}, which is a dummy parameter and allows doing single value 
            parameter sweep with current hilberspace parameters.
        swept_para: Dict[str, List[float] | np.ndarray]
            A dictionary of parameters to be swept. The values are lists or numpy arrays.
            By default, it's set to {"x": [0.0]}, which is a dummy parameter and allows
            doing single value parameter sweep with current hilberspace parameters.
        update_hilbertspace_by_keyword:
            A function that takes the signature 
            function(sweep, <keyword 1>, <keyword 2>, ...) 
            It's not necessary to include all of the parameters in para and swept_para.
        evals_count: int
            Number of eigenvalues to be calculated.
        num_cpus: int
            Number of cpus to be used.
        subsys_update_info: 
            Specify whether a parameter change will update a subsystem in the HilbertSpace.
            Should be a dictionary of the form {<parameter name>: <subsys update info>}.
            If <subsys update info> is None, then no update will be triggered.
            If <subsys update info> is a list, then the list should contain the 
            the subsystems to be updated.
            If a parameter name is not included in the dictionary, then it is assumed that 
            the parameter's <subsys update info> is the `subsys_update_default_info`.
        default_update_info: str | List | None
            If a parameter is not included in the `subsys_update_info` dictionary, then
            the parameter's <subsys update info> is the `subsys_update_default_info`. Can 
            also be "all", which means all subsystems will be updated.
        """
        # Parameters
        self.para = para
        self.swept_para = dict([(key, np.array(val)) 
            for key, val in swept_para.items()])
        self._check_valid_var_name()
        
        # Parameter setup
        self._complete_param_dict = self._get_complete_param_dict()
        _parameters = self._order_swept_para()
        self._swept_para_meshgrids = meshgrid_by_name(_parameters)
        self.dims = _parameters.counts
        self.hilbertspace = hilbertspace
        self._subsys_update_info = self._all_subsys_update_info(subsys_update_info, default_update_info)
        self._update_hilbertspace_by_keyword = update_hilbertspace_by_keyword
        evals_count = evals_count if evals_count is not None else self.hilbertspace.dimension

        # ParameterSweep
        self.sweep = ParameterSweep(
            hilbertspace=self.hilbertspace,
            paramvals_by_name=self._complete_param_dict,
            update_hilbertspace=self._build_update_hilbertspace_func(),
            evals_count=evals_count,
            subsys_update_info=self._subsys_update_info,
            deepcopy=False,
            num_cpus=num_cpus,
            autorun=True,
        )   

    def _check_valid_var_name(self):
        """
        Later, the parameters will be used as variable names in the update function.
        Therefore, the parameter names should be a valid variable names.
        """
        for key in self._all_param_names():
            if not key.isidentifier():
                raise ValueError(f"Invalid variable name: {key}")

    def _all_param_names(self) -> List[str]:
        key_set = set(self.para.keys())
        key_set.update(self.swept_para.keys())
        return list(key_set)
    
    def _order_swept_para(self) -> Parameters:
        # Meshgrids and shape of the sweep
        # if self.swept_para == {}:
        #     raise ValueError("No swept parameters are specified.")
        
        # order the swept parameters by the order of _complete_param_dict
        ordered_swept_para = {}
        for key, val in self._complete_param_dict.items():
            if key in self.swept_para.keys():
                ordered_swept_para[key] = self.swept_para[key]

        # # order the swept parameters by the order of para
        # ordered_swept_para = {}
        # for key, val in self.para.items():
        #     if key in self.swept_para.keys():
        #         ordered_swept_para[key] = self.swept_para[key]

        # put the rest of the swept parameters at the end
        ordered_swept_para.update(self.swept_para)
        
        parameters = Parameters(ordered_swept_para)
        return parameters

    def _get_complete_param_dict(self) -> Dict[str, np.ndarray]:
        param_by_name = {}
        for key, value in self.para.items():
            if key not in param_by_name:
                param_by_name[key] = np.array([value])

        param_by_name.update(self.swept_para)

        return param_by_name

    def _all_subsys_update_info(self, subsys_update_info, default) -> Dict:
        if default == "all":
            default = self.hilbertspace.subsystem_list

        # make a shallow copy of the update info
        subsys_update_info_copy = {}
        for key, val in subsys_update_info.items():
            subsys_update_info_copy[key] = val

        # fill in the default update info
        for key in self._all_param_names():
            if key not in subsys_update_info_copy.keys():
                subsys_update_info_copy[key] = default
        
        return subsys_update_info_copy

    def _build_update_hilbertspace_func(self):
        # Get the argument list for the function
        arg_name_list = list(self._complete_param_dict.keys())
        arg_name_str = ', '.join(arg_name_list)
        func_str = f"""
def update(ps, {arg_name_str}):
    arg_list = [{arg_name_str}]
    param_dict = dict(zip(arg_name_list, arg_list))

    if update_by_keyword is None:
        return
    
    return update_by_keyword(ps, **param_dict)"""
        local_vars = {
            'arg_name_list': arg_name_list,
            'update_by_keyword': self._update_hilbertspace_by_keyword,
        }
        # local_vars = {}
        exec(func_str, local_vars)

        return local_vars['update']

    @property
    def fixed_dim_slice(self) -> Tuple[slice]:
        """
        A slice that can slice out the fixed parameters. 
        """
        slc_list = []
        for key in self.para.keys():
            if key in self.swept_para.keys() and key != "dummy":
                continue
            slc_list.append(slice(key, 0))
    
        return tuple(slc_list)
    
    def full_slice(
        self, 
        slice_tuple: Tuple[slice | int, ...] | int | slice | None
    ) -> Tuple[slice, ...]:        
        base_slice = list(self.fixed_dim_slice)
        
        if slice_tuple is None: 
            return tuple(base_slice)
        
        
        if isinstance(slice_tuple, (int, slice)):
            slice_tuple = (slice_tuple,)
        
        if all(isinstance(slc, int) for slc in slice_tuple):
            # combine the key and the index
            swept_para_names = self._order_swept_para().names
            name_based_slc = [
                slice(key, idx) 
                for key, idx in zip(swept_para_names, slice_tuple)
            ]
            base_slice.extend(name_based_slc)
                    
        elif all(isinstance(slc, slice) for slc in slice_tuple):
            for slc in slice_tuple:
                assert isinstance(slc.start, str), "Invalid slice"
                assert isinstance(slc.stop, int | float), "Invalid slice"
                assert slc.step is None, "Invalid slice"
                base_slice.append(slc)
                
        else:
            raise ValueError("Invalid slice. Should be either a tuple of ints or"
                             " name based slices.")

        return tuple(base_slice)

    def __getitem__(self, key):
        """
        Since we swept all para and swpet_para, and user expect to get 
        a sweep result with axes being only the swept_para, we need to
        slice the result to remove the fixed parameters.
        """
        if isinstance(key, tuple):
            # scq.ParameterSweep usually requires multiple slicing
            para_name = key[0]
            further_slice = key[1:]
            return self.sweep[para_name][further_slice][self.fixed_dim_slice]

        if key in self.sweep.keys():
            arr = self.sweep[key]
            try: 
                arr = arr[self.fixed_dim_slice]
            except KeyError:
                # slice failed because it's an wrapped array 
                pass
            return arr
        
        elif key in self.swept_para.keys():
            return self._swept_para_meshgrids[key]
        
        elif key in self.para.keys():
            return NamedSlotsNdarray(
                np.ones(self.dims) * self.para[key],
                self.swept_para
            )
        
        else:
            raise KeyError(f"Key {key} is not found in the sweep.")

    def keys(self, sort: bool = True) -> List[str]:
        key_set = set(self._all_param_names())
        key_set.update(self.sweep.keys())

        if sort:
            return sorted(list(key_set))

        return list(key_set)
    
    def values(self) -> List[NamedSlotsNdarray]:
        return [self[key] for key in self.keys()]
        
    def items(self) -> List[Tuple[str, NamedSlotsNdarray]]:
        return [(key, self[key]) for key in self.keys()]

    def full_dict(self) -> Dict[str, NamedSlotsNdarray]:
        return dict(self.items())

    def transitions(
        self, *args, **kwargs
    ) -> SpectrumData | Tuple[
        List[Tuple[StateLabel, StateLabel]], 
        List[NamedSlotsNdarray]
    ]:
        """
        A wrapper of scq.ParameterSweep.transitions. It returns the 
        sliced data if the result is not a SpectrumData.
        """
        result = self.sweep.transitions(*args, **kwargs)

        if not isinstance(result, tuple):   
            warn("Result is a spectrum data and FlexibleSweep can't handle "
            "it for now, returning the result as is.")
            return result
        
        labels, data = result
        sliced_data = [data[self.fixed_dim_slice] for data in data]

        return labels, sliced_data

    def dressed_indices(self, bare_indices: Tuple[int, ...]) -> np.ndarray:
        """ 
        A wrapper of scq.ParameterSweep.dressed_indices. It helps to to ravel
        the bare indices.
        """
        raveled_idx = np.ravel_multi_index(bare_indices, self.hilbertspace.subsystem_dims)
        return self["dressed_indices"][..., raveled_idx]
    
    def branch_analysis(
        self, 
        mode: Literal["DF", "EF"] = "EF",
        mode_priority: Optional[List[int]] = None,
        truncate: int | None = None,
        check_all_prev_states: bool = False,
    ):
        branch_indices = branch_analysis(
            self.sweep,
            mode = mode,
            mode_priority = mode_priority, 
            transpose = False,
            truncate = truncate,
            check_all_prev_states = check_all_prev_states,
        )
        self.sweep.store_data(
            dressed_indices = branch_indices.reshape(
                *(self.sweep.parameters.counts + (-1,))
            )
        )
        (
            self.sweep._data["lamb"],
            self.sweep._data["chi"],
            self.sweep._data["kerr"],
            self.sweep._data["chi_prime"],      # only in Danyang's branch of scqubits
        ) = self.sweep._dispersive_coefficients()