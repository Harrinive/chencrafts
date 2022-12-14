from array import array
from locale import currency
import chencrafts

from typing import Callable, List, Union, Dict

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from scqubits.core.namedslots_array import NamedSlotsNdarray


class DimensionModify():
    """
    data[a(3), b(1), c(5)]
    -- drop idx --> 
    data[a(3), c(5)]
    -- perumute --> 
    data[c(5), a(3)]
    -- add idx --> 
    data[c(5), a(3), d(10)]
    """
    def __init__(
        self,
        current_shape_dict,
        target_shape_dict,
    ):
        current_shape_dict = current_shape_dict.copy()

        common_keys_in_target_order = []
        for key, val in target_shape_dict.items():
            if key in current_shape_dict:
                common_keys_in_target_order.append(key)
        
        # drop and check the dropped dim == 1
        for key, val in current_shape_dict.copy().items():
            if key not in common_keys_in_target_order:
                if val != 1:
                    raise ValueError(f"Init shape on this direction: {key} does not have"
                    " length 1 and should appear in the target shape.")
                else:
                    del current_shape_dict[key]
        self.shape_after_drop = np.array(list(current_shape_dict.values()))
        
        # permuete and check the common dims are the same
        unpermuted_keys = list(current_shape_dict.keys())
        self.permute_idx = []
        for key in common_keys_in_target_order:
            if current_shape_dict[key] != target_shape_dict[key]:
                raise ValueError(f"Init shape on this direction: {key} does not have the"
                " same shape as the target")
            self.permute_idx.append(unpermuted_keys.index(key))
        
        # add index
        self.new_axis_position_n_length = []
        for idx, (key, val) in enumerate(target_shape_dict.items()):
            if key not in common_keys_in_target_order:
                self.new_axis_position_n_length.append((idx, val))

    def __call__(self, data: np.ndarray):
        new_data = data.copy()
        # drop
        new_data = new_data.reshape(self.shape_after_drop)
        # permute
        new_data = np.transpose(new_data, self.permute_idx)
        # add idx
        new_shape = np.array(list(new_data.shape), dtype=int)
        for idx, length in self.new_axis_position_n_length:
            # insert a dimension to reshape data
            new_shape = np.insert(new_shape, idx, 1)    
            new_data = new_data.reshape(new_shape)
            new_data = np.repeat(new_data, length, axis=idx)
            new_shape[idx] = length
        
        return new_data


class NSArray(NamedSlotsNdarray):
    def __new__(
        cls, 
        input_array: np.ndarray | float, 
        values_by_name: Dict[str, np.ndarray] = None
    ) -> "NamedSlotsNdarray":
        if isinstance(input_array, float | int):
            return super().__new__(cls, np.array(input_array), {})
        elif values_by_name is None:
            raise ValueError("value_by_name shouldn't be None unless your input "
            "arra is actually a float number.")

        return super().__new__(cls, input_array, values_by_name)

    def __getitem__(self, index):
        if isinstance(index, dict):
            regular_index = []
            for key in self.param_info.keys():
                try:
                    regular_index.append(index[key])
                except KeyError:
                    regular_index.append(slice(None))
            return super().__getitem__(tuple(regular_index))

        return super().__getitem__(index)

    def reshape(self, *args, **kwargs):
        """
        Reshape breaks the structure of the naming method, return a normal ndarray
        """
        return np.array(super().reshape(*args, **kwargs))


def nd_interpolation(
    coord: List[np.ndarray],
    value: np.ndarray
) -> Callable:
    # detect nan in the value
    flattened_value = value.reshape(-1)
    val_not_nan = np.logical_not(np.isnan(flattened_value))

    # input 
    coord_size = [arr.size for arr in coord]
    if np.allclose(coord_size, value.size):
        # if the coords are already meshgrid
        coord_2_use = coord
    elif np.prod(coord_size) == value.size:
        coord_2_use = np.meshgrid(coord, indexing="ij")
    else:
        raise ValueError(f"Should input a coordinate list whose shapes' product"
        " equals to the value's size. Or just input a list of meshgrid of coordinates")

    coord_2_use = [arr.reshape(-1)[val_not_nan] for arr in coord_2_use]
    coord_2_use = np.transpose(coord_2_use)

    # get a linear interpolation using scipy
    interp = LinearNDInterpolator(
        coord_2_use,
        flattened_value[val_not_nan],
    )

    return interp


def merge_sort(arr: Union[List, np.ndarray], ascent: bool = True) -> np.ndarray:
    """
    Merge sort an array
    
    Parameters
    ----------
    arr:
        the array to be sorted
    ascent: 
        the order of the returned array

    Returns
    -------
        the sorted array
    """
    length = len(arr)
    array = np.array(arr)

    _merge_sort_kernel(array, length, ascent)

    return array

def _merge_sort_kernel(arr: Union[List, np.ndarray], length: int, ascent: bool) -> None:
    """
    Kernel function of chencrafts.merge_sort()
    """
    def swap(arr: np.ndarray, p_1: int = 0, p_2: int = 1):
        tmp = arr[p_1]
        arr[p_1] = arr[p_2]
        arr[p_2] = tmp

    def merge(arr: np.ndarray, l: int, l1: int, l2: int, ascent: bool):
        pointer = 0
        pointer_1 = 0
        pointer_2 = l1

        new_array = np.empty(l)

        while pointer_1 < l1 and pointer_2 < l:
            if (arr[pointer_1] < arr[pointer_2]) == ascent:
                new_array[pointer] = arr[pointer_1]
                pointer_1 += 1
            else:
                new_array[pointer] = arr[pointer_2]
                pointer_2 += 1
            pointer += 1

        if pointer_1 < l1:
            new_array[pointer:] = arr[pointer_1: l1]
        elif pointer_2 < l:
            new_array[pointer:] = arr[pointer_2:]
        
        arr[:] = new_array[:]

    if length == 1:
        return None
    if length == 2:
        if (arr[0] > arr[1]) == ascent:
            swap(arr)
        return None
    
    split_length_1 = int(length / 2)
    split_length_2 = length - split_length_1

    _merge_sort_kernel(arr[:split_length_1], split_length_1, ascent)
    _merge_sort_kernel(arr[split_length_1:], split_length_2, ascent)

    merge(arr, length, split_length_1, split_length_2, ascent)

    return None
    