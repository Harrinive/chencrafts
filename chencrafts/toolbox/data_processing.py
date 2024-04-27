
from typing import Callable, List, Union, Dict, Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from scqubits.core.namedslots_array import NamedSlotsNdarray

from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


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
        values_by_name: Dict[str, range | np.ndarray | None] = {}
    ) -> "NamedSlotsNdarray":
        if isinstance(input_array, float | int):
            return super().__new__(cls, np.array(input_array), {})

        elif isinstance(input_array, np.ndarray) and input_array.shape == tuple() and values_by_name == {}:
            return super().__new__(cls, input_array, {})
        
        elif isinstance(input_array, NamedSlotsNdarray) and values_by_name == {}:
            return super().__new__(cls, input_array, input_array.param_info)
        
        elif isinstance(input_array, NSArray) and values_by_name == {}:
            return input_array.copy()
        
        elif values_by_name == {}:
            raise ValueError("value_by_name shouldn't be empty unless your input "
            "array is a float number.")
        
        elif isinstance(input_array, list | np.ndarray | range):
            # set value to be range(dim) when it is None
            for idx, (key, val) in enumerate(values_by_name.items()):
                if val is None:
                    values_by_name[key] = range(input_array.shape[idx])

            input_array = np.array(input_array)
            data_shape = np.array(input_array.shape)
            name_shape = np.array([len(val) for val in values_by_name.values()])
            if len(data_shape) != len(name_shape):
                raise ValueError(f"Dimension of the input_array ({len(data_shape)}) doesn't match with the "
                    f"length of named slots ({len(name_shape)})")
            if (data_shape != name_shape).any():
                raise ValueError(f"Shape of the input_array {data_shape} doesn't match with the "
                    f"shape indicated by the named slots {name_shape}")
            
            ndarray_values_by_name = dict(zip(
                values_by_name.keys(),
                [np.array(val) for val in values_by_name.values()]
            ))          # without this value-based slicing won't work

            return super().__new__(cls, input_array, ndarray_values_by_name)
    
        else:
            raise ValueError(f"Your input data is incompatible.")

    def __getitem__(self, index):
        if isinstance(index, dict):
            regular_index = []
            for key in self.param_info.keys():
                try:
                    idx = index[key]
                except KeyError:
                    idx = slice(None)

                if isinstance(idx, np.ndarray):
                    if idx.shape == tuple():
                        idx = float(idx)

                regular_index.append(idx)

            return super().__getitem__(tuple(regular_index))

        return super().__getitem__(index)

    def reshape(self, *args, **kwargs):
        """
        Reshape breaks the structure of the naming method, return a normal ndarray
        """
        return np.array(super().reshape(*args, **kwargs))
    
    def transpose(self, axes = None):
        transposed_data = np.array(super().transpose(axes))

        if axes is None:
            dim = len(self.shape)
            axes = np.linspace(0, dim-1, dim, dtype=int)[::-1]

        transposed_param_info = {}
        key_list = list(self.param_info.keys())
        for dim_idx in axes:
            key = key_list[dim_idx]
            transposed_param_info[key] = self.param_info[key]

        return NSArray(transposed_data, transposed_param_info)


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

def scatter_to_mesh(
    x_data, y_data, z_data, 
    x_remeshed=None, y_remeshed=None
):
    """
    Convert scattered data points to a mesh using linear interpolation.

    Parameters
    ----------
    - x_data: 1-D array-like or iterable. The x-coordinates of the scattered data points.
    - y_data: 1-D array-like or iterable. The y-coordinates of the scattered data points.
    - z_data: 1-D array-like or iterable. The z-values of the scattered data points.
    - x_remeshed: 1-D array-like or iterable, optional. The x-coordinates of the remeshed data points.
    - y_remeshed: 1-D array-like or iterable, optional. The y-coordinates of the remeshed data points.

    Returns
    -------
    - interp: LinearNDInterpolator object. The linear interpolator object.
    - data: 1-D array or None. The interpolated z-values at the remeshed data points. If x_remeshed and y_remeshed are not provided, data will be None.

    Notes
    -----
    - x_data, y_data, and z_data should all be in the same shape.
    - If x_remeshed and y_remeshed are provided, the function will return both the interpolator object and the interpolated z-values at the remeshed data points.
    - If x_remeshed and y_remeshed are not provided, the function will only return the interpolator object.

    """
    x_ravel = np.array(x_data).reshape(-1)
    y_ravel = np.array(y_data).reshape(-1)
    z_ravel = np.array(z_data).reshape(-1)

    val_not_nan = list(np.logical_not(np.isnan(z_ravel)))

    input_xy = np.transpose([
        x_ravel[val_not_nan],
        y_ravel[val_not_nan]
    ])
    interp = LinearNDInterpolator(
        input_xy,
        z_ravel[val_not_nan],
    )

    if x_remeshed is not None and y_remeshed is not None:
        data = interp(x_remeshed, y_remeshed)
        return interp, data
    else:
        return interp


def find_envelope(
    data_array: np.ndarray | List[float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the envelope of a curve given the data points.

    Parameters
    ----------
    data_array: 
        1-D array-like or iterable. The data points of the curve.
    
    Returns
    -------
    top_envelope:
        1-D array. The top envelope of the curve.
    bottom_envelope:
        1-D array. The bottom envelope of the curve.
    """
    data_array = np.array(data_array)

    # Find the local maxima and minima using relative extrema
    maxima_indices = argrelextrema(data_array, np.greater)[0]
    minima_indices = argrelextrema(data_array, np.less)[0]

    # Check if any maxima or minima were found
    if len(maxima_indices) == 0 or len(minima_indices) == 0:
        raise ValueError("No envelope found for the given data")

    # Create interpolation functions with extrapolation using scipy
    top_envelope_func = interp1d(
        maxima_indices, data_array[maxima_indices], 
        kind="cubic", fill_value="extrapolate"
    )
    bottom_envelope_func = interp1d(
        minima_indices, data_array[minima_indices], 
        kind="cubic", fill_value="extrapolate"
    )

    # Compute the envelopes
    top_envelope = top_envelope_func(np.arange(len(data_array)))
    bottom_envelope = bottom_envelope_func(np.arange(len(data_array)))

    return top_envelope, bottom_envelope

def decay_rate(
    t_array: np.ndarray | List[float],
    data_array: np.ndarray | List[float],
    extract_envelope: bool = True,
    maxfev: int = 1000,
):
    """
    Find the decay rate of a curve by fitting the data curve. 

    Parameters
    ----------
    t_array: 1-D array-like or iterable. 
        The time points of the curve.
    data_array: 1-D array-like or iterable.
        The data points of the curve.
    extract_envelope: bool (optional).
        Whether to extract the envelope of the curve before fitting. 
        Default is True.
    maxfev: int (optional).
        The maximum number of function evaluations before the fit is terminated.
        Default is 1000.

    Returns
    -------
    Coefficient of function f(t) = a * exp(-b * t) + c
    
    a: float. 
        The amplitude of the decay rate.
    b: float. 
        The decay rate.
    c: float.
        The offset of the decay rate.

    """
    t_array = np.array(t_array)
    data_array = np.array(data_array)
    
    # find the envelope of the curve using numpy
    if extract_envelope:
        top_envelope, bottom_envelope = find_envelope(data_array)
        data_array = top_envelope - bottom_envelope

    # find the decay rate by fitting the envelope
    fit_func = lambda x, a, b, c: a * np.exp(-b * x) + c

    # fit the top envelope
    popt, pcov = curve_fit(
        fit_func, t_array, data_array, 
        p0 = [data_array.max(), 1/(t_array[1] - t_array[0]), data_array.min()],
        maxfev=maxfev
    )
    a, b, c = popt

    if pcov[1, 1] > 0.1:
        print("Warning: The decay rate fit may not be accurate.")

    # return the sum of the two decay rates
    return a, b, c