__all__ = [
    'datetime_dir',
    'save_variable_list_dict',
    'load_variable_list_dict',
    'save_variable_dict',
    'load_variable_dict',
    'dill_dump',
    'dill_load',
    'h5_dump',
    'h5_load',
]

import time
import os

import dill
import numpy as np
import pandas as pd
import h5py

from typing import Any, Dict, Literal

def datetime_dir(
    save_dir="./",
    dir_suffix=None,
):
    """
    Initialize a directory with the current datetime. 

    Parameters & Examples
    ---------------------
    save_dir : str
        The directory to save the data, default to be "./". Say the current
        datetime is 2021-01-31 12:34, then the directory will be
        "save_dir/Jan/31_12-34/".
    dir_suffix : str
        The suffix of the directory, default to be None. Say the current
        datetime is 2021-01-31 12:34, then the directory will be
        "save_dir/Jan/31_12-34_dir_suffix/".

    Returns
    -------
    current_date_dir : str
    """
    save_dir = os.path.normpath(save_dir)
    
    current_time = time.localtime()
    current_month_dir = save_dir + time.strftime("/%h/", current_time)
    current_date_dir = current_month_dir + \
        time.strftime("%d_%H-%M", current_time)

    if dir_suffix != "" and dir_suffix is not None:
        current_date_dir = current_date_dir + "_" + dir_suffix + "/"
    else:
        current_date_dir = current_date_dir + "/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(current_month_dir):
        os.mkdir(current_month_dir)
    if not os.path.exists(current_date_dir):
        os.mkdir(current_date_dir)

    print(f"Current datetime directory: {current_date_dir}")
    return current_date_dir

def save_variable_dict(file_name, variable_dict: Dict[str, float]):
    """
    Save a dictionary contains name-value pairs to a csv file.
    """
    new_dict = dict([(key, [val]) for key, val in variable_dict.items()])
    pd.DataFrame.from_dict(
        new_dict,
        orient="columns",
    ).to_csv(file_name)

def load_variable_dict(file_name) -> Dict[str, float]:
    """
    Load a dictionary contains name-value pairs from a csv file. The file should be 
    saved by save_variable_dict.
    """
    list_dict = pd.read_csv(
        file_name, 
        index_col=0,
        header=0
    ).to_dict(orient='list')
    new_dict = dict([(key, val[0]) for key, val in list_dict.items()])
    return new_dict

def save_variable_list_dict(
    file_name, 
    variable_list_dict: Dict[str, np.ndarray], 
    orient: Literal['columns', 'index'] = 'columns',
) -> None:
    """
    Save a dictionary contains name-value_list pairs to a csv file.

    orient = 'index' SHOULD be used when variable list are not equal in length
    """
    pd.DataFrame.from_dict(
        variable_list_dict,
        orient=orient,
    ).to_csv(file_name)

def load_variable_list_dict(
    file_name, 
    throw_nan = True, 
    orient: Literal['columns', 'index'] = 'columns'
) -> Dict[str, np.ndarray]:
    """
    Load a dictionary contains name-value_list pairs from a csv file. The file should be
    saved by save_variable_list_dict.

    throw_nan : bool
        If True, remove nan in the list. It's useful when the list is not equal in length.

    orient = 'index' should be used when variable list are not equal in length
    """
    if orient == 'index':
        variable_list_dict = pd.read_csv(
            file_name, index_col=0, header=0).transpose().to_dict(orient='list')
    elif orient == 'columns':
        variable_list_dict = pd.read_csv(
            file_name, index_col=0, header=0).to_dict(orient='list')
    else:
        raise ValueError("only recognize 'index' or 'columns' for orient")

    if not throw_nan:
        return dict([(key, np.array(val)) for key, val in variable_list_dict.items()])

    for key, val in variable_list_dict.items():
        new_val = np.array(val)
        new_val = new_val[~np.isnan(val)]
        variable_list_dict[key] = new_val
    return variable_list_dict

def dill_dump(obj: Any, filename: str) -> None:
    """Dump a python object to a file using dill."""
    filename = os.path.normpath(filename)
    file = open(filename, "wb")
    dill.dump(obj, file)
    file.close()

def dill_load(filename: str) -> Any:
    """Load a python object from a file using dill."""
    filename = os.path.normpath(filename)
    file = open(filename, "rb")
    obj = dill.load(file)
    file.close()

    return obj

# Save data to HDF5 file
def h5_dump(
    data_dict: Dict[str, Any],
    file_name: str, 
):
    """
    Dump a dictionary to a HDF5 file.

    Parameters
    ----------
    filename : str
        The filename of the HDF5 file.
    data_dict : Dict[str, Any]
    """
    with h5py.File(file_name, 'w') as f:
        # Create groups and datasets
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Create a group for nested data
                group = f.create_group(key)
                for subkey, subvalue in value.items():
                    group.create_dataset(subkey, data=subvalue)
            else:
                # Create a dataset for direct data
                f.create_dataset(key, data=value)
        
# Load data from HDF5 file
def h5_load(
    file_name: str,
) -> Dict[str, Any]:
    """
    Load a dictionary from a HDF5 file.

    Parameters
    ----------
    file_name : str
        The filename of the HDF5 file.

    Returns
    -------
    data_dict : Dict[str, Any]
        The dictionary loaded from the HDF5 file.
    """
    data_dict = {}
    with h5py.File(file_name, 'r') as f:
        # Helper function to read all items
        def extract_data(name, obj):
            if isinstance(obj, h5py.Dataset):
                data_dict[name] = obj[()]
            
        # Visit all items in the file
        f.visititems(extract_data)
        
        # Print file structure
        print("File structure:")
        def print_info(name, obj):
            print(f"{name}, shape: {obj.shape}" if isinstance(obj, h5py.Dataset) else name)
        f.visititems(print_info)
    
    return data_dict