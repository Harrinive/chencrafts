import numpy as np
from collections import OrderedDict

def generate_sweep_lists(free_variable_dict, length_dict):
    """
    free_variable_dict should be a dictionary of list for range or a one-element 
    list for selected value
    """
    assert len(free_variable_dict) == len(length_dict)

    variable_list_dict = {}
    for key, length in length_dict.items():

        if length == 1:
            if len(free_variable_dict[key]) == 1:
                variable_list_dict[key] = free_variable_dict[key]
            else:
                assert len(free_variable_dict[key]) == 2
                min_range, max_range = free_variable_dict[key]
                variable_list_dict[key] = [(min_range + max_range) / 2]
        else:
            assert len(free_variable_dict[key]) == 2
            min_range, max_range = free_variable_dict[key]
            variable_list_dict[key] = np.linspace(min_range, max_range, length)

    return OrderedDict(variable_list_dict)

def generate_variable_meshgrids(variable_list_dict, slc_dict):
    # get the changing variable name
    var_name_to_mesh = []
    for key, slc in slc_dict.items():
        if slc == slice(None):
            var_name_to_mesh.append(key)

    variable_mesh_dict = {}
    
    X_list, Y_list = [variable_list_dict[key] for key in var_name_to_mesh]
    X_mesh, Y_mesh = np.meshgrid(X_list, Y_list, indexing='ij')
    variable_mesh_dict[var_name_to_mesh[0]] = X_mesh
    variable_mesh_dict[var_name_to_mesh[1]] = Y_mesh
    
    for key, slc in slc_dict.items():
        if slc != slice(None):
            variable_mesh_dict[key] = np.ones_like(X_mesh) * variable_list_dict[key][slc]

    return OrderedDict(variable_mesh_dict), var_name_to_mesh