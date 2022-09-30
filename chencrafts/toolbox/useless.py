import chencrafts

from typing import List, Union
import numpy as np

from numpy import ndarray

def greet():
    """
    Greet from the package!
    """
    print(f"Hello, Danyang! It's chencrafts version {chencrafts.__version__}.")

def merge_sort(arr: Union[List, ndarray], ascent: bool = True) -> ndarray:
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

def _merge_sort_kernel(arr: Union[List, ndarray], length: int, ascent: bool) -> None:
    """
    Kernel function of chencrafts.merge_sort()
    """
    def swap(arr: ndarray, p_1: int = 0, p_2: int = 1):
        tmp = arr[p_1]
        arr[p_1] = arr[p_2]
        arr[p_2] = tmp

    def merge(arr: ndarray, l: int, l1: int, l2: int, ascent: bool):
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
    