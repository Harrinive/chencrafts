import numpy as np
from chencrafts.toolbox.useless import merge_sort, merge_sort_kernel

ARRAY_LENGTH = 100
example_array = np.random.rand(ARRAY_LENGTH)

class TestFuncs:
    def test_merge_sort(self):
        result_asc = merge_sort(example_array, ascent=True)
        result_dsc = merge_sort(example_array, ascent=False)

        for i in range(ARRAY_LENGTH - 1):
            assert result_asc[i] <= result_asc[i+1]
            assert result_dsc[i] >= result_dsc[i+1]

    def test_merge_sort_kernel(self):
        two_element_arr = np.array([2, 1])

        merge_sort_kernel(two_element_arr, 2, True)
        assert two_element_arr[0] == 1

        merge_sort_kernel(two_element_arr, 2, False)
        assert two_element_arr[0] == 2

