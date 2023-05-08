import inspect
from scqubits.core.param_sweep import ParameterSweep

def fill_in_kwargs_during_custom_sweep(
    sweep: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    func, external_kwargs,
    ignore_kwargs: list[str] = [],
):
    """
    Inside the custom sweep function A, I'll sometimes need to call other functions B. 
    Function B may have keyword arguments that should be specified in external keyword, 
    swept parameters or the parameters are already calculated in the sweep.
    This function will fill in the keyword arguments for function B.

    If not ignored, parameters in the external_kwargs will be given priority over the 
    swept parameters and sweep[<kwarg>] calculated.
    """
    overall_kwargs = {}
    for arg in inspect.signature(func).parameters.keys():
        if arg in ignore_kwargs:
            # will be taken care of later 
            continue

        elif arg in external_kwargs.keys():
            overall_kwargs[arg] = external_kwargs[arg]

        elif arg in sweep.parameters.names:
            overall_kwargs[arg] = paramvals_tuple[sweep.parameters.index_by_name[arg]]

        elif arg in sweep.keys():
            overall_kwargs[arg] = sweep[arg][paramindex_tuple]

        else:
            raise TypeError(f"{func} missing a required keyword argument: {arg}")
        
    return overall_kwargs