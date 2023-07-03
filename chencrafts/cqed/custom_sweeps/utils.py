import inspect
from scqubits.core.param_sweep import ParameterSweep

def fill_in_kwargs_during_custom_sweep(
    sweep: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    func, external_kwargs,
    ignore_kwargs: list[str] = [],
    ignore_pos: list[int] = [],
):
    """
    Inside the custom sweep function A, I'll sometimes need to call other functions B. 
    Function B may have keyword arguments that should be specified in external keyword, 
    swept parameters or the parameters are already calculated in the sweep.

    Priorities
    ----------
    This function will fill in the keyword arguments for function B. Arguments will be 
    filled in by the following priority (from high to low):
    - swept parameters
    - sweep[<name>]
    - external_kwargs[<name>] (an argument of this function)

    Ignore
    ------
    if ignored, then the argument will not be filled in for now. There are two ways to
    ignore an argument:
    - ignore_kwargs: a list of keyword argument names that will be ignored
    - ignore_pos: a list of positional argument index that will be ignored
    """
    overall_kwargs = {}

    parameters = inspect.signature(func).parameters
    for idx, arg in enumerate(parameters.keys()):
        if arg in ignore_kwargs or idx in ignore_pos:
            # ignored, will be taken care of later 
            continue

        elif parameters[arg].kind == inspect.Parameter.VAR_POSITIONAL:
            # it's a *args, ignore it
            continue

        elif parameters[arg].kind == inspect.Parameter.VAR_KEYWORD:
            # it's a **kwargs, ignore it
            # ignore the rest of the arguments
            break

        elif arg in sweep.parameters.names:
            overall_kwargs[arg] = paramvals_tuple[sweep.parameters.index_by_name[arg]]

        elif arg in sweep.keys():
            overall_kwargs[arg] = sweep[arg][paramindex_tuple]

        elif arg in external_kwargs.keys():
            overall_kwargs[arg] = external_kwargs[arg]

        else:
            raise TypeError(f"{func} missing a required keyword argument: {arg}")
        
    return overall_kwargs