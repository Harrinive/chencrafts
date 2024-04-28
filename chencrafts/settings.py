# Function checking whether code is run from a jupyter notebook or inside ipython
def executed_in_ipython():
    try:  # inside ipython, the function get_ipython is always in globals()
        shell = get_ipython().__class__.__name__
        if shell in ["ZMQInteractiveShell", "TerminalInteractiveShell"]:
            return True  # Jupyter notebook or qtconsole of IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# a switch for displaying of progress bar; default: show only in ipython
if executed_in_ipython():
    PROGRESSBAR_DISABLED = False
    IN_IPYTHON = True
else:
    PROGRESSBAR_DISABLED = True
    IN_IPYTHON = False


# qutip version related settings
import qutip as qt
QUTIP_VERSION = tuple(map(int, qt.__version__.split(".")))