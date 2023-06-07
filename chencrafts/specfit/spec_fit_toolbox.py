import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from typing import Dict, List, Tuple

# ##############################################################################
# extract data
def remove_repeated_legend(ax=None):
    """remove repeated legend"""
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

def plot_peaks(extracted_peaks: Dict, existed_scatter_list=[], ax=None, cmap='tab10'):
    """
    plot peaks from a dictionary of peaks with user defined cmaps
    
    Parameters
    ----------
    extracted_peaks : Dict
        a dictionary of peaks with the format of 
        {
            (0, 1): [[para, freq], [para, freq], ...],
            (0, 2): [[para, freq], [para, freq], ...],
            ...
        }
    existed_scatter_list : List
        a list of scatter plots that are plotted in the current axes by 
        `plt.scatter`. Should pass an empty list when users need to delete a series of
        scatter plots.
    ax : matplotlib.axes.Axes
    cmap : str
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=150)
        
    # plot and record the scatter plots
    scatter_series_num = len(extracted_peaks)
    scatter_list = [
        ax.scatter(
            *zip(*value), 
            label=key, 
            color=plt.cm.get_cmap(cmap)(idx / scatter_series_num), 
            zorder=1, 
            s=3
        ) for idx, (key, value) in enumerate(extracted_peaks.items()) if value != []
    ]
    existed_scatter_list.extend(scatter_list)

    # remove repeated legend
    remove_repeated_legend(ax)

    # update the plot
    plt.draw()

    return ax

def remove_all_scattered_dots(scat_list: List):
    """remove all scatter plots plotted in the current axes by plt.scatter"""    
    for scat in scat_list:
        scat.remove()

    # pop all of the items in the list
    scat_list.clear()

    plt.draw()

def _find_xy_index_w_value(x_list, y_list, x, y):
    """find the index of x and y in the x_list and y_list"""
    x_idx = np.argmin(np.abs(x_list - x))
    y_idx = np.argmin(np.abs(y_list - y))
    return x_idx, y_idx

def _slice_w_center_and_range(x_list, y_list, center_tuple, index_range_tuple):
    """slice the data with center and range"""
    # translate the peak to index
    x, y = center_tuple
    x_idx, y_idx = _find_xy_index_w_value(x_list, y_list, x, y)

    # translate range to left and right index
    x_range, y_range = index_range_tuple
    x_left_idx = x_idx - int(x_range / 2)
    x_right_idx = x_left_idx + x_range
    y_left_idx = y_idx - int(y_range / 2)
    y_right_idx = y_left_idx + y_range

    # make sure the index is within the range
    x_left_idx = max(0, x_left_idx)
    x_right_idx = min(len(x_list), x_right_idx)
    y_left_idx = max(0, y_left_idx)
    y_right_idx = min(len(y_list), y_right_idx)

    return x_left_idx, x_right_idx, y_left_idx, y_right_idx

def _polish_by_min_max(data) -> Tuple:
    """
    one choice of peak polishing method

    locate the minimum and maximum value in the data
    determine which looks more like a peak
    """

    # find the local minimum/maximum
    min_loc = np.argmin(data)
    min_loc = np.unravel_index(min_loc, data.shape)
    min_val = data[min_loc]

    max_loc = np.argmax(data)
    max_loc = np.unravel_index(max_loc, data.shape)
    max_val = data[max_loc]

    # determin which one is the peak by comparing the value with the average value of the range
    avg_val = np.mean(data)
    if (avg_val - min_val) > (max_val - avg_val):
        return min_loc
    else:
        return max_loc

def _polish_by_fit(data) -> Tuple:
    """
    one choice of peak polishing method

    average the data along the paramter axis and then fit the data with a 1D Lorentzian 
    function
    """
    data_1D = np.mean(data, axis=0)
    freq_length = len(data_1D)
    idx_list = np.arange(freq_length)

    # fit the data with a 1D Lorentzian function
    lorentzian = lambda idx, mid_idx, gamma, amp, bias: amp * gamma**2 / ((idx - mid_idx)**2 + gamma**2) + bias

    # guess
    mid_idx_guess = int(freq_length / 2)
    gamma_guess = freq_length / 5
    amp_guess = 0       # maybe positive or negative, so set it to 0
    bias_guess = np.mean(data_1D)

    popt, pcov = curve_fit(
        lorentzian, 
        idx_list, data_1D, 
        p0=[mid_idx_guess, gamma_guess, amp_guess, bias_guess],
    )

    # check convergence
    if pcov[0, 0] > np.max([(freq_length/100)**2, 1]):
        print(f"Warning: polish data by fitting may not converge. "
            f"The variance of estimating the center is {pcov[0, 0]}."
        )
        return int(data.shape[0] / 2), np.round(popt[0]).astype(int)
    else:
        return int(data.shape[0] / 2), np.round(popt[0]).astype(int)


def polish_peaks(x_list, y_list, data, peak_tuple, index_range_tuple) -> Tuple[int, int]:
    """
    polish a given peak by looking for a local minimum/maximum near the gien peak
    
    Parameters
    ----------
    x_list : List
        the x_list of the data
    y_list : List
        the y_list of the data
    data : List
        the data
    peak_tuple : Tuple
        the peak (x, y) indexes that will be polished
    index_range_tuple : Tuple
        a new peak will be found within the range of the index

    Returns
    -------
        (x, y), new peak
    """
    # translate range to left and right index
    x_left_idx, x_right_idx, y_left_idx, y_right_idx = _slice_w_center_and_range(
        x_list, y_list, peak_tuple, index_range_tuple
    )
    slc = slice(x_left_idx, x_right_idx), slice(y_left_idx, y_right_idx)

    # find the peaks
    peak_idx = _polish_by_fit(data[slc])

    return x_list[peak_idx[0] + x_left_idx], y_list[peak_idx[1] + y_left_idx]

def _plot_range(x_list, y_list, index_range_tuple, ax=None):
    """
    plot a square on the top right corner of the plot, indicating the range of the index_range_tuple
    """
    if ax is None:
        ax = plt.gca()

    x_left_idx = int(len(x_list) / 50) + 1
    x_right_idx = x_left_idx + index_range_tuple[0]
    y_right_idx = len(y_list) - int(len(y_list) / 50) - 1
    y_left_idx = y_right_idx - index_range_tuple[1]
    ax.plot(
        [x_list[x_left_idx], x_list[x_left_idx], x_list[x_right_idx], x_list[x_right_idx], x_list[x_left_idx]],
        [y_list[y_left_idx], y_list[y_right_idx], y_list[y_right_idx], y_list[y_left_idx], y_list[y_left_idx]],
        color="black", linestyle="--", linewidth=0.5, zorder=2
    )

def get_peaks(
    extracted_peaks, 
    transition_levels = (-1, -1),
    ax = None,
    polish=True, polish_index_range_tuple=(3, 10),
    x_list=None, y_list=None, data=None
):
    """
    extract peaks from a exist plots
    
    Parameters
    ----------
    extracted_peaks : Dict
        a dictionary of peaks with the format of
        {
            (0, 1): [[para, freq], [para, freq], ...],
            (0, 2): [[para, freq], [para, freq], ...],
            ...
        }
    transition_levels : Tuple
        the transition level that the user is trying to extract, in the format
        of (n, m). By default, it is (-1, -1), which means the user don't know
        the transition level.
    ax : matplotlib.axes.Axes
    polish : bool
        whether to polish the extracted peaks by looking for a local minimum/maximum
        near the click point. By default, it is True. 
    polish_index_range_tuple : Tuple
        If the polish is True, then the user can specify the range of the index
        that will be used to polish the extracted peaks. By default, it is (3, 10).
    x_list : List
        If the polish is True, then the user need to pass the x_list of the data.
    y_list : List
        If the polish is True, then the user need to pass the y_list of the data.
    data : List
        If the polish is True, then the user need to pass the data.
    """
    if polish:
        assert x_list is not None and y_list is not None and data is not None,        "x_list, y_list, and data must be passed if polish is True"

    if ax is None:
        ax = plt.gca()

    # plot the peaks and record the scatter plots in order for future removal
    scat_list = []
    plot_peaks(extracted_peaks, scat_list, ax=ax, cmap="rainbow")
    if polish:
        _plot_range(x_list, y_list, polish_index_range_tuple, ax=ax)

    # store the extracted peaks in a list, format [(x1, y1), (x2, y2), ...)]
    coords = []

    # define a function that will respond to the click event,
    # then record the click position (or polished peak) and plot the peak
    def onclick(event):
        nonlocal coords

        ix, iy = event.xdata, event.ydata

        if polish:
            ix, iy = polish_peaks(x_list, y_list, data, (ix, iy), polish_index_range_tuple)

        # record the peak
        coords.append((ix, iy))

        # plot the peak
        remove_all_scattered_dots(scat_list)
        plot_peaks(extracted_peaks, scat_list, ax=ax, cmap="rainbow")
        plot_peaks({"Current picks": coords}, scat_list, ax=ax, cmap="gray")
        
    fig = plt.gcf()
    cid = fig.canvas.mpl_connect("button_release_event", onclick)

    plt.show(block=True)

    fig.canvas.mpl_disconnect(cid)

    # save the extracted data
    if coords != []:
        try:
            extracted_peaks[transition_levels].extend(coords)
        except KeyError:
            extracted_peaks[transition_levels] = coords
        
def remove_peaks(extracted_peaks, remove_index_range_tuple=(np.inf, np.inf), ax=None):
    """remove peaks from a exist plots"""
    if ax is None:
        ax = plt.gca()

    # plot the peaks and record the scatter plots in order for future removal
    scat_list = []
    plot_peaks(extracted_peaks, scat_list, ax=ax, cmap="rainbow")

    # check if the extracted_peaks is empty
    empty = True
    for value in extracted_peaks.values():
        if value != []:
            empty = False
    if empty:
        return

    # define a function that will respond to the click event,
    # then remove the peak that is closest to the click and lies within the range
    def onclick(event):
        nonlocal extracted_peaks

        ix, iy = event.xdata, event.ydata

        # find a single peak that is closest to the click and lies within the range
        min_dist = np.inf
        min_key = None
        min_idx = None
        for key, value in extracted_peaks.items():
            for idx, (x, y) in enumerate(value):
                displacement = np.abs([ix - x, iy - y])
                if displacement[0] > remove_index_range_tuple[0] or displacement[1] > remove_index_range_tuple[1]:
                    continue

                dist = np.sqrt(np.sum(displacement ** 2))
                if dist < min_dist:
                    # update the min
                    min_dist = dist
                    min_key = key
                    min_idx = idx

        # remove the peak
        if min_key is not None:
            extracted_peaks[min_key].pop(min_idx)

        # plot the peak
        remove_all_scattered_dots(scat_list)
        plot_peaks(extracted_peaks, scat_list, ax=ax, cmap="rainbow")

    # connect the function with the click event
    fig = plt.gcf()
    cid = fig.canvas.mpl_connect("button_release_event", onclick)

    plt.show(block=True)

    # disconnect the function with the click event
    fig.canvas.mpl_disconnect(cid)
