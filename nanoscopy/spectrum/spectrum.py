from abc import abstractmethod
import numpy as np
import pandas as pd
from scipy import interpolate, integrate
from scipy.signal import savgol_filter
from typing import List, Tuple
from ..utilities import progbar

FILES_NOT_FOUND_ERROR = "No files found."
FILETYPE_ERROR = "SUPPORTED FILETYPE NOT FOUND. Only dat and 3ds are supported."


#  Convenience functions for calculations on spectra
def calculate_ndc(bias, current, dIdV) -> np.ndarray:
    """
    Calculates normalized differential conductance using bias, current, and dIdV
    """
    ndc = dIdV * (bias / current)

    return ndc


def integrate_current(dIdV) -> np.ndarray:
    """
    Calculates current by integrating the dIdV signal using the cumulative trapezoid rule.
    """
    # integrate dIdV
    current = integrate.cumulative_trapezoid(dIdV, initial=0)
    # find all the data points which lie in the bandgap
    gap_points = current[dIdV.index[dIdV == 0]]

    if len(gap_points) > 0:
        # re-scale current so the band gap has zero amplitude
        zero_point = gap_points[0]
        current = current - zero_point

    return current


def logscale(sts: np.ndarray) -> np.ndarray:
    """
    Applies logscaling to sts data.

    Args:
        sts (np.ndarray): numpy array of STS data

    Returns:
        Numpy array of logscaled STS data
    """
    offset = 0.1 if sts.min() == 0 else abs(sts.min())
    sts = sts + abs(sts.min()) + 0.5 * offset
    sts = np.log10(sts)

    return sts


def offset_correction(
    signal: np.ndarray = np.array([]), shift: bool = True
) -> np.ndarray:
    """
    Offset negative values the input array such that there are no negative values.

    Args:
        signal (np.ndarray): numpy array with signal data to be offset
        shift (bool): True will shift all values up such that the minimum is zero, False will set negative values to their 10% of their abs value

    Returns:
        numpy array with no negative values.
    """
    if shift:
        # shift values up so the min is set to zero
        signal = signal + abs(signal.min())
    else:
        # set negative values in dIdV to a small non-zero float
        signal[signal < 0] = 0.1 * np.abs(signal[signal < 0])

    return signal


def set_bandgap_zero(current: np.ndarray, dIdV: np.ndarray) -> np.ndarray:
    """
    Returns dIdV with the bandgap zeroed where the current is less than 1 pA.

    Args:
        current (np.ndarray): numpy array containing current data
        dIdV (np.ndarray): numpy array containing dIdV data

    Returns:
        numpy array with bandgap elements set to zero
    """
    lower = -1e-12
    upper = 1e-12
    # lower = -2e-13
    # upper = 2e-13
    curr_condition = (current > lower) & (current < upper)
    dIdV[curr_condition] = 0

    return dIdV


def mirrored_range_with_zero(maxval: int, inc: int) -> np.ndarray:
    """
    Creates a mirror symmetric numpy range which includes zero.

    Args:
        maxval (int): the maximum value of the range
        inc (int): the incriment used for the range

    Returns:
        numpy range from -maxval to maxval which includes zero
    """
    x = np.arange(inc, maxval, inc)
    if x[-1] != maxval:
        x = np.r_[x, maxval]
    return np.r_[-x[::-1], 0, x]


def range_with_zero(minval: int, maxval: int, inc: int) -> np.ndarray:
    """
    Creates a potentially asymmetric numpy range which includes zero.

    Args:
        minval (int): the minimum value of the range
        maxval (int): the maximum value of the range
        inc (int): the incriment used for the range

    Returns:
        numpy range from -maxval to maxval which includes zero
    """
    x_pos = np.arange(inc, maxval + inc, inc)
    x_neg = np.arange(minval, 0, inc)

    out = np.r_[x_neg, 0, x_pos]
    return out


def new_range(min=-3, max=3, step=0.0025, zero_centered=True):
    """
    Either creates a potentially asymmetric numpy range which includes zero
    or a numpy linspace which does not gaurentee the inclusion of zero.

    Args:
        min (int): the minimum value of the range/linspace
        max (int): the maximum value of the range/linspace
        step (int): the incriment used for the range/linspace
        zero_centered (bool): True will create a range including zero, False will create a linspace

    Returns:
        numpy range w/ zero or linspace from min to max
    """
    if zero_centered:
        new_bias = range_with_zero(min, max, step)
    else:
        n_points = int((max - min) / step)
        new_bias = np.linspace(min, max, n_points)

    return new_bias


def squared_error(array: np.ndarray) -> np.ndarray:
    """
    Computes the element-wise mean squared error of an input numpy array

    Args:
        array (np.ndarray): input numpy array for error calculation

    Returns:
        squared error of each element in the input as a numpy array
    """
    # get means for each row
    # means = [row.mean() for row in array]
    mean = array.mean()
    # print(array.shape, mean)

    # calculate squared errors
    # squared_error = [(row-mean)**2 for row, mean in zip(array, means)]
    squared_error = [(value - mean) ** 2 for value in array] / mean
    # print(squared_error)
    return squared_error


def interpolate_pad(X, Y, Xnew, pad):
    # interpolate curve
    f = interpolate.interp1d(X, Y)

    X_min, X_max = X.min(), X.max()
    Ynew = np.zeros_like(Xnew)

    for i, x in enumerate(Xnew):
        if x < X_min:
            Ynew[i] = -pad
        elif x > X_max:
            Ynew[i] = pad
        else:
            Ynew[i] = f(x)

    return Ynew


def interp(X, Y, Xnew, pad):
    # interpolate curve
    f = interpolate.interp1d(X, Y)

    X_min, X_max = X.min(), X.max()
    Ynew = np.zeros_like(Xnew)

    for i, x in enumerate(Xnew):
        Ynew[i] = f(x)

    return Ynew


def pad(X, Y, Xnew, pad):
    # pad curve
    f = interpolate.interp1d(X, Y)

    X_min, X_max = X.min(), X.max()
    Ynew = np.zeros_like(Xnew)

    for i, x in enumerate(Xnew):
        if x < X_min:
            Ynew[i] = -pad
        elif x > X_max:
            Ynew[i] = pad
        else:
            Ynew[i] = f(x)

    return Ynew


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalizes array to [0,1]

    Args:
        data (np.ndarray): numpy array to be normalized

    Returns:
        numpy array with values noramlized to [0,1]
    """
    data = (data - data.min()) / (data.max() - data.min())
    return data


def interpolate_STS_dfs(df_list, bias_label, lix_label, bias_new):
    sts = []

    for j, df in enumerate(df_list):
        bias = df[bias_label]
        # bias = np.unique(np.round(bias, 3))
        dIdV = df[lix_label]
        ydIdV = interpolate_pad(bias, dIdV, bias_new, pad=0)
        sts.append(ydIdV)
        # gaps.append(find_band_gap(bias, dIdV))

    sts = np.array(sts)
    return sts


class Spectrum(object):
    """
    Parent class for containing basic data on spectra.
    Intended to be inheritied from by more specific spectra.

    Args:
        dataframe (pd.DataFrame): Spectrum data stored in a dataframe
        metadata (pd.DataFrame): Spectrum metadata extracted from specturm file
        filepath (str): system file path string of the spectum file
    """

    def __init__(self, dataframe=None, metadata=None, filepath=""):
        self.dataframe = dataframe
        self.metadata = metadata
        self.filepath = filepath

    @abstractmethod
    def clean(self):
        pass


class STS(Spectrum):
    """
    Class for containing with an individual Scanning Tunneling Spectrum.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        # initialize default values
        self.b_label = "Bias (V)"
        self.i_label = "Current (A)"
        self.dIdV_label = "dIdV"
        self.ndc_label = "NDC"

        self.current = None
        self.dIdV = None
        self.ndc = None
        self.new_bias = None
        self.new_dIdV = None
        self.label = None

        # bias is assumed to be the first data column
        # rename dataframe columns
        self.bias = self.dataframe.iloc[:, 0]
        self.dataframe = self.dataframe.rename(
            columns={
                self.dataframe.columns[0]: self.b_label,
            }
        )

        if self.metadata is not None:
            # check Nanonis metadata for channels
            channels_df = self.metadata.loc["Bias Spectroscopy>Channels"]
            channels = channels_df.values[0].split(";")

            for i, channel in enumerate(channels):
                if "Current" in channel:
                    self.current = self.dataframe.iloc[:, i + 1]
                    self.dataframe = self.dataframe.rename(
                        columns={
                            self.dataframe.columns[i + 1]: self.i_label,
                        }
                    )
                elif "LIX" in channel:
                    self.dIdV = self.dataframe.iloc[:, i + 1]
                    self.dataframe = self.dataframe.rename(
                        columns={
                            self.dataframe.columns[i + 1]: self.dIdV_label,
                        }
                    )
        else:
            self.dIdV = self.dataframe.iloc[:, 1]
            self.dataframe = self.dataframe.rename(
                columns={
                    self.dataframe.columns[1]: self.dIdV_label,
                }
            )

        # get size of matrix after skipping the bias column
        self.n, self.m = self.dataframe.iloc[:, 1::].shape

        if self.m == 1:
            # only the bias and dIdV singals are present, so calculate current
            self._integrate_current()

        # all signals required to calculate ndc are available
        self._calculate_ndc()

    def _calculate_ndc(self) -> None:
        """
        Updates normalized differential conductance using bias, current, and dIdV
        """
        # calculate and update current from dIdV
        if self.current is None:
            self._integrate_current()

        # calculate ndc using bias, current, and dIdV
        ndc = calculate_ndc(self.bias, self.current, self.dIdV)

        # update signals
        self.ndc = ndc
        self.dataframe[self.ndc_label] = ndc

    def _integrate_current(self) -> None:
        """
        Updates current by integrating the dIdV signal using the cumulative trapezoid rule.
        """
        current = integrate_current(self.dIdV)

        # find all the data points which lie in the bandgap
        gap_points = current[self.dIdV.index[self.dIdV == 0]]

        if len(gap_points) > 0:
            zero_point = gap_points[0]

            # re-scale current so the band gap has zero amplitude
            current = current - zero_point

        # update current signal and dataframe store
        self.current = current
        self.dataframe[self.i_label] = current

    def _mean_signal(self, col_idx):
        """
        Returns average signal from repeated STS measurements as a Pandas Series.
        select all columns from col_idx to end of columns
        skipping 1 each time, then average all columns together.
        """
        # select all current columns (avg, fwd, bwd)
        # then average all columns together
        # m is divided by 2 since half of the columns are current
        mean_signal = self.dataframe.iloc[:, col_idx::2].cumsum(axis=1).iloc[:, -1] / (
            self.m / 2
        )

        return mean_signal

    def _offset_correction(self, shift=False):
        self.dIdV = offset_correction(self.dIdV, shift)

    def _zero_gap(self):
        self.dIdV = set_bandgap_zero(self.current, self.dIdV)

    def get_signal(self, signal_label):
        return self.dataframe[signal_label]

    def set_bias_label(self, label):
        self.b_label = label

    def set_dIdV_label(self, label):
        self.dIdV_label = label

    def set_mean_signals(self):
        bias = self.bias
        current = self._mean_signal(1)
        dIdV = self._mean_signal(2)
        ndc = dIdV * (bias / current)

        mean_df = pd.concat(
            [
                bias.rename(self.b_label),
                current.rename(self.i_label),
                dIdV.rename(self.dIdV_label),
                ndc.rename(self.ndc_label),
                pd.DataFrame([self.filepath], columns=["filepath"]),
            ],
            axis=1,
        )

        self.dataframe = mean_df
        self.current = current
        self.dIdV = dIdV
        self.ndc = ndc

    def preprocess(
        self, n: int, norm: bool = True, zero_gap: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        current = np.array(self.current)
        dIdV = np.array(self.dIdV)
        ndc = np.array(self.ndc)

        if zero_gap:
            dIdV = set_bandgap_zero(current, dIdV)

        if (dIdV < 0).any():
            dIdV = offset_correction(dIdV, shift=False)

        if norm:
            dIdV = normalize(dIdV)
            current = normalize(current)
            ndc = normalize(ndc)

        fill_value = 0
        f = interpolate.interp1d(
            self.bias, dIdV, bounds_error=False, fill_value=fill_value
        )
        g = interpolate.interp1d(
            self.bias, current, bounds_error=False, fill_value=fill_value
        )
        h = interpolate.interp1d(
            self.bias, ndc, bounds_error=False, fill_value=fill_value
        )

        # new_bias = new_range(self.bias.min(), self.bias.max(), step, zero_centered=False)
        new_bias = np.linspace(self.bias.min(), self.bias.max(), n)
        new_bias = np.round(new_bias, 3)
        # print(new_bias.shape)

        new_dIdV = f(new_bias)
        new_current = g(new_bias)
        new_ndc = h(new_bias)

        return new_bias, new_dIdV, new_current, new_ndc

    def preprocess_dIdV(
        self, new_bias: np.ndarray, norm: bool = True, zero_gap: bool = True
    ) -> np.ndarray:
        """
        Processes the dIdV signal of this spectrum using the new bias by
        selecting dIdV values strictly within the new bias range then,
        zeroing the bandgap, offsetting negative values,
        and optionally normalizing to [0,1],
        followed by interpolating the signals according to the new bias range.

        Args:
            new_bias (np.ndarray): new bias range to be used for preprocessing
            norm (bool): True will normalize all signals to [0,1], while False will not.
            zero_gap (bool): True will zero the band gap.
        Returns:
            Interpoalted dIdV in range specified by new bias signal
        """
        bias = np.array(self.bias)
        dIdV = np.array(self.dIdV)

        # create a mask to filter the signals based on the new bias
        target_min, target_max = new_bias.min(), new_bias.max()
        min_mask = np.round(self.dataframe[self.b_label], 2) >= target_min
        max_mask = np.round(self.dataframe[self.b_label], 2) <= target_max
        mask = min_mask & max_mask

        # apply the mask to the dataframe to select signal values within in the new bias range
        sel_current = self.dataframe[mask][self.i_label].reset_index(drop=True)
        sel_dIdV = self.dataframe[mask][self.dIdV_label].reset_index(drop=True)
        sel_bias = np.round(
            self.dataframe[mask][self.b_label].reset_index(drop=True), 2
        )

        # round endpoints to ensure selected range matched target end points
        sel_bias.iat[0] = np.round(sel_bias.iat[0], 1)
        sel_bias.iat[-1] = np.round(sel_bias.iat[-1], 1)

        if zero_gap:
            sel_dIdV = set_bandgap_zero(sel_current, sel_dIdV)

        if (sel_dIdV < 0).any():
            sel_dIdV = offset_correction(sel_dIdV, shift=False)

        if norm:
            sel_dIdV = normalize(sel_dIdV)

        # interpolate the dIdV signal according to the new bias signal (range and spacing)
        try:
            f = interpolate.interp1d(sel_bias, sel_dIdV)
            new_dIdV = f(new_bias)
            self.new_bias = new_bias
            self.new_dIdV = new_dIdV
            return new_dIdV

        except ValueError as e:
            print(self.filepath[-17:])
            print("raw bias:", bias.min(), bias.max())
            print("new bias:", new_bias.min(), new_bias.max())
            print("selected bias:", sel_bias.min(), sel_bias.max())
            raise e

    def preprocess_new_bias(self, new_bias, norm=True, extrapolate=False):
        bias = np.array(self.bias)
        dIdV = np.array(self.dIdV)

        target_min, target_max = new_bias.min(), new_bias.max()
        min_mask = np.round(self.dataframe[self.b_label], 3) >= target_min
        max_mask = np.round(self.dataframe[self.b_label], 3) <= target_max
        mask = min_mask & max_mask

        dIdV = self.dataframe[mask][self.dIdV_label]
        bias = self.dataframe[mask][self.b_label]
        current = self.dataframe[mask][self.i_label]
        ndc = self.dataframe[mask][self.ndc_label]

        if norm:
            dIdV = normalize(dIdV)
            current = normalize(current)
            ndc = normalize(ndc)

        fill_value = "extrapolate" if extrapolate else 0
        f = interpolate.interp1d(bias, dIdV, bounds_error=False, fill_value=fill_value)
        g = interpolate.interp1d(
            bias, current, bounds_error=False, fill_value=fill_value
        )
        h = interpolate.interp1d(
            bias, current, bounds_error=False, fill_value=fill_value
        )

        new_dIdV = f(new_bias)
        new_current = g(new_bias)
        new_ndc = h(new_bias)

        self.new_bias = new_bias
        self.new_dIdV = new_dIdV

        return new_dIdV, new_current, new_ndc
