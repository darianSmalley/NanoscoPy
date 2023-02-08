from abc import abstractmethod
import torch
import numpy as np
import pandas as pd
from scipy import interpolate, integrate
from scipy.signal import savgol_filter

def mirrored_range_with_zero(maxval, inc):
    x = np.arange(inc, maxval, inc)
    if x[-1] != maxval:
        x = np.r_[x, maxval]
    return np.r_[-x[::-1], 0, x]

def range_with_zero(minval, maxval, inc):
    x_pos = np.arange(inc, maxval+inc, inc)
    x_neg = np.arange(minval, 0, inc)

    # if x_pos[-1] != maxval:
    #     x_pos = np.r_[x_pos, maxval]

    out = np.r_[x_neg, 0, x_pos]
    return out

def new_range(min=-3, max=3, step=0.0025, zero_centered = True):
    if zero_centered:
        new_bias = range_with_zero(min, max, step)
    else:
        n_points = int((max - min)/step)
        new_bias = np.linspace(min, max, n_points)

    return new_bias

def squared_error(array):
    # get means for each row
    # means = [row.mean() for row in array]
    mean = array.mean()
    # print(array.shape, mean)

    # calculate squared errors
    # squared_error = [(row-mean)**2 for row, mean in zip(array, means)]
    squared_error = [(value-mean)**2 for value in array] / mean
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

def normalize(data):
    # Normalize to (0, 1)
    data = (data - data.min()) / (data.max() - data.min())
    return data

def interpolate_STS_dfs(df_list, bias_label, lix_label, bias_new):
    sts = []

    for j, df in enumerate(df_list):
        bias = df[bias_label]
        # bias = np.unique(np.round(bias, 3))   
        dIdV = df[lix_label]
        ydIdV = interpolate_pad(bias, dIdV, bias_new, pad = 0)
        sts.append(ydIdV)
        # gaps.append(find_band_gap(bias, dIdV))

    sts = np.array(sts)
    return sts


class Spectrum(object):
    def __init__(self, dataframe =  None, metadata = None, filepath = ''):
        self.dataframe = dataframe
        self.metadata = metadata
        self.filepath = filepath
    
    @abstractmethod
    def clean(self):
        pass

class STS(Spectrum):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

        # initialize default values
        self.b_label = 'Bias (V)'
        self.i_label = 'Current (A)'
        self.dIdV_label = 'dIdV'
        self.ndc_label = 'NDC'

        self.current = None
        self.dIdV = None
        self.ndc = None
        self.new_bias = None
        self.new_dIdV = None
        self.label = None

        # bias is assumed to be the first data column
        self.dataframe = self.dataframe.rename(columns={self.dataframe.columns[0]: self.b_label})
        self.bias = self.dataframe.iloc[:, 0]
        # get size of matrix after skipping the bias column
        self.n, self.m = self.dataframe.iloc[:, 1::].shape

        if self.m > 1:
            self.set_mean_signals()
        else: 
            self.dataframe = self.dataframe.rename(columns={self.dataframe.columns[1]: self.dIdV_label})
            self.dIdV = self.dataframe.iloc[:, 1]
            self.integrate_current()
            self.calculate_ndc()

        if (self.dIdV < 0).any(): 
            self.offset_correction()

    def _mean_signal(self, col_idx):
        """
        Returns average signal from repeated STS measurements as a Pandas Series.
        select all columns from col_idx to end of columns
        skipping 1 each time, then average all columns together.
        """
        # select all current columns (avg, fwd, bwd)
        # then average all columns together
        # m is divided by 2 since half of the columns are current
        mean_signal = self.dataframe.iloc[:, col_idx::2].cumsum(axis=1).iloc[:, -1] / (self.m / 2)

        return mean_signal

    def _calculate_ndc(self):
        self.integrate_current()
        self.ndc = self.dIdV * (self.bias/self.current)

    def _offset_correction(self):
        # shift values up so the min is set to zero
        self.dIdV = self.dIdV + abs(self.dIdV.min())

    def _integrate_current(self):
        current = integrate.cumulative_trapezoid(self.dIdV, initial=0)
        # find all the data points which lie in the bandgap
        gap_points = current[self.dIdV.index[self.dIdV == 0]]

        if len(gap_points) > 0:
            zero_point = gap_points[0]
            # re-scale current so the band gap has zero amplitude
            self.current = current - zero_point
        else:
            self.current = current

    def get_signal(self, signal_label):
        return self.dataframe[signal_label]

    def set_bias_label(self, label):
        self.b_label = label

    def set_dIdV_label(self, label):
        self.dIdV_label = label

    def set_mean_signals(self):   
        bias = self.bias      
        current = self.mean_signal(1)
        dIdV = self.mean_signal(2)
        ndc = dIdV * (bias/current)

        mean_df = pd.concat([bias.rename(self.b_label), 
                            current.rename(self.i_label), 
                            dIdV.rename(self.dIdV_label), 
                            ndc.rename(self.ndc_label),
                            pd.DataFrame([self.filepath], columns=['filepath'])], axis=1)

        self.dataframe = mean_df
        self.current = current
        self.dIdV = dIdV
        self.ndc = ndc

    def preprocess(self, n, norm=True):
        current = np.array(self.current)
        dIdV = np.array(self.dIdV)
        ndc = np.array(self.ndc)

        if norm: 
            dIdV = normalize(dIdV)
            current = normalize(current)
            ndc = normalize(ndc)

        fill_value = 0
        f = interpolate.interp1d(self.bias, dIdV, bounds_error=False, fill_value=fill_value)
        g = interpolate.interp1d(self.bias, current, bounds_error=False, fill_value=fill_value)
        h = interpolate.interp1d(self.bias, ndc, bounds_error=False, fill_value=fill_value)

        # new_bias = new_range(self.bias.min(), self.bias.max(), step, zero_centered=False)
        new_bias = np.linspace(self.bias.min(), self.bias.max(), n)
        new_bias = np.round(new_bias, 3)
        # print(new_bias.shape)

        new_dIdV = f(new_bias)
        new_current = g(new_bias)
        new_ndc = h(new_bias)

        return new_bias, new_dIdV, new_current, new_ndc
        
    def preprocess_dIdV(self, new_bias, norm = True, extrapolate = False, zero_gap = False):
        bias = np.array(self.bias)
        dIdV = np.array(self.dIdV)

        target_min, target_max = new_bias.min(), new_bias.max()
        min_mask = np.round(self.dataframe[self.b_label], 4) >= target_min
        max_mask = np.round(self.dataframe[self.b_label], 4) <= target_max
        # min_mask = self.dataframe[self.b_label] >= target_min
        # max_mask = self.dataframe[self.b_label] <= target_max
        mask = min_mask & max_mask

        sel_current = self.dataframe[mask][self.i_label].reset_index(drop=True)
        sel_dIdV = self.dataframe[mask][self.dIdV_label].reset_index(drop=True)
        sel_bias = self.dataframe[mask][self.b_label].reset_index(drop=True)

        # round endpoints to ensure selected range matched target end points
        sel_bias.iat[0] = np.round(sel_bias.iat[0])
        sel_bias.iat[-1] = np.round(sel_bias.iat[-1])

        if zero_gap:
            curr_condition = (sel_current > -1e-12) & (sel_current < 1e-12)
            # dIdV_condition = (sel_dIdV > -1e-12) & (sel_dIdV < 1e-12)
            sel_dIdV[curr_condition] = 0

        if norm: 
            sel_dIdV = normalize(sel_dIdV)

        fill_value = 'extrapolate' if extrapolate else 0

        try:
            # f = interpolate.interp1d(sel_bias, sel_dIdV, bounds_error=False, fill_value=fill_value)
            f = interpolate.interp1d(sel_bias, sel_dIdV)
            new_dIdV = f(new_bias)
        except ValueError as e:
            print(self.filepath)
            print(bias.min(), bias.max())
            print(sel_bias.min(), sel_bias.max())
            raise e
        
        self.new_bias = new_bias
        self.new_dIdV = new_dIdV

        return new_dIdV

    def preprocess_new_bias(self, new_bias, norm = True, extrapolate = False):
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

        fill_value = 'extrapolate' if extrapolate else 0
        f = interpolate.interp1d(bias, dIdV, bounds_error=False, fill_value=fill_value)
        g = interpolate.interp1d(bias, current, bounds_error=False, fill_value=fill_value)
        h = interpolate.interp1d(bias, current, bounds_error=False, fill_value=fill_value)

        new_dIdV = f(new_bias)
        new_current = g(new_bias)
        new_ndc = h(new_bias)
        
        self.new_bias = new_bias
        self.new_dIdV = new_dIdV

        return new_dIdV, new_current, new_ndc
