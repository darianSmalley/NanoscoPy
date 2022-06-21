import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from patsy import dmatrix
from scipy.optimize import minimize, curve_fit
import itertools
from matplotlib.ticker import PercentFormatter
import math

# test function
def test_function(data, a, b, c):
    x = data[0]
    y = data[1]
    return a * (x**b) * (y**c)

def test_function2(data, a, b, c, d, e):
    x = data[0]
    y = data[1]
    return a + b * x + c * y + d*x*y + e

def nonliner_function(data, a, b, c, d, e, f, g, h, k):
    x = data[0]
    y = data[1]
    return a + b * x + c * y + d*x*y + f*(x**2) + h*(y**2) + g*(x*y)**2 + e

def fit_data(fn, X, Y, Z):
    # get fit parameters from scipy curve fit
    parameters, covariance = curve_fit(fn, [X, Y], Z)

    # create surface function model
    # setup data points for calculating surface model
    model_x_data = np.linspace(min(X), max(X), 50)
    model_y_data = np.linspace(min(Y), max(Y), 50)
    # create coordinate arrays for vectorized evaluations
    X_fit, Y_fit = np.meshgrid(model_x_data, model_y_data)
    # calculate Z coordinate array
    Z_fit = fn(np.array([X_fit, Y_fit]), *parameters)

    return X_fit, Y_fit, Z_fit

def calc_main_effect(data, factor, response):
        high = data[factor].max()
        low = data[factor].min()

        means = {}
        for level in [high, low]:
            factor_level_mask = data[factor] == level
            level_response_mean = data[factor_level_mask][response].mean()
            means[level] = level_response_mean

        return means[high] - means[low]

def calc_interaction_effect(data, factor1, factor2, response):
    high1 = data[factor1].max()
    low1 = data[factor1].min()
    high2 = data[factor2].max()
    low2 = data[factor2].min()

    means = {}
    for level2 in [high2, low2]:
        factor2_level_mask = data[factor2] == level2
        factor2_level_responses = data[factor2_level_mask][response]

        factor2_level_factor1 = data[factor2_level_mask][factor1]
        conditions = [(factor2_level_factor1 == high1),(factor2_level_factor1 == low1)]
        choices = [1, -1]
        factor_signs = np.select(conditions, choices)
        factor2_responses_factor1_signs = factor2_level_responses * factor_signs
        factor1_factor2_responses_mean = factor2_responses_factor1_signs.mean()
        means[level2] = factor1_factor2_responses_mean

    return means[high2] - means[low2]    

def calc_effects(data, response_label, factor_labels):
    main_effects = [calc_main_effect(data, factor_label, response_label) for factor_label in factor_labels]
    factor_pairs = itertools.combinations(factor_labels, 2)
    interaction_effects = []

    for factor_label1, factor_label2 in factor_pairs:
            interaction_effect = calc_interaction_effect(data, factor_label1, factor_label2, response_label)
            interaction_effects.append(interaction_effect)        
    
    effects = main_effects + interaction_effects
    return effects

def _plot_response_surfaces(data, response_label, factor_labels):
        cmap = "cool"
        cmap = "viridis"
        label_pad = 10
        axis_font_size = 12
        tick_font_size = 14

        n = len(factor_labels)
        fig_scale = 4
        fig, axs = plt.subplots(n, n, figsize=(fig_scale*n,fig_scale*n), constrained_layout=True, subplot_kw={'projection': '3d'})
        plt.suptitle(response_label + " Response Surfaces", fontsize=22)

        # build a rectangle in axes coords
        left, width = -0.33, 1.66
        bottom, height = -0.2, 1.4
        right = left + width
        top = bottom + height

        Z = data[response_label]
        C = data[response_label]
        ln_C = np.log(C)
        S = np.full((len(Z),), 100)

        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                factor_A_label = factor_labels[i]
                factor_B_label = factor_labels[j]

                X = data[factor_A_label]
                Y = data[factor_B_label]
                fit = fit_data(test_function2, X, Y, Z)
                ax.plot_surface(*fit, alpha=0.5)
                ax.scatter(X, Y, Z, s=S, c=ln_C, label = C, cmap=cmap, alpha=1.0)
                # handles1, labels1 = scatter1.legend_elements(prop="colors")
                # legend1 = ax.legend(handles1, labels1, loc="lower right", title=c_label)
                # ax.set_title("Maximum CNTF Height Response", fontsize=18, pad=20)
                ax.view_init(15, 60)
                ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
                ax.set_xlabel(factor_A_label, fontsize=axis_font_size, labelpad=label_pad)
                ax.set_ylabel(factor_B_label, fontsize=axis_font_size, labelpad=label_pad)
                ax.set_zlabel(response_label, fontsize=axis_font_size)
                ax.invert_xaxis()

                # handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
                # legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
                # for i in range(len(X)): #plot each point + it's index as text above
                #     ax.text(X[i]+0.5, Y[i]+0.5, Z[i], f'Flow: {C[i]}', size=16, zorder=1, color='k', weight='bold') 

        plt.show()

def linear(x, m, b):
    return m*x + b

def _plot_effect_grid(data, response_label, factor_labels):
    fontsize = 16
    n = len(factor_labels)
    fig_scale = 4
    fig, axs = plt.subplots(n, n, figsize=(fig_scale*n,fig_scale*n))
    y = data[response_label]
    C = data[response_label]

    # build a rectangle in axes coords
    left, width = -0.33, 1.66
    bottom, height = -0.2, 1.4
    right = left + width
    top = bottom + height

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            factor = factor_labels[i]

            if i == j:
                # main effects
                x = data[factor]

                # linear fit
                param, param_cov = curve_fit(linear, x, y)
                # ans stores the new y-data according to
                # the coefficients given by curve-fit() function
                ans = param[0]*x + param[1]
                # print(param[0], param[1])

                # z = np.polyfit(x, y1, 1)
                # p = np.poly1d(z)
                # print(z)

                scatter = ax.scatter(x, y, s=100, c=C)
                ax.plot(x, ans, linestyle='dashed', color ='black', label='linear fit')
                # ax.text(1.0, 0.2, f'Intercept = {param[0]:.2}, Slope = {param[1]:.2}', fontsize = 11)

                # ax.set_ylim(ymin=0, ymax=200)
                ax.set_xlabel(factor)
                ax.set_ylabel(response_label)
                ax.set_title(f'{factor}')
                ax.legend(loc='best')
                # fig.colorbar(scatter, label=response_2_label)

                if i == 0:
                    ax.text(0.5*(left+right), top, factor,
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes,
                            fontsize=fontsize, 
                            fontweight='bold')
                    ax.text(left, 0.5*(bottom+top), factor,
                            horizontalalignment='center',
                            verticalalignment='center',
                            rotation=90,
                            transform=ax.transAxes,
                            fontsize=fontsize, 
                            fontweight='bold')

            else:
                # interaction effects
                factor1 = factor
                factor2 = factor_labels[j]

                high1 = data[factor1].max()
                low1 = data[factor1].min()
                high2 = data[factor2].max()
                low2 = data[factor2].min()

                maskHighAHighB = (data[factor1] == high1) & (data[factor2] == high2)
                maskHighALowB = (data[factor1] == high1) & (data[factor2] == low2)
                maskLowAHighB = (data[factor1] == low1) & (data[factor2] == high2)
                maskLowALowB = (data[factor1] == low1) & (data[factor2] == low2)

                y1 = data[maskLowAHighB][response_label].mean() 
                y2 = data[maskLowALowB][response_label].mean()
                y3 = data[maskHighAHighB][response_label].mean()
                y4 = data[maskHighALowB][response_label].mean()

                ax.plot([low1, high1], [y1,y3], label=f'+ {factor2}')
                ax.plot([low1, high1], [y2,y4], label=f'- {factor2}')
                ax.set_xlabel(f'{factor1}')
                ax.set_ylabel(response_label)
                ax.set_title(f'{factor1}:{factor2}')
                ax.legend(loc='best')

                if i == 0:
                    ax.text(0.5*(left+right), top, factor2,
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes,
                            fontsize=fontsize, 
                            fontweight='bold')
                if j == 0:
                    ax.text(left, 0.5*(bottom+top), factor,
                            horizontalalignment='center',
                            verticalalignment='center',
                            rotation=90,
                            transform=ax.transAxes,
                            fontsize=fontsize, 
                            fontweight='bold')

    fig.tight_layout()
    plt.show()

def _Pareto_plot(effects, factor_labels, xlabel, ylabel):
    colors = []
    signs = []
    for effect in effects:
        if effect < 0:
            colors.append('gray')
            signs.append(-1)
        else:
            colors.append('royalblue')
            signs.append(1)

    effects_df = pd.DataFrame({
                    'factors': factor_labels,
                    'effects': np.abs(effects),
                    'signs': signs,
                    'colors': colors
                })

    effects_df_sorted = effects_df.sort_values('effects', ascending=False)
    x = effects_df_sorted['factors'].values
    y = effects_df_sorted['effects'].values
    c = effects_df_sorted['colors'].values
    s = effects_df_sorted['signs'].values
    weights = y / y.sum()
    cumsum = weights.cumsum()    

    fig, ax1 = plt.subplots()
    legend_labels = ['Positive Sign','Negative Sign']
    handles = [plt.Rectangle((0,0), 1,1, color='royalblue'), plt.Rectangle((0,0), 1,1, color='gray')]
    plt.legend(handles, legend_labels, loc='center right')

    ax1.bar(x, y, color = c)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = ax1.twinx()
    ax2.plot(x, cumsum, '-ro', alpha=0.5)
    ax2.set_ylabel('', color='r')
    ax2.tick_params('y', colors='r')

    vals = ax2.get_yticks()
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # hide y-labels on right side
    show_pct_y = False
    if not show_pct_y:
        ax2.set_yticks([])

    pct_format='{0:.0%}'
    formatted_weights = [pct_format.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], cumsum[i]), fontweight='heavy')    

    title = f'Pareto plot of Main and Interaction Effects on Response'
    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

class factorial_doe:
    """
    This class facilitates the analysis of full factorial and/or central composite design experiments.
    """
    def __init__(self, data):
        """
            Inputs:
                data: Pandas DataFrame. Contains the data for the experiments, including the factors and the responses. 
        """
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
        self.data_raw = data
        self.data = pd.DataFrame()
        
        self.factor_names = []
        self.factor_labels = []
        self.factor_lookup = dict()
        self.factor_ranges = []
        self.models = []
        self.results = []
        self.results_details = []

    def encode(self, value, lower, upper):
        """
        Converts factor values into their design space equivalent values.
        """
        # Convert tuples to numpy arrays, perform elementwise operations, then convert result back to a tuple to preserve typing.
        if type(value) is tuple:
            value = np.array(value)
            out = (value - (lower+upper)/2) / ((upper-lower)/2)
            out = tuple(out)
        # Perform elementwise operations on single values or numpy arrays and preserve typing.
        else:
            out = (value - (lower+upper)/2) / ((upper-lower)/2)

        return out

    def decode(self, value_coded, lower, upper):
        """
        Converts factor values from design space to experimental values.
        """
        # Convert tuples to numpy arrays, perform elementwise operations, then convert result back to a tuple to preserve typing.
        if type(value_coded) is tuple:
            value_coded = np.array(value_coded)
            out = value_coded * (upper-lower)/2 + (upper+lower)/2
            out = tuple(out)
        # Perform elementwise operations on single values or numpy arrays and preserve typing.
        else:
            out = value_coded * (upper-lower)/2 + (upper+lower)/2

        return out
    
    def select_data(self, factor_names, data_bounds = None):
        """
        Converts user-specified columns of data to their design space values and creates a new dataframe to hold their values.

        Inputs:
            factor_names: list of strings. Contains the column headers that correspond to the desired factors to be analyzed.
            data_bounds: dict. Specifies the upper and lower bounds for the values each factor can take. Contains a key:value pair for each factor of the form: 'factor_name': [lower_bound, upper_bound].
        """
        self.factor_names = factor_names

        # Convert each factor's values to design space values and append to dataframe, with generic factor label.
        for i, factor_name in enumerate(factor_names):
            if data_bounds is None:
                bound_low = np.min(self.data_raw[factor_name].values)
                bound_high = np.max(self.data_raw[factor_name].values)
            else:
                bound_low = data_bounds[factor_name][0]
                bound_high = data_bounds[factor_name][1]
            
            # Add a new column to dataframe containing the encoded factor values.
            self.data[self.letters[i]] = self.encode(self.data_raw[factor_name].values, bound_low, bound_high)

            # Append factor label to list of labels for future reference.
            self.factor_labels.append(self.letters[i])

            # Add factor name to a dictionary for later use in translating the generic label.
            self.factor_lookup[self.letters[i]] = factor_name

            # Record the min and max ranges for the factor, for use in conversion later.
            self.factor_ranges.append((bound_low, bound_high))
        
    def select_responses(self, response_names):
        """
        Specifies which column(s) of data can be used as the response for the analysis.

        Input:
            response_name: list of strings. Strings should be the column headers for the desired data.
        """
        for i, response_name in enumerate(response_names):
            self.data[f'Y{i}'] = self.data_raw[response_name].values
    
    def encode_factors(self, factors):
        """
        Convert a complete set of factors from experimental values to their corresponding design space values.

        Inputs:
            factors: list-like. Contains the set of factors to be converted. Each element should reprsent a separate factor.
        """
        encoded_values = []
        for factor, bounds in zip(factors, self.factor_ranges):
            encoded_values.append(self.encode(factor, bounds[0], bounds[1]))
        
        return encoded_values

    def decode_factors(self, factors):
        """
        Convert a complete set of factors from design space values to their corresponding experimental values.

        Inputs:
            factors: list-like. Contains the set of factors to be converted. Each element should reprsent a separate factor.
        """
        decoded_values = []
        for factor, bounds in zip(factors, self.factor_ranges):
            decoded_values.append(self.decode(factor, bounds[0], bounds[1]))
        
        return decoded_values

    def define_model(self, model_types = ['linear'], response = 'Y0'):
        """
            Generates a model string and formula for use in least squares fitting and subsequent model evaluation.

            Inputs:
                model_types: list of strings. Options are: 'linear', 'interaction', and 'quadratic'.
                response: string. Specifies which coded response should be used for fitting the model under construction.
        """
        # Initialize a dictionary to hold model info
        model_dict = dict()

        # Initialize the model string.
        model_string = ''

        # Generate the model string
        for model_type in model_types:
            # Add terms that are linear in the factors (example: 'A')
            if model_type == 'linear':
                for index, label in enumerate(self.factor_labels):
                    model_string = model_string + f'{label} + '
            # Add interaction terms (example: A:B)
            if model_type == 'interaction':
                for index, label in enumerate(self.factor_labels):
                    for i in range(index+1, len(self.factor_labels)):
                        model_string = model_string + f'{self.factor_labels[index]}:{self.factor_labels[i]} + '
            # Add terms that are quadratic in the factors (example: A^2)
            if model_type == 'quadratic':
                for index, label in enumerate(self.factor_labels):
                    model_string = model_string + f'np.power({label},2) + '

        # Remove the extra ' + ' from the end of the model string.
        model_string = model_string[:-3]
        
        # Store for use in model evaluation in the future.
        model_dict['model_string'] = model_string

        # Store for model fitting in the future.
        model_dict['formula'] = f'{response} ~ {model_string}'

        self.models.append(model_dict)

    
    def fit_models(self):
        """
            Performs least-squares fitting of the model to previously-specified data.
        """
        for model in self.models:
            y, X = dmatrices(model['formula'], data=self.data, return_type='dataframe')
            mod = sm.OLS(y, X)
            model['model_fit'] = mod.fit()

    def model_predict(self,factor_values, model_index = 0, guess = None):
        """
            Evaluates a fitted model at a specified set of input values.

            Inputs:
                factor_values: list. Contains the input values to feed into the model. Should be the same length as the number of independent variables in the model.
        """
        if guess == None:
            guess = np.zeros_like(factor_values)
        x1 = pd.DataFrame(data=[factor_values],columns=self.factor_labels)
        X = dmatrix(self.models[model_index]['model_string'], data=x1, return_type='dataframe')
        return self.models[model_index]['model_fit'].predict(exog =X).values[0]

    def model_optimize(self, guess = None, maximize = False, bounds = None):
        """
            Find a factor values that give a local optimum of the model for the response surface. The default behavior is to find the minimum value of the function.

            Inputs:
                guess: list-like containing ints or floats. Optional. Specifies an initial guess for the optimizer. Should contain one element for each of the factors.
                maximize: bool. Specifies that the optimizer should find the maximum value rather than the minimum.
                bounds: 
        """
        # Provide an initial guess of the optimum coordinates if one is given.
        if guess == None:
            guess = np.zeros((1,len(self.factor_labels)))
        
        # Convert the bounds for optimization parameters to design space value equivalents if bounds are given.
        if bounds is not None:
            bounds = self.encode_factors(bounds)

        for index in range(len(self.models)):
            opt = minimize(
                lambda factors: (-1)**(maximize) * self.model_predict(factors, model_index = index), 
                guess, 
                bounds = bounds)
            self.results_details.append(opt)
            self.results.append(self.decode_factors(opt.x))

    # ANALYSIS 
    def plot_response_surfaces(self, response_name):
        _plot_response_surfaces(self.data_raw, response_name, self.factor_names)
    
    def plot_effect_grid(self, response_name):
        _plot_effect_grid(self.data_raw, response_name, self.factor_names)

    def pareto_plot(self, response_name):
        effects = calc_effects(self.data_raw, response_name, self.factor_names)
        xlabel = 'Factors'
        ylabel = f'Magnitude of Effect on {response_name}'
        factor_letter_pairs = itertools.combinations(self.factor_labels, 2)
        factor_letters = self.factor_labels + list(factor_letter_pairs)
        _Pareto_plot(effects, factor_letters, xlabel, ylabel)