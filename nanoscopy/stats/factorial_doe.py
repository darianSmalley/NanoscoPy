import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from patsy import dmatrix
from scipy.optimize import minimize

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
                