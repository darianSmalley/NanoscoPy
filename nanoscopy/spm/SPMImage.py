from collections.abc import MutableMapping

from numpy import exp
import pandas as pd

CHANNEL_ERROR = 'Channel not supported.'
DIRECTION_ERROR = 'Direction not supported.'

class SPMImage(MutableMapping):
    def __init__(self, path='', *args, **kwargs):
        self.path = path
        self.params = dict()
        self.headers = dict()
        self.data = {'Z':[], 'Current':[]}
        self.traces = {'Z':[], 'Current':[]}
        self.update(*args, **kwargs)
    
    def reformate(self, channel):
        cols = [
            'channels',
            'paths',
            'setpoints',
            'voltages',
            'widths',
            'heights',
            'datetimes',
            'directions',
            'traces',
            'images'
        ]
        data = {col:[] for col in cols}
        n = len(self.data[channel])

        for i in range(n):
            data['channels'].append(channel)
            data['directions'].append(self.traces[channel][i]['direction'])
            data['traces'].append(self.traces[channel][i]['trace'])
            data['setpoints'].append(self.params['setpoint_value'])
            data['voltages'].append(self.params['setpoint_value'])
            data['widths'].append(self.params['setpoint_value'])
            data['heights'].append(self.params['setpoint_value'])
            data['datetimes'].append(self.params['date_time'].isoformat())
            data['paths'].append(self.path)
            img = self.data[channel][i]
            data['images'].append(img)

        return data

    def as_dataframe(self):
        Z_data = self.reformate('Z')
        Z_dataframe = pd.DataFrame(Z_data)

        I_data = self.reformate('Current')
        I_dataframe = pd.DataFrame(I_data)

        return pd.concat([Z_dataframe,I_dataframe])

    def add_param(self, param, value):
        self.params[param] = value

    def add_data(self, channel, data):
        if channel not in self.data:
            self.data[channel] = []
            
        self.data[channel].append(data)

    def add_trace(self, channel, direction, trace):
        if channel not in self.data:
            self.traces[channel] = []

        self.traces[channel].append({'direction': direction, 'trace': trace})

    def add_dataframe(self, dataframe):
        self.dataframe = dataframe

    def get_data(self, channel):
        return dict(zip(self.traces[channel], self.data[channel]))
    
    def set_headers(self, headers):
        self.headers = headers

    def summary(self):
        sample_id = self.params['sample_id']
        date = self.params['date_time'].strftime('%y%m%d')
        bias = self.params['bias']
        size = round(self.params['width'] * pow(10,9))
        rec_index = self.params['rec_index']
        return f"{sample_id}_{date}_{bias}V_{size}x{size}_Z_{rec_index}"

    # The next five methods are requirements of the collection
    def __setitem__(self, key, value):
        self.add_data(key, value)
    def __getitem__(self, key):
        return self.data[key]
    def __delitem__(self, key):
        del self.data[key]
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        '''returns simple dict representation of the mapping'''
        # return ', '.join("%s: %s" % item for item in vars(self).items())
        return f'Path: {self.path}\nParams: {self.params}'
    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return '{}, SPMImage({})'.format(super(SPMImage, self).__repr__(), 
                                  self.data)        