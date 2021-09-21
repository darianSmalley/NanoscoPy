from collections.abc import MutableMapping

CHANNEL_ERROR = 'Channel not supported.'
DIRECTION_ERROR = 'Direction not supported.'

class SPMImage(MutableMapping):
    def __init__(self, path='', *args, **kwargs):
        self.path = path
        self.params = dict()
        self.data = dict()
        self.traces = dict()
        self.update(*args, **kwargs)
    
    def add_data(self, channel, data):
        self.data[channel].append(data)
        
    def add_trace(self, channel, direction, trace):
        self.traces[channel].append({'direction': direction, 'trace': trace})

    def add_param(self, param, value):
        self.params[param] = value

    def get_data(self, channel):
        return zip(self.traces[channel], self.data[channel])

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
        return str(self.data)
    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return '{}, D({})'.format(super(SPMImage, self).__repr__(), 
                                  self.data)        