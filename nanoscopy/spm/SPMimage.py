from collections.abc import MutableMapping

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

    def get_data(self, channel):
        return zip(self.traces[channel], self.data[channel])
    
    def set_headers(self, headers):
        self.headers = headers

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
        return ', '.join("%s: %s" % item for item in vars(self).items())
    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return '{}, SPMImage({})'.format(super(SPMImage, self).__repr__(), 
                                  self.data)        