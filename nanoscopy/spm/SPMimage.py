CHANNEL_ERROR = 'Channel not supported.'
DIRECTION_ERROR = 'Direction not supported.'

class SPMImage:
    traces = ['forward/up', 'forward/down', 'backward/up', 'backward/down']
    channels = ['Z' , 'Current', 'Amplitude']

    def __init__(self, path='', *args, **kwargs):
        self.path = path
        self.params = dict()
        self.data = dict()
    
    def add_data(self, data, channel_name = 'Z', trace = 'forward/up'):
        if trace not in traces: 
            raise ValueError(DIRECTION_ERROR)

        self.data[channel_name].append({trace: data})
    