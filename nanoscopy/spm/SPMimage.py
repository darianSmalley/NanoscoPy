class SPMImage:
    def __init__(self, path=''):
        self.path = path
        self.data = dict()
        self.parameters = dict()
    
    def add_data(self , data , channel_name , direction = 'Forward'):
        if channel_name not in self.data.keys():
            self.data[channel_name] = {'Forward': None , 'Backward': None }

        self.data[channel_name][direction] = data