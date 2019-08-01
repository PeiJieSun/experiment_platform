import ConfigParser as cp
import re, os

class ParserConf():

    def __init__(self, config_path):
        self.config_path = config_path

    def processValue(self, key, value):
        #print(key, value)
        value = value.replace(',', '')
        tmp = value.split(' ')
        dtype = tmp[0]
        value = tmp[1:]
        #print(dtype, value)

        if 'show_list' in key:
            for i in value:
                self.show_list.append(i)

        if value != None:
            if dtype == 'string':
                self.conf_dict[key] = vars(self)[key] = value[0]
            elif dtype == 'int':
                self.conf_dict[key] = vars(self)[key] = int(value[0])
            elif dtype == 'float':
                self.conf_dict[key] = vars(self)[key] = float(value[0])
            elif dtype == 'list':
                self.conf_dict[key] = vars(self)[key] = [i for i in value]
            elif dtype == 'int_list':
                self.conf_dict[key] = vars(self)[key] = [int(i) for i in value]
            elif dtype == 'float_list':
                self.conf_dict[key] = vars(self)[key] = [float(i) for i in value]
        else:
            print('%s value is None' % key)

    def parserConf(self, project='', platform_config='', data_name='', model_name=''):
        conf = cp.ConfigParser()
        conf.read(self.config_path)
        self.conf = conf

        self.conf_dict = {}
        # sometimes the show_list may contain too many keys, we have to specify the show list with multilines, thus we 
        # store the show keys with a set
        self.show_list = []
        for section in conf.sections():
            for (key, value) in conf.items(section):
                #print(key, value)
                self.processValue(key, value)

        if platform_config != '':
            [app_log_dir, app_data_dir, app_out_dir] = platform_config
            self.model_name = model_name
            self.data_name = data_name
            self.project_name = project
            # self.root_dir example: /home/sunpeijie/files/task/pyrec/log/dual_learning/amazon_books
            self.root_dir = os.path.join(app_log_dir, project, data_name)
            # self.data_dir example: /home/sunpeijie/files/task/pyrec/data/dual_learning/amazon_books/amazon_books
            self.data_dir = os.path.join(app_data_dir, project, data_name, data_name)
            # self.out_dir example: /home/sunpeijie/files/task/pyrec/out/dual_learning/amazon_books/
            self.out_dir = os.path.join(app_out_dir, project, data_name)