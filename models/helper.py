import configparser

""" Method to read and print config file settings """
def read_config(print_res: bool):
    config = configparser.ConfigParser()
    config.read('configurations.ini')
    file=open('configurations.ini','r')
    settings=file.read()
    if print_res:
        print(settings)
    return config