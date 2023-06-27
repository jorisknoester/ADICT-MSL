import configparser
import os

# Create object
config_file = configparser.ConfigParser()

# Add FTPSettings sections
config_file.add_section('SimulationSettings')
config_file.add_section('RegressionSettings')
config_file.add_section('ADICTSettings')

# Add settings to the FTPSettings sections
# Add the paths
PATH = os.getcwd()
PATH_data = PATH + '\\Data\\'
PATH_res = PATH + '\\Run results\\'
config_file.set('SimulationSettings', 'PATH_data', PATH_data)
config_file.set('SimulationSettings', 'PATH_res', PATH_res)
# Add the regression settings
config_file.set('RegressionSettings', 'ta_2', '1')
config_file.set('RegressionSettings', 'countries', '1')
config_file.set('RegressionSettings', 'year_effects', '0')
config_file.set('RegressionSettings', 'newey_west', '1')
config_file.set('RegressionSettings', 'lags', '1')
config_file.set('RegressionSettings', 'regress_excel', '0')
# Add the ADICT settings
config_file.set('ADICTSettings', 'interaction_scalar', '10')
# Add the simulation settings
config_file.set('SimulationSettings', 'ncomp_bc', '9')
config_file.set('SimulationSettings', 'ncomp_trend', '3')
config_file.set('SimulationSettings', 'copula', '1')
config_file.set('SimulationSettings', 'n_economic', '4')
config_file.set('SimulationSettings', 'prev_years', '43')
config_file.set('SimulationSettings', 'current_year', '2023')
config_file.set('SimulationSettings', 'eco_data_excel', '0')

# Save config file
with open(r'configurations.ini', 'w') as configfileObj:
    config_file.write(configfileObj)
    configfileObj.flush()
    configfileObj.close()

print('Config file configurations.ini created')

# Print file content
read_file = open('configurations.ini', 'r')
content = read_file.read()
print('Content of the config file are:\n')
print(content)
read_file.flush()
read_file.close()
