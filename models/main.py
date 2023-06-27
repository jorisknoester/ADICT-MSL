import pandas as pd
import numpy as np
from os import path
import warnings

import helper
from Regression.regression import regression
from Data.climate_data import climate_data
from ADICT.eco_monte_carlo_simulation import eco_monte_carlo_simulation
from Data.climate_tipping_point_data import climate_tipping_point_data
from ADICT.environment import environment

def main(rcp: str, ctp_module: bool, msl_module: bool, ta_0: float, msl_0: float, years: int, scenarios: int, omega: float, tau: float, print_res: bool, calibrate: bool, random_scenario: bool, climate_excel: bool, eco_excel: bool, countries_excel: bool):
    """ Main function that combines the different python modules of the ADICT+MSL system. Afterwards, it writes the results to Excel

        Parameters
        ----------
        rcp: str
            string that determines the RCP emission trajectory to be used for the TCRE temperature projection
        ctp_module: bool
            boolean that determines whether or not climate tipping points are included in the simulation
        msl_module: bool
            boolean that determines whether or not the mean sea level impact of the climate tipping points is included in the simulation
        ta_0: float
            initial global temperature
        msl_0: float
            initial global mean sea level
        years: int
            the number of simulation years
        scenarios: int
            the number of simulation scenarios
        omega: float
            the interaction parameter
        tau: float
            the temperature activation threshold parameter
        print_res:
            boolean that determines whether inner steps are printed in the console
        calibrate: bool
            boolean that determines whether the run is used for calibrating omega and tau
        random_scenario: bool
            boolean that determines whether a random example scenario is written to Excel
        climate_excel: bool
            boolean that determines whether the climate outcomes from the environment are written to Excel
        eco_excel: bool
            boolean that determines whether the economic outcomes are written to Excel
        countries_excel: bool
            boolean that determines whether the regional outcomes are written to Excel instead of only global weighted results
    """
    
    # Ignore runtime warnings
    warnings.filterwarnings('ignore')

    # Retrieve the fixed settings from the config file
    config = helper.read_config(print_res)

    PATH_data = config['SimulationSettings']['PATH_data']
    PATH_res = config['SimulationSettings']['PATH_res']
    ta_2 = bool(int(config['RegressionSettings']['ta_2']))
    countries = bool(int(config['RegressionSettings']['countries']))
    year_effects = bool(int(config['RegressionSettings']['year_effects']))
    newey_west = bool(int(config['RegressionSettings']['newey_west']))
    lags = int(config['RegressionSettings']['lags'])
    regress_excel = bool(int(config['RegressionSettings']['regress_excel']))
    int_scalar = float(config['ADICTSettings']['interaction_scalar'])
    ncomp_bc = int(config['SimulationSettings']['ncomp_bc'])
    ncomp_trend = int(config['SimulationSettings']['ncomp_trend'])
    copula = bool(int(config['SimulationSettings']['copula']))
    n_economic = int(config['SimulationSettings']['n_economic'])
    prev_years = int(config['SimulationSettings']['prev_years'])
    current_year = int(config['SimulationSettings']['current_year'])
    eco_data_excel = bool(int(config['SimulationSettings']['eco_data_excel']))

    print('RCP' + rcp + ' CTP ' + str(ctp_module) + ' MSL ' + str(msl_module) + ' Omega ' + str(omega) + ' Tau ' + str(tau))    

    # Create all the necessary objects for the simulation
    regress = regression(ta_2, countries, year_effects, newey_west, lags, prev_years, regress_excel, print_res)
    regress.estimate_regression(PATH_data, PATH_res)
    clim_data = climate_data(scenarios)
    clim_data.retrieve_data(PATH_data, rcp)
    ctp_data = climate_tipping_point_data(int_scalar)
    ctp_data.retrieve_data(PATH_data)
    ctp_data.compute_interaction_matrix()
    eco_data = eco_monte_carlo_simulation(ncomp_bc, ncomp_trend, copula, n_economic, regress.n_countries, prev_years, years, scenarios, eco_data_excel, print_res)
    eco_data.read_data(PATH_data)
    # Retrieve the regional data
    countries_data = pd.DataFrame(pd.read_excel(PATH_data + 'Input.xlsx', sheet_name='Countries', header=0)).values
    env = environment(regress, clim_data, eco_data, ctp_data, countries_data, ctp_module, msl_module, ta_2, ta_0, msl_0, years, scenarios, omega, tau)

    # Perform the simulation per year in the environment
    for t in range(years):
        env.run(t)
    
    # The booleans decide which data is written to Excel
    if calibrate:
        write_gdp_adict_to_excel(PATH_res, rcp, ctp_module, msl_module, omega, tau, years, current_year, env)
    if random_scenario:
        write_random_scenario_to_excel(PATH_res, scenarios, years, current_year, env, ctp_data)
    if climate_excel:
        write_climate_data_to_excel(PATH_res, rcp, ctp_module, msl_module, years, env)
    if eco_excel:
        eco_data.generate_eco_data(PATH_data, PATH_res, env.weights, env.weighted_gdp, env.gdp)
        write_world_series_to_excel(PATH_res, rcp, ctp_module, msl_module, omega, tau, years, current_year, env, eco_data)
        if countries_excel:
            write_eco_series_to_excel(PATH_res, rcp, ctp_module, msl_module, omega, tau, years, current_year, eco_data)

def write_gdp_adict_to_excel(PATH_res: path, rcp: str, ctp_module: bool, msl_module: bool, omega: float, tau: float, years: int, current_year: int, env: environment):
    """ Writes the calibration data to Excel

        Parameters
        ----------
        PATH_res: path
            The path where the results are written
        rcp: str
            string that determines the RCP emission trajectory to be used for the TCRE temperature projection
        ctp_module: bool
            boolean that determines whether or not climate tipping points are included in the simulation
        msl_module: bool
            boolean that determines whether or not the mean sea level impact of the climate tipping points is included in the simulation
        omega: float
            the interaction parameter
        tau: float
            the temperature activation threshold parameter
        years: int
            the number of simulation years
        current_year: int
            the current start year of the simulation, e.g. 2023
        env: environment
            the environment that is used for the ADICT+MSL simulation and contains the climate and gdp data
        """
    
    index_list = ['TA', 'GDP']
    climate_res = []
    for i in range(len(index_list)):
        if i == 0:
            climate_res.append(compute_res(env.delta_ta_arr, years))
        elif i == 1:
            climate_res.append(compute_res(env.weighted_gdp, years))
    with pd.ExcelWriter(PATH_res + 'Calibration data ' + str(rcp) + ' CTP ' + str(ctp_module) + ' MSL ' + str(msl_module) + ' ' + str(omega) + '-' + str(tau) + '.xlsx') as writer:
        for clim_r, clim_res in enumerate(climate_res):
            df = pd.DataFrame(clim_res, index=['2.5%', '25%', 'Median', '75%', '97.5%', 'Avg', 'Std', 'Min', 'Max'], columns=range(current_year, current_year + years))
            df.to_excel(writer, sheet_name=index_list[clim_r])

def write_random_scenario_to_excel(PATH_res: path, scenarios: int, years: int, current_year: int, env: environment, ctp_data: climate_tipping_point_data):
    """ Writes a random scenario to Excel

        Parameters
        ----------
        PATH_res: path
            The path where the results are written
        scenarios: int
            the number of simulation scenarios
        years: int
            the number of simulation years
        current_year: int
            the current start year of the simulation, e.g. 2023
        env: environment
            the environment that is used for the ADICT+MSL simulation and contains the climate and gdp data
        ctp_data: climate_tipping_point_data
            the climate_tipping_point_data object that contains the climate tipping point data
        """
    
    index_list = ctp_data.ctp[:, 0]
    index_list = np.append(index_list, ['TA', 'MSL', 'Delta TA', 'Delta MSL', 'GDP World'])
    random_s = int(scenarios * np.random.random(1))
    index_res = [env.ta_arr[:, random_s], env.ta_arr[:, random_s], env.msl_arr[:, random_s], env.delta_ta_arr[:, random_s], env.delta_msl_arr[:, random_s], env.delta_msl_arr[:, random_s]]
    n_tipping_points = len(ctp_data.ctp[:, 0])
    res = []
    for i in range(n_tipping_points):
        res.append(env.act_df[i, :, random_s])
    for i in range(len(index_list) - n_tipping_points):
        res.append(index_res[i])
    res = np.reshape(res, (len(index_list), years))
    df = pd.DataFrame(res, index=index_list, columns=range(current_year, current_year + years))
    df.to_excel(PATH_res + 'Random future scenario.xlsx', sheet_name='Scenario ' + str(random_s))

def write_climate_data_to_excel(PATH_res: path, rcp: bool, ctp_module: bool, msl_module: bool, years: int, current_year: int, env: environment):
    """ Writes the climate outcomes to Excel

        Parameters
        ----------
        PATH_res: path
            The path where the results are written
        rcp: str
            string that determines the RCP emission trajectory to be used for the TCRE temperature projection
        ctp_module: bool
            boolean that determines whether or not climate tipping points are included in the simulation
        msl_module: bool
            boolean that determines whether or not the mean sea level impact of the climate tipping points is included in the simulation
        years: int
            the number of simulation years
        current_year: int
            the current start year of the simulation, e.g. 2023
        env: environment
            the environment that is used for the ADICT+MSL simulation and contains the climate and gdp data
        """
    
    index_list = ['TA', 'MSL', 'Delta TA', 'Delta MSL', 'GDP']
    index_results = [env.ta_arr, env.msl_arr, env.delta_ta_arr, env.delta_msl_arr, env.weighted_gdp]
    climate_res = []
    for i in range(len(index_list)):
        climate_res.append(compute_res(index_results[i], years))
    with pd.ExcelWriter(PATH_res + 'Results climate impact RCP ' + str(rcp) + ' CTP ' + str(ctp_module) + ' MSL ' + str(msl_module) + '.xlsx') as writer:
        for clim_r, clim_res in enumerate(climate_res):
            df = pd.DataFrame(clim_res, index=['2.5%', '25%', 'Median', '75%', '97.5%', 'Avg', 'Std', 'Min', 'Max'], columns=range(current_year, current_year + years))
            df.to_excel(writer, sheet_name=index_list[clim_r])

def write_world_series_to_excel(PATH_res: path, rcp: bool, ctp_module: bool, msl_module: bool, omega: float, tau: float, years: int, current_year: int, env: environment, eco_data: eco_monte_carlo_simulation):
    """ Writes the weighted world series to Excel

        Parameters
        ----------
        PATH_res: path
            The path where the results are written
        rcp: str
            string that determines the RCP emission trajectory to be used for the TCRE temperature projection
        ctp_module: bool
            boolean that determines whether or not climate tipping points are included in the simulation
        msl_module: bool
            boolean that determines whether or not the mean sea level impact of the climate tipping points is included in the simulation
        omega: float
            the interaction parameter
        tau: float
            the temperature activation threshold parameter
        years: int
            the number of simulation years
        current_year: int
            the current start year of the simulation, e.g. 2023
        env: environment
            the environment that is used for the ADICT+MSL simulation and contains the climate and gdp data
        eco_data: eco_monte_carlo_simulation
            the eco_monte_carlo_simulation object that contains the projected economic time series
        """
    
    gdp_world = compute_world(0, env, eco_data)
    eq_world = compute_world(1, env, eco_data)
    cpi_world = compute_world(2, env, eco_data)
    nglr_world = compute_world(3, env, eco_data)
    unem_world = compute_world(4, env, eco_data)
    series_world = [gdp_world, eq_world, cpi_world, nglr_world, unem_world, env.ta_arr, env.msl_arr, env.weighted_gdp]
    world_stats = []
    for serie in series_world:
        world_stats.append(compute_res(serie, years))
    names = ['GDP stats', 'EQ stats', 'CPI stats', 'NGLR stats', 'UNEM stats', 'Temperature stats', 'Mean sea level stats', 'GDP ADICT stats']
    with pd.ExcelWriter(PATH_res + 'Future world data ' + str(rcp) + ' CTP ' + str(ctp_module) + ' MSL ' + str(msl_module) + ' ' + str(omega) + '-' + str(tau) + '.xlsx') as writer:
        for s, serie in enumerate(world_stats):
            df = pd.DataFrame(serie, index=['2.5%', '25%', 'Median', '75%', '97.5%', 'Avg', 'Std', 'Min', 'Max'], columns=range(current_year, current_year + years))
            df.to_excel(writer, sheet_name=names[s])

def compute_world(i: int, env: environment, eco_data: eco_monte_carlo_simulation) -> np.array:
    """ Computes the weighted global outcomes from the regional values

        Parameters
        ----------
        i: int
            integer that represents the index of the economic series, e.g. 1 represents equity returns
        env: environment
            the environment that is used for the ADICT+MSL simulation and contains the climate and gdp data
        eco_data: eco_monte_carlo_simulation
            the eco_monte_carlo_simulation object that contains the projected economic time series
        """
    
    series = []
    for c, serie in enumerate(eco_data.eco_series):
        if c % (eco_data.n_economic + 1) == i:
            series.append(env.countries[0][int(np.floor(c/(eco_data.n_economic + 1)))].weight_gdp * serie)              
    return np.sum(series, axis=0)

def compute_res(df: np.array, years: int) -> np.array:
    """ Computes the output in the required format to design the graphs in Excel. It estimates a couple of statistics from the simulated paths, such as percentiles and standard deviation.

        Parameters
        ----------
        df: np.array
            the time series that contains the economic series.
        years: int
            the number of simulation years
        """
    
    res = np.zeros((9, years))
    for y in range(years):
        res[0, y] = np.quantile(df[y], 0.025)
        res[1, y] = np.quantile(df[y], 0.25) - res[0, y]
        res[2, y] = np.median(df[y])
        res[3, y] = np.quantile(df[y], 0.75) - np.quantile(df[y], 0.25)
        res[4, y] = np.quantile(df[y], 0.975) - np.quantile(df[y], 0.75)
        res[5, y] = np.average(df[y])
        res[6, y] = np.std(df[y])
        res[7, y] = np.min(df[y])
        res[8, y] = np.max(df[y])
    return res

def write_eco_series_to_excel(PATH_res: path, rcp: bool, ctp_module: bool, msl_module: bool, omega: float, tau: float, years: int, current_year: int, eco_data: eco_monte_carlo_simulation):
    """ Writes the regional economic future time series to excel

        Parameters
        ----------
        PATH_res: path
            The path where the results are written
        rcp: str
            string that determines the RCP emission trajectory to be used for the TCRE temperature projection
        ctp_module: bool
            boolean that determines whether or not climate tipping points are included in the simulation
        msl_module: bool
            boolean that determines whether or not the mean sea level impact of the climate tipping points is included in the simulation
        omega: float
            the interaction parameter
        tau: float
            the temperature activation threshold parameter
        years: int
            the number of simulation years
        current_year: int
            the current start year of the simulation, e.g. 2023
        eco_data: eco_monte_carlo_simulation
            the eco_monte_carlo_simulation object that contains the projected economic time series
        """
    
    with pd.ExcelWriter(PATH_res + 'Future region eco data ' + str(rcp) + ' CTP ' + str(ctp_module) + ' MSL ' + str(msl_module) + ' ' + str(omega) + '-' + str(tau) + '.xlsx') as writer:          
        for v in range(len(eco_data.column_names)):
            res = compute_res(eco_data.eco_series[v], years) 
            df = pd.DataFrame(res, index=['2.5%', '25%', 'Median', '75%', '97.5%', 'Avg', 'Std', 'Min', 'Max'], columns=range(current_year, current_year + years))
            df.to_excel(writer, sheet_name=str(eco_data.column_names[v]))

if __name__ == '__main__':
    rcp = ['4.5']
    ctp_module = [True]
    msl_module = [True]
    omegas = [1]
    taus = [0.4]
    years = 100
    scenarios = 10
    ta_0 = 1.25
    msl_0 = 7.16
    print_res = False
    calibrate = False
    random_scenario = True
    climate_excel = False
    eco_excel = True
    countries_excel = False
    for r in rcp:
        for c in ctp_module:
            for m in msl_module:
                if not c and m:
                    continue
                for o in omegas:
                    for t in taus:
                        main(r, c, m, ta_0, msl_0, years, scenarios, o, t, print_res, calibrate, random_scenario, climate_excel, eco_excel, countries_excel)