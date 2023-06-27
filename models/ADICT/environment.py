import numpy as np

from Regression.regression import regression
from Data.climate_data import climate_data
from Data.climate_tipping_point_data import climate_tipping_point_data
from Data.country import country
from ADICT.climate_tipping_point import climate_tipping_point
from ADICT.eco_monte_carlo_simulation import eco_monte_carlo_simulation

class environment:
    """
    A class that represents the environment in which the climate tipping points affect the gdp growth rate outcomes.

    ...

    Attributes
    ----------
    regress: regression
        a regression object that includes the attributes of the initial damage regression
    clim_data: climate_data
        a climate_data object that includes the future projected climate temperature
    eco_data: economic_data
        an economic_data object that includes the data to create the individual climate tipping points
    ctp_data: climate_tipping_point_data
        an climate_tipping_point_data object that includes the data to create the climate tipping points and the interaction matrix
    ctps: List[climate_tipping_point]
        a list with all the climate tipping points as individual objects
    countries: List[country]
        a list with all the countries as individual objects
    ctp_module: bool
        boolean that determines whether or not climate tipping points are included in the simulation
    msl_module: bool
        boolean that determines whether or not the mean sea level impact of the climate tipping points is included in the simulation
    ta_2: bool
        that determines whether or not the squared temperature is part of the damage function
    n_climate: int
        the number of climate variables
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
    act_df: np.array
        binary array that contains the binary activation value of all climate tipping points for each year for the different simulations
    gdp: np.array  
        gdp array that contains the gdp values of all countries for each year for the different simulations
    weights: np.array
        array that consists of the gdp weights of each country to determine the global weighted gdp
    weighted_gdp: np.array
        array that contains the projected weighted gdp for each year in all the scenario paths
    ta_arr: np.array
        array that contains the projected global temperature for each year in all the scenario paths
    msl_arr: np.array
        array that contains the projected global mean sea level for each year in all the scenario paths
    delta_ta_arr: np.array
        array that contains the projected delta global temperature for each year in all the scenario paths, compared to the initial temperature
    delta_msl_arr: np.array
        array that contains the projected delta global mean sea level for each year in all the scenario paths, compared to the initial mean sea level

    Methods
    ---------- 
    run(t: int)
        Runs one year for all the scenarios in which it computes the weighted gdp 
    run_scenario(t: int, s: int)
        Runs one scenario for one specific year in which it estimates the projected temperature and mean sea level change and the impact on global gdp
    compute_activation_array(t: int, s: int)
        Fills the binary activation array for each year t and scenario s, dependent on the activation values of the climate tipping points 
    compute_gdp(t: int, s: int, delta_ta: float, delta_msl: float)
        Computes the gdp for each country, dependent on the damage function, the initial and global temperature, and the initial and global mean sea level
    """

    def __init__(self, regress: regression, clim_data: climate_data, eco_data: eco_monte_carlo_simulation, ctp_data: climate_tipping_point_data, countries_data: np.array, ctp_module: bool, msl_module: bool, ta_2: bool, ta_0: float, msl_0: float, years: int, scenarios: int, omega: float, tau: float):
        """
        Parameters
        ----------
        regress: regression
            a regression object that includes the attributes of the initial damage regression
        clim_data: climate_data
            a climate_data object that includes the future projected climate temperature
        eco_data: economic_data
            an economic_data object that includes the data to create the individual climate tipping points
        ctp_data: climate_tipping_point_data:
            an climate_tipping_point_data object that includes the data to create the climate tipping points and the interaction matrix
        countries_data: np.array
            an array with all the data for the countries
        ctp_module: bool
            boolean that determines whether or not climate tipping points are included in the simulation
        msl_module: bool
            boolean that determines whether or not the mean sea level impact of the climate tipping points is included in the simulation
        ta_2: bool
            that determines whether or not the squared temperature is part of the damage function
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
        """
        self.regress = regress
        self.clim_data = clim_data
        self.eco_data = eco_data
        self.ctp_data = ctp_data
        self.ctps = [[climate_tipping_point] * len(ctp_data.ctp) for c in range(scenarios)]
        self.countries = [[country] * len(countries_data) for c in range(scenarios)]
        total_gdp = sum(countries_data[:, 2])
        for s in range(scenarios):
            for c, ctp_arr in enumerate(ctp_data.ctp):
                self.ctps[s][c] = climate_tipping_point(c, ctp_arr[0], ctp_arr[1], ctp_arr[2], ctp_arr[3], ctp_arr[4], ctp_arr[5], ctp_arr[6], ctp_arr[7])
                self.ctps[s][c].compute_delta()
            for c, country_arr in enumerate(countries_data):
                self.countries[s][c] = country(c, country_arr[0], country_arr[1], country_arr[2])
                self.countries[s][c].compute_weight_gdp(total_gdp)
        self.ctp_module = ctp_module
        self.msl_module = msl_module
        self.ta_2 = ta_2
        self.n_climate = 2
        if ta_2:
            self.n_climate = 3
        self.ta_0 = ta_0
        self.msl_0 = msl_0      
        self.years = years
        self.scenarios = scenarios
        self.omega = omega
        self.tau = tau
        self.act_df = np.zeros((len(self.ctps[0]), years, scenarios))
        self.gdp = np.full((len(countries_data), years, self.scenarios), np.nan)
        self.weights = np.full(len(countries_data), np.nan)
        self.weighted_gdp = np.full((years, self.scenarios), np.nan)
        self.ta_arr = np.full((years, scenarios), np.nan)
        self.msl_arr = np.full((years, scenarios), np.nan)
        self.delta_ta_arr = np.full((years, scenarios), np.nan)
        self.delta_msl_arr = np.full((years, scenarios), np.nan)
    
    def run(self, t: int):
        """ Runs one year for all the scenarios. First it computes the country-specific gdps, which is converted to a weighted global gdp afterwards.

        Parameters
        ----------
        t: int
            The year of the simulation
        """
        for s in range(self.scenarios):
            self.run_scenario(t, s)
        for s in range(self.scenarios):
            weighted_gdp = 0
            for c in range(len(self.countries[0])):
                if s == 0 and t == 0:
                    # Only if it is the first scenario in the first year of the simulation, the weights of the countries need to be determined once.
                    self.weights[c] = self.countries[s][c].weight_gdp
                weighted_gdp += self.countries[s][c].return_weighted_gdp()
            self.weighted_gdp[t, s] = weighted_gdp
    
    def run_scenario(self, t: int, s: int):
        """ Runs one year for one specific scenarios. It retrieves the impact of the individual climate tipping points on the global temperature and mean sea level to be used later to estimate the effect on regional gdp.

        Parameters
        ----------
        t: int
            The year of the simulation
        s: int
            The specific scenario path
        """
        ta_global = self.clim_data.ta[s, t]
        delta_msl = 0
        delta_ta = 0
        if self.ctp_module:
            # If the climate tipping points are included, estimate their impact on global temperature and mean sea level
            for ctp in self.ctps[s]:
                # Compute the impact of the climate tipping points on temperature and mean sea level
                ctp.step()
                ctp.impact(self.tau)
                delta_ta += ctp.imp_val_ta
                if self.msl_module:
                    # If the mean sea level impact of climate tipping points is included in the simulation, add this effect as well
                    delta_msl += ctp.imp_val_msl
            for ctp in self.ctps[s]:
                ctp.activation(ta_global)
            # Update the activation array
            self.compute_activation_array(t, s)
            for ctp in self.ctps[s]:
                ctp.interaction(self.ctp_data.interaction, self.act_df[:, t, s], self.omega)
        ta_global += delta_ta
        # Formulas to determine the global mean sea level, is based on historical data. The if statements are introduced due to the fact that the arrays do not contain the initial years of the mean sea level and temperature statistics.
        msl = -0.7602 + 0.76041 * ta_global
        if t == 0:
            msl += 0.942326 * self.msl_0 + 0.709683 * self.clim_data.ta_hist[t]
        elif 0 < t < 20:
            msl += 0.942326 * self.msl_arr[t-1, s] + 0.709683 * self.clim_data.ta_hist[t]
        else:
            msl += 0.942326 * self.msl_arr[t-1, s] + 0.709683 * self.ta_arr[t-20, s]
        msl += delta_msl
        # Update the temperature and mean sea level arrays
        self.delta_ta_arr[t, s] = delta_ta
        self.delta_msl_arr[t, s] = delta_msl
        self.ta_arr[t, s] = ta_global
        self.msl_arr[t, s] = msl
        # If statements introduced because msl_arr is first empty and does not contain historical mean sea level
        if t > 0:
            delta_t_msl = msl - self.msl_arr[t-1, s]
        else:
            delta_t_msl = msl - self.msl_0
        # Compute the regional gdp
        self.compute_gdp(t, s, ta_global - self.ta_0, delta_t_msl)
    
    def compute_activation_array(self, t: int, s: int):
        """ Updates the binary activation array for each year in all the scenarios. Depends on the logistic activation scalar of the climate tipping points and the threshold parameter tau.

        Parameters
        ----------
        t: int
            The year of the simulation
        s: int
            The specific scenario path
        """
        for c, ctp in enumerate(self.ctps[s]):
            if ctp.act_val > self.tau:
                self.act_df[c, t, s] = 1

    def compute_gdp(self, t: int, s: int, delta_ta: float, delta_t_msl: float):
        """ Computes the regional projected gdp from the global temperature and mean sea level statistics.

        Parameters
        ----------
        t: int
            The year of the simulation
        s: int
            The specific scenario path
        delta_ta: float
            the additional global temperature from the initial starting temperature
        delta_t_msl: float
            the difference in mean sea level from this year and previos year
        """
        coefs = self.regress.coefs
        for c in range(len(self.countries[0])):
            # For each country estimate the gdp statistic
            gdp = coefs[-1]
            gdp += coefs[self.n_climate + c]
            ta = self.countries[s][c].initial_ta + delta_ta
            if self.ta_2:
                # If the squared temperature is included in the damage function add also this effect
                ta_2 = np.square(ta)
                gdp += coefs[0] * ta + coefs[1] * ta_2 + coefs[2] * delta_t_msl
            else:
                gdp += coefs[0] * ta + coefs[1] * delta_t_msl
            # Gdp rate cannot be lower than -1, so a floor is implemented to ensure this does not happen. It does not influence the results, but is added as a guarantee.
            gdp = max(-1, gdp)
            self.countries[s][c].update_gdp(gdp)
            self.countries[s][c].update_ta(ta)
            self.gdp[c, t, s] = gdp