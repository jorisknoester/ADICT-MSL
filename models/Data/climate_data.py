import pandas as pd
from os import path

class climate_data:
    """
    A class that retrieves the climate data

    ...

    Attributes
    ----------
    scenarios: int
        the number of simulation scenarios
    climate_scenarios: int
        the number of projected termperature paths in Excel
    ta_hist: float
        the historical temperature data
    ta: np.array
        the array with the future temperature data from the TCRE projection

    Methods
    ---------- 
    retrieve_data(PATH: path, rcp: str)
        Obtains the climate data from the Excel files, dependent on which temperature trajectory is selected
    """

    def __init__(self, scenarios: int):
        """
        Parameters
        ----------
        scenarios: int
            number of scenarios for the simulation
        """

        self.scenarios = scenarios
        self.climate_scenarios = 2000
        self.ta_hist = None
        self.ta = None

    def retrieve_data(self, PATH: path, rcp: str):
        """ Retrieves the historical and future climate data. The future climate data depends on the RCP emission path that is selected.

        Parameters
        ----------
        PATH: path
            Path where the data is saved
        rcp: str
            String that represents the RCP temperature path
        """

        df_ta_hist = pd.DataFrame(pd.read_excel(PATH + 'Historical data graphs.xlsx', sheet_name='Climate data', header=0))
        self.ta_hist = df_ta_hist['Temperature'].values[len(df_ta_hist)-20:]
        df_ta = pd.DataFrame(pd.read_excel(PATH + 'Future climate data.xlsx', sheet_name='Temperature', header=0))
        if rcp == '2.6':
            self.ta = df_ta.iloc[:self.scenarios, :].values
        elif rcp == '4.5':
            self.ta = df_ta.iloc[self.climate_scenarios: self.climate_scenarios + self.scenarios, :].values
        elif rcp == '8.5':
            self.ta = df_ta.iloc[2* self.climate_scenarios: 2 * self.climate_scenarios + self.scenarios, :].values
        else:
            raise ValueError('RCP path not existing in database')
            