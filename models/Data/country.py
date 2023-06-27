class country:
    """
    A class that represents the individual countries in the simulation

    ...

    Attributes
    ----------
    id: int
        an id to identify the country
    name: str
        the name of the country
    initial_ta: float
        the initial temperature in the country
    initial_gdp: float
        the initial gdp in the country
    weight_gdp: float
        the weight of the country to compute a weighted global gdp
    gdp: float
        the current simulation gdp in the country
    ta: float
        the current simulation temperature in the country

    Methods
    ---------- 
    compute_weight_gdp(total_gdp: float)
        Computes the relative weight of the country with respect to the global aggregate gdp
    update_ta(ta: float)
        Updates the simulation temperature
    update_gdp(gdp: float)
        Updates the simulation gdp
    return_weighted_gdp()
        Returns the weighted gdp of this country
    """
    
    def __init__(self, id: int, name: str, initial_ta: float, initial_gdp: float):
        """
        Parameters
        ----------
        id: int
            id to identify the country
        name: str
            name of the country
        initial_ta: float
            initial temperature in the country
        initial_gdp: float
            initial gdp in the country
        """

        self.id = id
        self.name = name
        self.initial_ta = initial_ta
        self.initial_gdp = initial_gdp
        self.weight_gdp = None
        self.ta = None
        self.gdp = None 
    
    def compute_weight_gdp(self, total_gdp: float):
        """ Computes the weight of this country as a ratio to the total aggregate gdp in the first year of the simulation. This weight stays constant throughout the simulation.

        Parameters
        ----------
        total_gdp: float
            The aggregate total gdp in the initial year of the simulation
        """

        self.weight_gdp = self.initial_gdp / total_gdp
    
    def update_ta(self, ta: float):
        """ Updates the temperature.

        Parameters
        ----------
        ta: float
            The current simulation temperature
        """

        self.ta = ta

    def update_gdp(self, gdp: float):
        """ Updates the gdp.

        Parameters
        ----------
        gdp: float
            The current simulation gdp
        """

        self.gdp = gdp
    
    def return_weighted_gdp(self) -> float:
        """ Returns the weighted gdp of this country to be later used to estimate the aggregated weighted global gdp
        """

        return self.gdp * self.weight_gdp