import numpy as np


class climate_tipping_point:
    """
    A class that represent the climate tipping points.

    ...

    Attributes
    ----------
    id: int
        a number to create differences between the multiple climate tipping points
    name: str
        the name of the climate tipping point
    min_ta: float
        minimal temperature for which the climate tipping point shifts regimes
    thresh_ta: float
        expected temperature for which the climate tipping point shifts regimes
    max_ta: float
        maximal temperature for which the climate tipping point shifts regimes
    uncert: int
        uncertainty regarding the temperature critical threshold values
    impact_ta: float
        maximal temperature impact of the climate tipping point
    impact_msl: float
        maximal mean sea level impact of the climate tipping point
    period: int
        the number of years the climate tipping point affects the environment

    Methods
    ----------
    compute_delta()
        computes the delta associated with the regime shifting mechanism
    step()
        adds one every year to the time once the climate tipping point has shifted regimes.
    interaction(int_arr: np.array, act_arr: np.array, omega: float)
        calculates the interaction value of the climate tipping point using the interaction matrix, the activations of the other climate tipping points, and the interaction parameter omega  
    activation(ta: float)
        estimates the activation scalar of the climate tipping point using a logistic activation function
    impact(tau: float)
        computes the temperature and mean sea level damage of the climate tipping point using the activation statistic
    """

    def __init__(self, id: int, name: str, min_ta: float, thresh_ta: float, max_ta: float, uncert: int, impact_ta: float, impact_msl: float, period: int):
        """
        Parameters
        ----------
        id: int
            The id of the climate tipping point
        name: str
            The name of the climate tipping point
        min_ta: float
            The minimal temperature for which the climate tipping point shifts regimes
        thresh_ta: float
            The expected temperature for which the climate tipping point change states
        max_ta: float
            The maximal temperature for which the climate tipping point shifts regimes
        uncert: int
            The uncertainty regarding the temperature critical threshold values
        impact_ta: float
            The maximal temperature impact of the climate tipping point
        impact_msl: float
            The maximal mean sea level impact of the climate tipping point
        period: int
            The number of years the climate tipping point affects the environment
        """

        self.id = id
        self.name = name
        self.min_ta = min_ta
        self.thresh_ta = thresh_ta
        self.max_ta = max_ta
        self.uncert = uncert
        self.impact_ta = impact_ta
        self.impact_msl = impact_msl
        self.period = period
        self.delta = 1
        self.start_t = 0
        self.int_val = 0
        self.act_val = 0
        self.imp_val_ta = 0
        self.imp_val_msl = 0

    def compute_delta(self):
        """Estimates the delta for the climate tipping point using a formula that is based on literature and graphical analysis. It is a function of the uncertainty with respect to the tipping temperatures and the bandwidth between the maximum and minimum temperature.
        """

        self.delta = 5 - (1 + 0.2 * self.uncert) * (self.max_ta - self.min_ta)/2

    def step(self):
        """Adds one every year to the activated time parameter, start_ta, once the climate tipping point has changed states."""

        if self.start_t > 0:
            self.start_t += 1

    def interaction(self, int_arr: np.array, act_arr: np.array, omega: float):
        """Computes the interaction statistics, which is based on binary activation values of the other climate tipping points, the interaction matrix, and omega, which symbolizes the maginuted of the interaction.
        
        Parameters
        ----------
        int_arr: np.array
            The interaction matrix that contains the interaction statistics for every climate tipping point combination
        act_arr: np.array
            A binary vector with activation values of the other climate tipping points, which is either a one or a zero and depends on the logistic activation function and the threshold parameter.
        omega: float
            The interaction parameter that determines the impact of the interaction among the climate tipping points
        """

        int_val = 0
        for j in range(len(int_arr)):
            if j == self.id:
                continue
            int_val += int_arr[j, self.id] * act_arr[j]
        self.int_val = omega * int_val

    def activation(self, ta: float):
        """Defines the activation of the climate tipping point using the current temperature in logistic activation function with the interaction statistic and the estimated parameter delta.
        
        Parameters
        ----------
        ta: float
            The current global temperature
        """

        thresh = self.thresh_ta - self.int_val
        denominator = 1 + np.exp(-self.delta*(ta - thresh))
        self.act_val = max(self.act_val, 1 / denominator)

    def impact(self, tau: float):
        """Calculates the environmental impact of the climate tipping point on the global temperature and the mean sea level. Whether there is any impact depens on the threshold parameter tau. The magnitude of impact is defined by the maximum temperature and mean sea level effect, the number of tipped years, and the maximum period of damage.
        
        Parameters
        ----------
        tau: float
            The threshold parameter that determines from which logistic activation statistic the climate tipping point has really shifted regimes.
        """

        if self.act_val < tau:
            # If the activation scalar is lower than the actual required threshold parameter tau, no damage is exerted, there is no damage.
            self.imp_val_ta = 0
            self.imp_val_msl = 0
        elif self.act_val >= tau and self.start_t <= self.period:
            # Else if the threshold is passed and the number of tipped years is still lower than the maximum period of damage, the impact increases yearly.
            if self.start_t == 0:
                self.start_t = 1
            self.imp_val_ta = self.start_t/self.period * self.impact_ta
            self.imp_val_msl = self.start_t/self.period * self.impact_msl
        else:
            # Else the total impact is equal to the maximum temperature and mean sea level impact and stays constant.
            self.imp_val_ta = self.impact_ta
            self.imp_val_msl = self.impact_msl