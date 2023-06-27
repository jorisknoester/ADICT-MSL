from os import path
import pandas as pd
import numpy as np

class climate_tipping_point_data:
    """
    A class that retrieves the climate tipping points data

    ...

    Attributes
    ----------
    int_scalar: int
        the parameter to scale the interactions. For the report, this parameter is set to 10.
    ctp: np.array
        an array that contains the data for all the climate tipping points in the simulation
    interaction: np.array
        the array that contains the interaction matrix

    Methods
    ---------- 
    retrieve_data(PATH: path)
        Obtains the climate tipping points data from the Excel files
    compute_interaction_matrix()
        Retrieves the interaction matrix from the Excel file.
    """

    def __init__(self, int_scalar: int):
        """
        Parameters
        ----------
        scenarios: int
            number of scenarios for the simulation
        """

        self.int_scalar = int_scalar
        self.ctp = None
        self.interaction = None
    
    def retrieve_data(self, PATH: path):
        """ Retrieves the climate tipping point and interaction matrices.

        Parameters
        ----------
        PATH: path
            Path where the data is saved
        """

        self.ctp = pd.DataFrame(pd.read_excel(PATH + 'Input.xlsx', sheet_name='CTP', header=0)).values
        self.interaction = pd.DataFrame(pd.read_excel(PATH + 'Input.xlsx', sheet_name='Interaction', header=0, index_col=0)).values
    
    def compute_interaction_matrix(self):
        """ Calculates the interaction matrix from the input with the interaction scalar and a square root function.
        """

        domino_effects = self.interaction[:int(len(self.interaction)/2), :]
        hidden_feedbacks = self.interaction[int(len(self.interaction)/2):, :]
        self.interaction = 1 / self.int_scalar * np.sqrt(domino_effects + hidden_feedbacks)
        np.fill_diagonal(self.interaction, 1)