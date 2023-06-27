import pandas as pd
from os import path
import statsmodels.tsa.filters.hp_filter as hp_filter
import statsmodels.api as sm
import statsmodels.regression.linear_model as lm
import statsmodels.tsa.stattools as stattools
import statsmodels.tsa.ar_model as ar
from copulas.multivariate import GaussianMultivariate
import numpy as np
import numpy.ma as ma
from typing import Tuple

class eco_monte_carlo_simulation:
    """
    A class that computes the economic future time series

    ...

    Attributes
    ----------
    ncomp_bc: int
        the number of pca factors for the business cycle domain
    ncomp_trend: int
        the number of pca factors for the trend domain
    copula: bool
        boolean that determines whether the monte carlo residuals are sampled from the gaussian copula or directly with independent samples from cholesky decomposition
    n_economic: int
        the number of economic variables
    n_countries: int
        the number of countries
    prev_years: int
        the number of previous years in the data that is used in the regression
    years: int
        the number of simulation years
    scenarios: int
        the number of simulation scenarios
    eco_data_excel: bool
        boolean that determines whether the inner outcomes are written to excel, e.g pca factors
    print_res: bool
        boolean that determines if some steps are printed in the console
    df: pd.DataFrame
        dataframe that contains the historical economic data
    df_array: np.array
        array that contains the historical economic data
    hist_means: np.array
        historical means of the economic time series
    hp_series: np.array
        array that consists of the hodrick-prescott filtered series
    hp_series_stand: np.array
        array that contains the standardized hodrick-prescott filtered series
    pca_bc: np.array
        array that consists of the pca factor for the business cycle domain
    pca_trend: np.array
        array that consists of the pca factor for the trend domain
    yw_bc: np.array
        the yule-walker coefficients for the business cycle domain
    yw_trend : np.array
        the yule-walker coefficients for the trend domain
    cov_pca_bc : np.array
        the covariance matrix of the residuals for the pca factors of the business cycle domain
    cov_pca_trend : np.array
        the covariance matrix of the residuals for the pca factors of the trend domain
    err_pca_bc : np.array
        the simulated error terms to be added in the monte carlo simulation for the pca factors of the business cycle domain
    err_pca_trend : np.array
        the simulated error terms to be added in the monte carlo simulation for the pca factors of the trend domain
    fut_pca_bc : np.array
        the simulated future pca factors for the business cycle domain
    fut_pca_trend : np.array
        the simulated future pca factors for the trend domain
    coefs_bc : np.array
        array with the coefficients of the final economic regression for the business cycle domain
    coefs_trend : np.array
        array with the coefficients of the final economic regression for the trend domain
    cov_coefs_bc : np.array
        the covariance matrix of the residuals for the economic time series of the business cycle domain
    cov_coefs_trend : np.array
        the covariance matrix of the residuals for the economic time series of the trend domain
    err_coefs_bc : np.array
        the simulated error terms to be added in the monte carlo simulation for the economic time series of the business cycle domain
    err_coefs_trend : np.array
        the simulated error terms to be added in the monte carlo simulation for the economic time series of the trend domain
    eco_series = []
        the array containing the final list of future simulated economic series
    column_names : np.array
        array with all the time series names

    Methods
    ---------- 
    generate_eco_data(PATH_data: path, PATH_res: path, weights: np.array, world_gdp_arr: np.array, countries_gdp_arr: np.array)
        Main function that simulates the economic future outcomes
    read_data(PATH_data: path)
        Retrieves the Excel file with the historic data and removes the climate variables
    prepare_data(self)
        Transforms the data to the appropiate format
    hp_filtering(PATH_res: path)
        Filters the time series with the hodrick-prescott filter in a business cycle and trend domain
    compute_hp_series(hp_serie: np.array, start_index: int)
        Computes the standardized array
    compute_hist_means(PATH_res: path)
        Retrieves the historical means of the data
    pca(PATH_res: path, weights: np.array)
        Estimates the historical pca factors
    compute_historic_world_gdp(weights: np.array, gdp_df: np.array)
        Calculates the historical weighted gdp
    compute_pca(PATH_res: path, hp_series: np.array, bc: bool)
        Subfunction of pca() to estimate the historical pca factors
    yule_walker(self)
        Main function for yule-walker function to divide the business cycle and trend domain
    compute_yule_walker(bc: bool)
        Yule-walker method to determine the var model for the pca factors
    future_pca(gdp_world: np.array)
        Main function for pca simulation function to divide the business cycle and trend domain
    compute_future_pca(hist_pca: np.array, bc: bool)
        Simulates the pca factors with the computed yule-walker coefficients into the future
    draw_samples(pca: bool)
        Draws error terms to be added to the future pca factors and economic time series
    estimate_cholesky_samples(pca: bool, bc: bool)
        Uses cholesky decomposition with an independent uniform sampling method
    estimate_copula_samples(pca: bool, bc: bool)
        Uses cholesky decomposition with the gaussian copula
    simulate_future_pca(PATH_res: path)
        Main function of compute_future_sim_pca to seperate business cycle and trend domain
    compute_future_sim_pca(bc: bool)
        Adds the error terms to the projected future pca factors
    retrieve_hist_eco_coefs(PATH_res: path)
        Main function of compute_hist_eco_coefs()
    compute_hist_eco_coefs(PATH_res: path, bc: bool)
        Estimates the coefficients for the economic time series regression on the pca factors
    simulate_future_eco(countries_gdp_arr: np.array)
        Main function of compute_future_eco()
    compute_future_eco(bc: bool, countries_gdp_arr: np.array)
        Main function of compute_fut_eco()
    compute_fut_eco(hp_series: np.array, err: np.array, coefs: np.array, fut_pca: np.array, v: int)
        Simulates the final economic series with the use of the pca factors, historical regression coefficients, and the error terms
    """

    def __init__(self, ncomp_bc: int, ncomp_trend: int, copula: bool, n_economic: int, n_countries: int, prev_years: int, years: int, scenarios: int, eco_data_excel: bool, print_res: bool):
        """
        Parameters
        ----------
        ncomp_bc: int
            the number of pca factors for the business cycle domain
        ncomp_trend: int
            the number of pca factors for the trend domain
        copula: bool
            boolean that determines whether the monte carlo residuals are sampled from the gaussian copula or directly with independent samples from cholesky decomposition
        n_economic: int
            the number of economic variables
        n_countries: int
            the number of countries
        prev_years: int
            the number of previous years in the data that is used in the regression
        years: int
            the number of simulation years
        scenarios: int
            the number of simulation scenarios
        eco_data_excel: bool
            boolean that determines whether the inner outcomes are written to excel, e.g pca factors
        print_res: bool
            boolean that determines if some steps are printed in the console
        """

        self.ncomp_bc = ncomp_bc
        self.ncomp_trend = ncomp_trend
        self.copula = copula
        self.n_economic = n_economic
        self.n_countries = n_countries
        self.prev_years = prev_years
        self.years = years
        self.scenarios = scenarios
        self.eco_data_excel = eco_data_excel
        self.print_res = print_res
        self.df = None
        self.df_array = None
        self.hist_means = None
        self.hp_series = None
        self.hp_series_stand = None
        self.pca_bc = None
        self.pca_trend = None
        self.yw_bc = None
        self.yw_trend = None
        self.cov_pca_bc = None
        self.cov_pca_trend = None
        self.err_pca_bc = None
        self.err_pca_trend = None
        self.fut_pca_bc = None
        self.fut_pca_trend = None
        self.coefs_bc = None
        self.coefs_trend = None
        self.cov_coefs_bc = None
        self.cov_coefs_trend = None
        self.err_coefs_bc = None
        self.err_coefs_trend = None
        self.eco_series = []
        self.column_names = None

    def generate_eco_data(self, PATH_data: path, PATH_res: path, weights: np.array, world_gdp_arr: np.array, countries_gdp_arr: np.array):
        """ Calls all individual function and sets attributes equal to the outcomes of these functions.

        Parameters
        ----------
        PATH_data: path
            Path where the data is retrieved
        PATH_res: path
            Path where the data is saved
        weights: np.array
            array with the country-specific gdp weights
        world_gdp_arr: np.array
            array with the simulated world gdp rate
        countries_gdp_arr: np.array
            array with the regional simulated gdp rates
        """

        self.read_data(PATH_data)
        self.prepare_data()
        self.column_names = self.df.columns
        self.df_array = self.df.values
        self.hp_filtering(PATH_res)
        self.compute_hist_means(PATH_res)
        self.pca(PATH_res, weights)
        self.yule_walker()
        self.future_pca(world_gdp_arr)
        self.err_pca_bc, self.err_pca_trend = self.draw_samples(True)
        self.simulate_future_pca(PATH_res)
        self.retrieve_hist_eco_coefs(PATH_res)
        self.err_coefs_bc, self.err_coefs_trend = self.draw_samples(False)
        self.simulate_future_eco(countries_gdp_arr)

    def read_data(self, PATH_data: path):
        """ Retrieves the historical economic data and removes unnecessary columns

        Parameters
        ----------
        PATH_data: path
            Path where the data is retrieved
        """

        df = pd.DataFrame(pd.read_excel(PATH_data + 'Regression data.xlsx', sheet_name='Data'))
        df.columns = df.columns.astype(str)
        df =  df.loc[:, ~df.columns.str.startswith('Temperature')]
        df =  df.loc[:, ~df.columns.str.startswith('Mean sea level')]
        df =  df.loc[:, ~df.columns.str.startswith('1')]
        df =  df.loc[:, ~df.columns.str.startswith('2')]
        self.df = df

    def prepare_data(self):
        """ Adjusts the data to ensure the rows and columns correspond to the same years
        """

        countries = self.df.columns[self.n_economic+1:]
        asset_classes = self.df.columns[:self.n_economic+1]
        final_columns = []
        for country in countries:
            for asset in asset_classes:
                # For every country and every economic serie, add the name to the columns of the dataframe
                final_columns.append(country + ' ' + asset)
        final_df = pd.DataFrame(np.full((0, len(final_columns)), np.nan), columns=final_columns)
        for country in countries:
            for asset in asset_classes:
                # For every economic time series ensure that the start years are similar and set missing years to nan
                rows = self.df.drop(self.df[self.df[country] != 1].index, inplace=False)
                rows = rows[asset]
                missing_years = self.prev_years - len(rows)
                if missing_years > 0:
                    rows = np.concatenate([np.full((missing_years), np.nan), rows])
                else:
                    rows.reset_index(drop=True, inplace=True)
                final_df[country + ' ' + asset] = rows
        final_df.dropna(axis=1, how='all', inplace=True)
        self.df = final_df.iloc[:self.prev_years, :]

    def hp_filtering(self, PATH_res: path):
        """ Filters the historical time series in a business cycle and trend domain

        Parameters
        ----------
        PATH_res: path
            Path where the data is saved
        """

        self.hp_series = []
        self.hp_series_stand = []
        for s in range(len(self.df.columns)):
            # For every time series s, filter the data in two elements
            start_index = np.where(~np.isnan(self.df_array[:, s]))[0][0]
            hp_serie = hp_filter.hpfilter(self.df_array[start_index:, s], lamb=6.25)
            hp_serie_bc = hp_serie[0]
            hp_serie_trend = hp_serie[1]
            self.compute_hp_series(hp_serie_bc, start_index)
            self.compute_hp_series(hp_serie_trend, start_index)
        self.hp_series, self.hp_series_stand = np.reshape(self.hp_series, (len(self.df_array[0]), 2*len(self.df_array))), np.reshape(self.hp_series_stand, (len(self.df_array[0]), 2*len(self.df_array)))
        self.hp_series, self.hp_series_stand = np.transpose(self.hp_series), np.transpose(self.hp_series_stand)
        if self.eco_data_excel:
            # Write the filtered series to Excel if wanted
            hp_df = pd.DataFrame(self.hp_series, columns=self.df.columns)
            hp_df.to_excel(PATH_res + 'Filtered series.xlsx', index=False, sheet_name='Filtered')
            hp_df = pd.DataFrame(self.hp_series_stand, columns=self.df.columns)
            hp_df.to_excel(PATH_res + 'Filtered series standardized.xlsx', index=False, sheet_name='Filtered stand')

    def compute_hp_series(self, hp_serie: np.array, start_index: int):
        """ Ensures that data years are similar across series and estimates standardized filtered series

        Parameters
        ----------
        hp_serie: np.array
            array with the filtered serie for one economic variable
        start_index: int
            start index of the data for this variable
        """

        if start_index > 0:
            hp_serie = np.concatenate([np.full((start_index), np.nan), hp_serie])
        self.hp_series = np.append(self.hp_series, hp_serie)
        hp_serie = (hp_serie - np.nanmean(hp_serie))/np.nanstd(hp_serie)
        self.hp_series_stand = np.append(self.hp_series_stand, hp_serie)

    def compute_hist_means(self, PATH_res: path):
        """ Computes the historical means of the economic variables

        Parameters
        ----------
        PATH_res: path
            Path where the data is saved
        """

        self.hist_means = self.df.mean(axis=0).values
        if self.eco_data_excel:
            hist_means_df = pd.DataFrame(self.hist_means, index=self.df.columns.values)
            hist_means_df.to_excel(PATH_res + 'Historical means.xlsx', sheet_name='Long-term means')
        
    def pca(self, PATH_res: path, weights: np.array):
        """ Main function to compute the historical pca factors for both frequency domains and replaces the first trend pca factor with the historic weighted world gdp rate

        Parameters
        ----------
        PATH_res: path
            Path where the data is saved
        weights:
            array with the country-specific gdp weights
        """

        years = int(len(self.hp_series_stand)/2)
        hp_series = np.transpose(self.hp_series_stand)
        hp_series_bc = hp_series[:, :years]
        self.hist_means = np.reshape(self.hist_means, (len(self.hist_means), 1))
        hp_series_trend = hp_series[:, years:] + self.hist_means
        self.pca_bc = self.compute_pca(PATH_res, hp_series_bc, True)
        self.pca_trend = self.compute_pca(PATH_res, hp_series_trend, False)
        gdp_df = self.df.loc[:, self.df.columns.str.endswith('GDP')].values
        self.pca_trend[:, 0] = self.compute_historic_world_gdp(weights, gdp_df)
    
    def compute_pca(self, PATH_res: path, hp_series: np.array, bc: bool) -> np.array:
        """ Computes the historical pca factors

        Parameters
        ----------
        PATH_res: path
            Path where the data is saved
        hp_series: np.array
            the filtered series
        bc: bool
            boolean that separates business cycle with trend
        """

        hp_series = np.transpose(hp_series)
        if bc:
            ncomp = self.ncomp_bc
        else:
            ncomp = self.ncomp_trend
        pca = sm.multivariate.PCA(hp_series, ncomp=ncomp, standardize=True, demean=True, normalize=True, gls=False, missing='drop-col')
        if self.eco_data_excel:
            print('PCA square', pca.rsquare)
            df_pca = pd.DataFrame(pca.factors)
            df_pca.to_excel(PATH_res + 'PCA factors ' + str(ncomp) +  '.xlsx', sheet_name='PCA')
        return pca.factors
    
    def compute_historic_world_gdp(self, weights: np.array, gdp_df: np.array) -> np.array:
        """ Computes the weighted world gdp rate from the individual country rates.

        Parameters
        ----------
        weights:
            array with the country-specific gdp weights
        gdp_df:
            array with the historical gdp rate
        """

        sum_weights = np.full((len(gdp_df)), np.nan)
        # Not all series have the same starting year, so the weights of the countries are adjusted to this
        for y in range(len(gdp_df)):
            sum_weight = 0
            for c in range(len(gdp_df[0])):
                if not np.isnan(gdp_df[y, c]):
                    sum_weight += weights[c]
            sum_weights[y] = sum_weight
        adjusted_weights = np.full((len(gdp_df), len(gdp_df[0])), np.nan)
        for y in range(len(gdp_df)):
            for c in range(len(gdp_df[0])):
                if not np.isnan(gdp_df[y, c]):
                    adjusted_weights[y, c] = weights[c] / sum_weights[y]
        gdp_world = []
        # Estimate the weighted historical gdp rate
        for y in range(len(gdp_df)):
            weighted_gdp = 0
            for c in range(len(gdp_df[0])):
                if not np.isnan(gdp_df[y, c]):
                    weighted_gdp += gdp_df[y, c] * adjusted_weights[y, c]
            gdp_world.append(weighted_gdp)
        return gdp_world

    def yule_walker(self):
        """ Main function that calls compute_yule_walker for both domains
        """

        self.yw_bc, self.cov_pca_bc = self.compute_yule_walker(True)
        self.yw_trend, self.cov_pca_trend = self.compute_yule_walker(False)

    def compute_yule_walker(self, bc: bool) -> Tuple[np.array, np.array]:
        """ Estimates the yule-walker coefficients of the historical pca factors 

        Parameters
        ----------
        bc: bool
            boolean that separates business cycle and trend domain
        """

        if bc:
            pca = self.pca_bc
        else:
            pca = self.pca_trend
        order = 1
        yw_list = []
        resids = []
        for f in range(len(pca[0])):
            mean = np.mean(pca[:, f])
            res = lm.yule_walker(pca[:, f], order=order, demean=True)[0]
            res = np.insert(res, 0, mean)
            yw_list.append(res)
            residuals = []
            # Save the residuals for the Monte carlo simulation
            for y in range(order, len(pca)):
                residual = pca[y, f]
                for n in range(order):
                    residual -=  yw_list[f][n] * pca[y-(n+1), f]
                residuals.append(residual)      
            resids.append(residuals)
        if self.print_res:
            print('PACF', f)
            print(stattools.acf(pca[:, f], nlags=order, qstat=True, alpha=0.05)[3])
            print(stattools.pacf(pca[:, f], nlags=order, alpha=0.05)[0])
            model = ar.AutoReg(pca[:, f], lags=order).fit()
            print(model.params)
            print(model.pvalues)
            print(model.aic)
        return np.array(yw_list, dtype=np.float32), np.cov(np.array(resids, dtype=np.float32))

    def future_pca(self, gdp_world: np.array):
        """ Main function that calls compute_future_pca and replaces the first trend pca factor with the simulated weighted world gdp rate

        Parameters
        ----------
        gdp_world: np.array
            the weighted gdp world rate
        """

        hist_pca_bc = self.pca_bc[len(self.pca_bc)-1:, :]
        hist_pca_trend = self.pca_trend[len(self.pca_trend)-1:, :]
        self.fut_pca_bc = self.compute_future_pca(hist_pca_bc, True)
        self.fut_pca_trend = self.compute_future_pca(hist_pca_trend, False)
        self.fut_pca_trend[:, 0] = np.median(gdp_world, axis=1)

    def compute_future_pca(self, hist_pca: np.array, bc: bool) -> np.array:
        """ Computes the future pca factors

        Parameters
        ----------
        hist_pca: np.array
            historical pca factors required for the first observation
        bc: bool
            boolean that separates business cycle and trend domain
        """

        if bc:
            yw = self.yw_bc
        else:
            yw = self.yw_trend
        fut_pca = []
        for y in range(self.years):
            pcas = []
            for f in range(len(yw)):
                pca = yw[f, 0]
                for p in range(len(yw[0])-1):
                    # If statements introduced for obtaining the historical pca factor. Function is written for more lags than one.
                    if y < len(yw[0]) - 2:
                        h_pca = hist_pca[y + p, f]
                    elif y == len(yw[0]) - 2:
                        if p == 0:
                            h_pca = hist_pca[y, f]
                        else:
                            h_pca = fut_pca[y - len(yw[0]) + (p+1)][f]
                    else:
                        h_pca = fut_pca[y - len(yw[0]) + (p+1)][f]
                    pca += yw[f, p+1] * h_pca
                pcas.append(pca)
            fut_pca.append(pcas)
        return np.array(fut_pca, dtype=np.float32)

    def draw_samples(self, pca: bool) -> Tuple[np.array, np.array]:
        """ Draws samples to be used as error terms for the monte carlo simulation from either copula or cholesky

        Parameters
        ----------
        pca: bool
            boolean that separates the error terms of the pca factors and the economic series
        """

        if self.copula:
            err_bc = self.estimate_copula_samples(pca, True)
            err_trend = self.estimate_copula_samples(pca, False)
        else:
            err_bc = self.estimate_cholesky_samples(pca, True)
            err_trend = self.estimate_cholesky_samples(pca, False)
        return err_bc, err_trend

    def estimate_cholesky_samples(self, pca: bool, bc: bool) -> np.array:
        """ Uses an independent transformation method to sample error terms from the cholesky decomposition. Takes 5 minutes to simulate 100 years for 2000 simulations

        Parameters
        ----------
        pca: bool
            boolean that separates pca factor covariance matrix from economic variables
        bc: bool
            boolean that separates business cycle and trend domain
        """

        if pca:
            if bc:
                cov = self.cov_pca_bc
            else:
                cov = self.cov_pca_trend
            
        else:
            if bc:
                cov = self.cov_coefs_bc
            else:
                cov = self.cov_coefs_trend
        L = np.linalg.cholesky(cov)
        err = []
        for y in range(self.years):
            U_1 = np.random.uniform(0, 1, size=(len(L), self.scenarios))
            U_2 = np.random.uniform(0, 1, size=(len(L), self.scenarios))
            Z = np.sqrt(-2 * np.log(U_1)) * np.cos(2 * np.pi * U_2)
            err.append(np.matmul(L, Z))
        return np.array(err,  dtype=np.float32)
    
    def estimate_copula_samples(self, pca: bool, bc: bool) -> np.array:
        """ Uses copula theory to sample error terms from the cholesky decomposition. Takes 3.5 hours to simulate 100 years for 2000 simulations for one model.

        Parameters
        ----------
        pca: bool
            boolean that separates pca factor covariance matrix from economic variables
        bc: bool
            boolean that separates business cycle and trend domain
        """

        if pca:
            if bc:
                cov = self.cov_pca_bc
            else:
                cov = self.cov_pca_trend
        else:
            if bc:
                cov = self.cov_coefs_bc
            else:
                cov = self.cov_coefs_trend
        L = np.linalg.cholesky(cov)
        copula = GaussianMultivariate()
        copula.fit(L)
        err = []
        for y in range(self.years):
            # Very slow computation
            if y % 20 == 0:
                print(y)
            err.append(np.transpose(copula.sample(self.scenarios)))
        return np.array(err,  dtype=np.float32)

    def simulate_future_pca(self, PATH_res: path):
        """ Main function that calls compute_future_sim_pca

        Parameters
        ----------
        PATH_res: path
            Path where data is saved
        """

        self.fut_pca_bc = self.compute_future_sim_pca(True)
        self.fut_pca_trend = self.compute_future_sim_pca(False)
        if self.eco_data_excel:
            with pd.ExcelWriter(PATH_res + 'PCA factors BC.xlsx') as writer:          
                for v in range(len(self.fut_pca_bc[0])):
                    df = pd.DataFrame(self.fut_pca_bc[:, v])
                    df.to_excel(writer, sheet_name=str(v))
            with pd.ExcelWriter(PATH_res + 'PCA factors trend.xlsx') as writer:          
                for v in range(len(self.fut_pca_trend[0])):
                    df = pd.DataFrame(self.fut_pca_trend[:, v])
                    df.to_excel(writer, sheet_name=str(v))

    def compute_future_sim_pca(self, bc: bool) -> np.array:
        """ Adds the error terms to the pca factors to obtain different scenarios.

        Parameters
        ----------
        bc: bool
            boolean that separates business cycle and trend domain
        """

        if bc:
            err = self.err_pca_bc
            fut_pca = self.fut_pca_bc
        else:
            err = self.err_pca_trend
            fut_pca = self.fut_pca_trend
        fut_pca_sim = []
        for y in range(self.years):
            pca_sim = []
            for f in range(len(err[0])):
                pca_s = []
                for s in range(len(err[0][0])):
                    pca_s.append(fut_pca[y, f] + err[y, f, s])
                pca_sim.append(pca_s)
            fut_pca_sim.append(pca_sim)
        fut_pca_sim = np.array(fut_pca_sim, dtype=np.float32)
        return fut_pca_sim

    def retrieve_hist_eco_coefs(self, PATH_res: path):
        """ Calls compute_hist_eco_coefs twice for both domains
        """
        self.coefs_bc, self.cov_coefs_bc = self.compute_hist_eco_coefs(PATH_res, True)
        self.coefs_trend, self.cov_coefs_trend = self.compute_hist_eco_coefs(PATH_res, False)

    def compute_hist_eco_coefs(self, PATH_res: path, bc: bool) -> np.array:
        """ Estimates the historical regression coefficients for the economic variables against the pca factors

        Parameters
        ----------
        PATH_res: path
            Path where data is saved
        bc: bool
            boolean that separates business cycle and trend domain
        """

        if bc:
            hp_series = self.hp_series[:int(len(self.hp_series)/2), :]
            pca = self.pca_bc
        else:
            hp_series =  self.hp_series[int(len(self.hp_series)/2):, :]
            pca = self.pca_trend
        eco_coefs = []
        eco_residuals = []
        for h, hp_serie in enumerate(np.transpose(hp_series)):
            exogenous_var = pca[1:]
            exogenous_var = np.insert(exogenous_var, 0, hp_serie[:-1], axis=1)
            exogenous_var = sm.add_constant(exogenous_var, prepend=True)
            coefs = lm.OLS(hp_serie[1:], exogenous_var, missing='drop', hasconst=True).fit().params
            if not bc:
                # If trend domain, add penalty weights to the first order var coefficient to avoid exploding series. Multiple steps are applied because some series contain heavy first order coefficients
                if coefs[1] > 0.95:
                    weights = np.full(len(exogenous_var[0]), 0.0)  
                    weights[1] = 0.00001
                    coefs = lm.OLS(hp_serie[1:], exogenous_var, missing='drop', hasconst=True).fit_regularized(alpha=weights, L1_wt=0).params
                    if coefs[1] > 0.95:
                        weights[1] = 0.0001
                        coefs = lm.OLS(hp_serie[1:], exogenous_var, missing='drop', hasconst=True).fit_regularized(alpha=weights, L1_wt=0).params
                        if coefs[1] > 1:
                            weights[1] = 0.01
                            coefs = lm.OLS(hp_serie[1:], exogenous_var, missing='drop', hasconst=True).fit_regularized(alpha=weights, L1_wt=0).params
            eco_coefs.append(coefs)
            residuals = []
            # Save residuals for the error terms of the monte carlo simulation
            for n in range(len(hp_serie)-1):
                residual = hp_serie[n + 1]
                for c, coef in enumerate(eco_coefs[h]):
                    residual -= coef * exogenous_var[n, c]
                residuals.append(residual)
            eco_residuals.append(residuals)
        bootstrapped_residuals = []
        # Because the historical data contains 43 years and 216 series, the residuals are bootstrapped with a small adjustment to avoid a non-positive definite matrix for the cholesky decomposition
        for i in range(30):
            bootstrapped_residuals.append(eco_residuals)
        bootstrapped_residuals = np.array(bootstrapped_residuals, dtype=np.float32)
        bootstrapped_residuals = np.reshape(bootstrapped_residuals, (len(bootstrapped_residuals[0]), 30 * len(bootstrapped_residuals[0][0])))
        for i in range(len(bootstrapped_residuals)):
            for j in range(len(bootstrapped_residuals[0])):
                if np.isnan(bootstrapped_residuals[i, j]):
                    bootstrapped_residuals[i, j] = bootstrapped_residuals[i, j]
                else:
                    bootstrapped_residuals[i, j] += - 0.0000001 + 0.0000002 * np.random.rand()
        eco_residuals = bootstrapped_residuals
        if self.eco_data_excel:
            df = pd.DataFrame(eco_coefs)
            df.to_excel(PATH_res + 'Eco historical coefficients ' + str(len(eco_coefs[0])) + '.xlsx')
        return np.array(eco_coefs, dtype=np.float32), ma.cov(ma.masked_invalid(np.array(eco_residuals, dtype=np.float32)))

    def simulate_future_eco(self, countries_gdp_arr: np.array):
        """ Main function that calls the sub functions to simulate the final time series

        Parameters
        ----------
        countries_gdp_arr: np.array
            the simulated gdp rates of the individual countries
        """

        series_bc = self.compute_future_eco(True, countries_gdp_arr)
        series_trend = self.compute_future_eco(False, countries_gdp_arr)
        series = series_bc + series_trend
        series = np.clip(series, -1, 1)
        self.eco_series.append(series)
        self.eco_series = np.array(self.eco_series, dtype=np.float32)
        self.eco_series = np.reshape(self.eco_series, (len(self.eco_series[0]), len(self.eco_series[0][0]), len(self.eco_series[0][0][0])))

    def compute_future_eco(self, bc: bool, countries_gdp_arr: np.array) -> np.array:
        """ Selects the required input arguments and passes them to compute_fut_eco to estimate the scenarios.

        Parameters
        ----------
        countries_gdp_arr: np.array
            the simulated gdp rates of the individual countries
        bc: bool
            boolean that separates business cycle and trend domain
        """

        if bc:
            hp_series = self.hp_series[:int(len(self.hp_series)/2), :]
            err = self.err_coefs_bc
            fut_pca = self.fut_pca_bc
            coefs = self.coefs_bc   
        else:
            hp_series =  self.hp_series[int(len(self.hp_series)/2):, :]
            err = self.err_coefs_trend
            fut_pca = self.fut_pca_trend
            coefs = self.coefs_trend
        series = []
        for v in range(self.n_countries * (self.n_economic + 1)):
            if v % (self.n_economic + 1) == 0:
                if not bc:
                    series.append(countries_gdp_arr[int((v/(self.n_economic + 1)))])
                else:
                    series.append(np.full((len(countries_gdp_arr[0]), len(countries_gdp_arr[0][0])), 0))
            else:
                eco_variables = self.compute_fut_eco(hp_series, err, coefs, fut_pca, v)
                series.append(eco_variables)
        return np.array(series, dtype=np.float32)
    
    def compute_fut_eco(self, hp_series: np.array, err: np.array, coefs: np.array, fut_pca: np.array, v: int) -> np.array:
        """ Simulate the future economic time series

        Parameters
        ----------
        hp_series: np.array
            The historical hodrick-prescott filtered required for the lag in the regression
        err: np.array
            The error terms to be added for the monte carlo simulation
        coefs: np.array
            The regression coefficients of the economic pca regression
        fut_pca: np.array
            The future pca simulated factors
        v: int
            The index of the time series
        """

        eco_variables = []
        for y in range(self.years):
            series_year = []
            for s in range(self.scenarios):
                if y > 0:
                    previous_var = eco_variables[y-1][s]
                else:
                    previous_var = hp_series[-1, v]
                res = coefs[v, 0] + err[y, v, s]
                count = 0
                for c in range(0, len(coefs[0]) - 1):
                    if c == 0:
                        res += coefs[v, 1] * previous_var
                        count = 1
                    else:
                        res += coefs[v, c + 1] * fut_pca[y, c - count, s]
                # Clip the annual rates to [-1, 1] per year. Lower is impossible and higher also not logical
                res = max(-1, res)
                res = min(1, res)
                if v % (self.n_economic + 1) == self.n_economic:
                    # If unemployment rate, minimum is 0%
                    res = max(0, res)             
                series_year.append(res)
            eco_variables.append(series_year)
        return eco_variables