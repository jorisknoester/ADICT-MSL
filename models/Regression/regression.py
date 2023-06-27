from typing import Tuple
import pandas as pd
import numpy as np
from os import path
import statsmodels.api as sm_api
import statsmodels.stats.descriptivestats as sm_descrstats
import statsmodels.stats.stattools as sm_stattools
import statsmodels.stats.diagnostic as sm_diag
import statsmodels.stats.outliers_influence as sm_outliers
from scipy import stats

class regression:
    """
    A class that estimates the coefficients of the damage function

    ...

    Attributes
    ----------
    ta_2: bool
        boolean that determines whether the squared temperature is included in the regression
    countries: bool
        boolean that determines whether the country-specific fixed effects are included in the regression
    year_effects:
        boolean that determines whether the yearly fixed effects are included in the regression
    newey_west: bool
        boolean that determines whether the newey-west standard errors are applied to the regression
    lags: int
        the number of lags for the newey-west standard errors
    prev_years: int
        the number of previous years included in the regression
    excel: bool
        boolean that determines whether the results are written to Excel
    print_res: bool
        boolean that determines whether the results are printed in the console
    model: regression_model
        the regression model from statsmodels that contains the damage function outcomes
    coefs: np.array
        the coefficients of the damage function
    ses: np.array
        the standard errors of the parameters in the damage function
    test_res: bool
        boolean that shows whether the regression test fails or not
    stats: float
        the statistical outcome of the mathematical regression tests
    pvals: float
        the p-value corresponding with the regression tests
    resids: np.array
        the residuals of the damage function
    n_countries: int
        the number of countries in the simulation

    Methods
    ---------- 
    estimate_regression(self, PATH_data: path, PATH_res: path)
        the main function that estimates the damage function and tests the results
    get_data(self, PATH_data: path)
        transforms the data to correspond with the required format
    perform_regression(self, df: pd.DataFrame)
        the (OLS)-regression
    return_result(self, stat: float, p_value: float)
        function that returns the outcomes of the tests in standard format
    t_test(self)
        the t-test
    endogeneity_test(self, df: pd.DataFrame)
        the endogeneity test
    serial_correlation_test(self)
        the breusch-godfrey serial correlation test
    heteroskedasticity_test(self, df: pd.DataFrame)
        the white heteroskedasticity test
    normality_test(self)
        the jarque-bera normality test
    multicollinearity_test(self, df: pd.DataFrame)
        the vif multicollinearity test
    """

    def __init__(self, ta_2: bool, countries: bool, year_effects: bool, newey_west: bool, lags: int, prev_years: int, excel: bool, print_res: bool):
        """
        Parameters
        ----------
        ta_2: bool
            boolean that determines whether the squared temperature is included in the regression
        countries: bool
            boolean that determines whether the country-specific fixed effects are included in the regression
        year_effects:
            boolean that determines whether the yearly fixed effects are included in the regression
        newey_west: bool
            boolean that determines whether the newey-west standard errors are applied to the regression
        lags: int
            the number of lags for the newey-west standard errors
        prev_years: int
            the number of previous years included in the regression
        excel: bool
            boolean that determines whether the results are written to Excel
        print_res: bool
            boolean that determines whether the results are printed in the console
        """

        self.ta_2 = ta_2
        self.countries = countries
        self.newey_west = newey_west
        self.lags = lags
        self.prev_years = prev_years
        self.year_effects = year_effects
        self.excel = excel
        self.print_res = print_res
        self.model = None
        self.coefs = None
        self.ses = None
        self.test_res = None
        self.stats = None
        self.pvals = None
        self.resids = None
        self.n_countries = 0

    def estimate_regression(self, PATH_data: path, PATH_res: path):
        """ The main function that estimates the coefficients of the damage function. Afterwards the outcomes are tested by the individual test functions. It prints the results if the print_res boolean is set to yes. It writes the regression outcomes to Excel, dependent on parameter excel.

        Parameters
        ----------
        PATH_data: path
            Path where the data is saved
        PATH_res: path
            Path to write the results to if wanted 
        """

        df = self.get_data(PATH_data)
        if self.excel:
            df.to_excel(PATH_data + 'Regression in between data.xlsx', index=False)
        self.perform_regression(df)
        self.t_test()
        if self.print_res:
            print('t', self.test_res, self.stats, self.pvals)
        self.endogeneity_test(df)
        if self.print_res:
            print('corr', self.test_res, self.stats, self.pvals)
        self.serial_correlation_test()
        if self.print_res:
            print('bg', self.test_res, self.stats, self.pvals)
        self.heteroskedasticity_test(df)
        if self.print_res:
            print('white', self.test_res, self.stats, self.pvals)
        self.normality_test()
        if self.print_res:
            print('jb', self.test_res, self.stats, self.pvals)
        self.multicollinearity_test(df)
        if self.print_res:
            print('vif', self.test_res, self.stats, self.pvals)
            print()
            print()
        if self.excel:
            final_df = pd.DataFrame([self.coefs, self.ses, self.pvals], columns=['Coefficients', 'Standard errors', 'P-values'])
            final_df.to_excel(PATH_res + 'Regression results.xlsx', sheet_name='Regression')

    def get_data(self, PATH_data: path) -> pd.DataFrame:
        """ Transforms the historical dataframe by eliminating columns to correspond with the format required by the boolean settings.

        Parameters
        ----------
        PATH_data: path
            Path where the data is saved
        """

        df = pd.DataFrame(pd.read_excel(PATH_data + 'Regression data.xlsx', sheet_name='Data'))
        df.columns = df.columns.astype(str) 
        countries_names = pd.DataFrame(pd.read_excel(PATH_data + 'Regression data.xlsx', sheet_name='Countries'))
        self.n_countries = len(countries_names.columns)
        economic_series = ['GDP', 'EQ', 'CPI', 'NGLR', 'UNEM']
        climate_series = ['Temperature', 'Temperature_2', 'Mean sea level']
        if not self.ta_2:
            # If the squared temperature is not included, delete the corresponding columns
            df.drop([climate_series[1]], axis=1, inplace=True)
        if not self.year_effects:
            # If the yearly fixed effects are not included, delete these columns from the dataframe
            df =  df.loc[:, ~df.columns.str.startswith('1')]
            df =  df.loc[:, ~df.columns.str.startswith('2')]
        # Remove the economic series, except for the gdp, from the data
        df.drop([economic_series[1], economic_series[2], economic_series[3], economic_series[4]], axis=1, inplace=True)
        if not self.countries:
            # If the regional fixed effects are not applied, delete these dummy columns. The index depends on the squared temperature boolean
            if self.ta_2:
                df.drop(df.iloc[:, 4:], axis=1, inplace=True)
            else:
                df.drop(df.iloc[:, 3:], axis=1, inplace=True)
        else:
            # If the countries have fixed effects, ensure that the dummies are correct and every observation belongs to one country
            countries_names_series = [economic_series[0], climate_series[0], climate_series[1], climate_series[2]]
            for country in countries_names:
                countries_names_series.append(country)
            for t in range(self.prev_years):
                countries_names_series.append(str(2023 - self.prev_years + t))
            df = df[df.columns.intersection(countries_names_series)]
            if self.ta_2:
                countries_dummies = df.values[:, 4:]
            else:
                countries_dummies = df.values[:, 3:]
            sum_dummies = np.sum(countries_dummies, axis=1)
            for s, dum in enumerate(sum_dummies):
                if dum == 0:
                    df.drop(index=s, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.dropna(axis=0, inplace=True)
        df = df.loc[:, (df != 0).any(axis=0)]
        return df

    def perform_regression(self, df: pd.DataFrame):
        """ Estimates the regression coefficients of the damage function

        Parameters
        ----------
        df: pd.DataFrame
            dataframe that contains the regression data
        """

        df = sm_api.add_constant(df, prepend=False)
        if self.newey_west:
            # If the newey_west standard errors are applied, include them in the regression
            model = sm_api.OLS(df.iloc[:, 0], df.iloc[:, 1:]).fit(use_t=True, cov_type='HAC', cov_kwds={'maxlags': self.lags})
        else:
            model = sm_api.OLS(df.iloc[:, 0], df.iloc[:, 1:]).fit()
        if self.print_res:
            print(model.summary())
        self.model = model
        self.coefs = model.params.values
        self.ses = model.bse
        self.pvals = model.pvalues
        self.resids = model.resid
        if self.print_res:
            # If the print_res boolean is True, print the inner autocorrelations of the residuals to determine the optimal number of lags for newey-west
            for i in range(1,5):
                print(np.corrcoef(self.resids[:len(self.resids)-i], self.resids[i:]))

    def return_result(self, stat: float, p_value: float) -> Tuple[bool, float, float]:
        """ Returns the result of the individual tests in the same format with a 5%-significance level

        Parameters
        ----------
        stat: float
            the test statistic
        p_value: float
            the p-value of the test statistic
        """

        if p_value < 0.05:
            return False, stat, p_value
        else:
            return True, stat, p_value

    def t_test(self):
        """ Performs the t-test
        """

        mean = np.mean(self.resids)
        std_dev = np.sqrt(np.var(self.resids))
        t_stat = mean/std_dev
        p_value = 2*(1-stats.t.cdf(t_stat, len(self.resids)-1))
        self.test_res, self.stats, self.pvals = self.return_result(t_stat, p_value)

    def endogeneity_test(self, df: pd.DataFrame):
        """ Performs the endogeneity test among the columns with the pearson correlation statistic test

        Parameters
        ----------
        df: pd.DataFrame
            dataframe that contains the regression data
        """

        corr_stats = []
        corr_res = []
        for col in df.columns:
            if col == 'Constant' or col == 'GDP':
                continue
            corr = np.corrcoef(self.resids, df.loc[:, col])[0][1]
            z_stat = corr / np.sqrt(np.square(1-np.square(corr))) * np.sqrt(len(self.resids)-1)
            p_value = 1-stats.norm.cdf(z_stat, len(self.resids)-1)
            corr_stats.append([z_stat, p_value])
            if p_value < 0.05:
                if self.print_res:
                    print('corr', col)
                corr_res.append(1)
            else:
                corr_res.append(0)
        try:
            p_value = sm_descrstats.sign_test(corr_res, mu0=0)
        except ValueError:
            p_value = 1
        corr_stats = np.array(corr_stats)
        if p_value < 0.05:
            self.test_res, self.stats, self.pvals =  False, np.mean(corr_stats[:, 0]), np.mean(corr_stats[:, 1])
        else:
            self.test_res, self.stats, self.pvals = True, np.mean(corr_stats[:, 0]), np.mean(corr_stats[:, 1])

    def serial_correlation_test(self):
        """ Performs the breusch-godfrey serial correlation test
        """

        res = sm_diag.acorr_breusch_godfrey(self.model)
        bg_stat = res[0]
        p_value = res[1]
        self.test_res, self.stats, self.pvals = self.return_result(bg_stat, p_value)

    def heteroskedasticity_test(self, df: pd.DataFrame):
        """ Performs the heteroskedasticity test

        Parameters
        ----------
        df: pd.DataFrame
            dataframe that contains the regression data
        """

        try:
            res = sm_diag.het_white(self.resids, df.iloc[:, 1:9])
        except AssertionError:
            res = sm_diag.het_breuschpagan(self.resids, df.iloc[:, 1:9], robust=True)
        white_stat = res[0]
        p_value = res[1]
        self.test_res, self.stats, self.pvals = self.return_result(white_stat, p_value)
        
    def normality_test(self):
        """ Performs the jarque-bera normality test
        """

        res = sm_stattools.jarque_bera(self.resids)
        jb_stat = res[0]
        p_value = res[1]
        self.test_res, self.stats, self.pvals = self.return_result(jb_stat, p_value)

    def multicollinearity_test(self, df: pd.DataFrame):
        """ Estimates the Variance-Inflation-Factor of the data with attention to dummies and squared temperature.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe that contains the regression data
        """

        vif_stats = []
        vif_res = []
        vif_df = df.iloc[:, self.countries+1:]
        for c, col in enumerate(vif_df.columns[1:]):
            vif = sm_outliers.variance_inflation_factor(vif_df.values, c)
            vif_stats.append(vif)
            if vif > 5:
                if self.print_res:
                    print('vif > 5:', col, vif)
            elif vif > 10:
                vif_res.append(1)
            else:
                vif_res.append(0)
        try:
            p_value = sm_descrstats.sign_test(vif_res, mu0=0)[1]
        except ValueError:
            p_value = 1
        vif_stats = np.array(vif_stats)
        if p_value < 0.05:
            self.test_res, self.stats, self.pvals = False, np.mean(vif_stats), p_value
        else:
            self.test_res, self.stats, self.pvals = True, np.mean(vif_stats), p_value