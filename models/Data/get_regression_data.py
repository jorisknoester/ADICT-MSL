from typing import List
import pandas as pd
import numpy as np
import os

def main():
    """ Converts the combined historical data file to an excel sheet that can be directly used in a regression.
    """
    PATH = os.getcwd()
    PATH += '\\Data\\Historical data.xlsx'
    final_df = pd.read_excel(PATH, sheet_name='Tabel', header=0)
    countries = final_df.columns[9:]
    economic_df = pd.read_excel(PATH, sheet_name='Economic data')
    climate_df = pd.read_excel(PATH, sheet_name='Climate data', index_col=0)
    ta, ta_2, msl, msl_2 = climate_df['Temperature'].values, climate_df['Temperature_2'].values, climate_df['Mean sea level'].values, climate_df['Mean sea level_2'].values
    final_df = match_data(economic_df, countries, ta, ta_2, msl, msl_2, final_df)
    final_df.dropna(subset=['GDP'], inplace=True)
    final_df.to_excel(os.getcwd() + '\\Data\\Regression data.xlsx', sheet_name='Data', index=False)

def match_data(data: pd.DataFrame, countries: List[str], ta: np.array, ta_2: np.array, msl: np.array, msl_2: np.array, final_df: pd.DataFrame) -> pd.DataFrame:
    """ Matches the individual countries of the different data structures and ensures that every observation for one specific country starts from the same year and ends in the same year.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe that contains the unstructured data
    countries: List[str]
        List with the names of the countries
    ta: np.array
        Array with the historical global temperature
    ta_2: np.array
        Array with the squared historical global temperature
    msl: np.array
        Array with the historical global mean sea level
    msl_2: np.array
        Array with the squared historical global mean sea level
    final_df: pd.DataFrame
        The result dataframe that is filled with the data
    """
    start_years = np.full((len(countries)), np.inf)
    countries_assets = np.zeros((len(countries)))
    columns = data.columns
    for c, country in enumerate(countries):
        # For each country find the start year in the data
        for i in range(len(data)):
            if country == data.loc[i, 'Country']:
                countries_assets[c] += 1
                values = np.array(data.iloc[i, 6:].values, dtype=np.float16)
                start_years[c] = min(start_years[c], int(columns[6 + np.where(~np.isnan(values))[0][0]]))
    start_years = np.array(start_years, dtype=np.int64)
    countries_assets = np.array(countries_assets, dtype=np.int64)
    asset_classes = data.columns[1:6]
    max_index = 0
    row = 0
    for c, country in enumerate(countries):
        # For evet country fill in the required data and interpolate if there is data missing. A check has already ocurred that ensures that big data gaps are avoided.
        values = data.iloc[row: row + countries_assets[c], 6:]
        values = values.interpolate(axis=1)
        years = 2022 - start_years[c] + 1
        for y in range(years):
            final_df.loc[y + max_index, 'Temperature'] = ta[start_years[c]-1980 + y]
            final_df.loc[y + max_index, 'Temperature_2'] = ta_2[start_years[c]-1980 + y]
            final_df.loc[y + max_index, 'Mean sea level'] = msl[start_years[c]-1980 + y]
            final_df.loc[y + max_index, 'Mean sea level_2'] = msl_2[start_years[c]-1980 + y]
            final_df.iloc[y + max_index, 9:] = 0
            final_df.loc[y + max_index, country] = 1
            for c_a in range(countries_assets[c]):
                for asset in asset_classes:
                    if data.loc[row + c_a, asset] == 1:
                        final_df.loc[y + max_index, asset] = values.iloc[c_a, start_years[c]-1980 + y]
        max_index += years
        row += countries_assets[c]
    return final_df


if __name__ == "__main__":
    main()

