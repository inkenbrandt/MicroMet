# Written By Kat Ladig and Paul Inkenbrandt

import numpy as np
import pandas as pd
import pathlib
import configparser
import datetime

try:
    config = configparser.ConfigParser()
    config.read('../secrets/config.ini')
    passwrd = config['DEFAULT']['pw']
    ip = config['DEFAULT']['ip']
    login = config['DEFAULT']['login']
except KeyError:
    print('credentials needed')

default = ['TIMESTAMP_START',
           'TIMESTAMP_END',
           'CO2',
           'CO2_SIGMA',
           'H2O',
           'H2O_SIGMA',
           'FC',
           'FC_SSITC_TEST',
           'LE',
           'LE_SSITC_TEST',
           'ET',
           'ET_SSITC_TEST',
           'H',
           'H_SSITC_TEST',
           'G',
           'SG',
           'FETCH_MAX',
           'FETCH_90',
           'FETCH_55',
           'FETCH_40',
           'WD',
           'WS',
           'WS_MAX',
           'USTAR',
           'ZL',
           'TAU',
           'TAU_SSITC_TEST',
           'MO_LENGTH',
           'U',
           'U_SIGMA',
           'V',
           'V_SIGMA',
           'W',
           'W_SIGMA',
           'PA',
           'TA_1_1_1',
           'RH_1_1_1',
           'T_DP_1_1_1',
           'TA_1_1_2',
           'RH_1_1_2',
           'T_DP_1_1_2',
           'TA_1_1_3',
           'RH_1_1_3',
           'T_DP_1_1_3',
           'TA_1_1_4',
           'VPD',
           'T_SONIC',
           'T_SONIC_SIGMA',
           'PBLH',
           'TS_1_1_1',
           'TS_1_1_2',
           'SWC_1_1_1',
           'SWC_1_1_2',
           'ALB',
           'NETRAD',
           'SW_IN',
           'SW_OUT',
           'LW_IN',
           'LW_OUT',
           'P']

math_soils = ["VWC_5cm_N_Avg", "VWC_5cm_S_Avg", "Ka_5cm_N_Avg", "T_5cm_N_Avg", "BulkEC_5cm_N_Avg", "VWC_10cm_N_Avg",
              "Ka_10cm_N_Avg", "T_10cm_N_Avg", "BulkEC_10cm_N_Avg", "VWC_20cm_N_Avg", "Ka_20cm_N_Avg", "T_20cm_N_Avg",
              "BulkEC_20cm_N_Avg", "VWC_30cm_N_Avg", "Ka_30cm_N_Avg", "T_30cm_N_Avg", "BulkEC_30cm_N_Avg",
              "VWC_40cm_N_Avg", "Ka_40cm_N_Avg", "T_40cm_N_Avg", "BulkEC_40cm_N_Avg", "VWC_50cm_N_Avg", "Ka_50cm_N_Avg",
              "T_50cm_N_Avg", "BulkEC_50cm_N_Avg", "VWC_60cm_N_Avg", "Ka_60cm_N_Avg", "T_60cm_N_Avg",
              "BulkEC_60cm_N_Avg", "VWC_75cm_N_Avg", "Ka_75cm_N_Avg", "T_75cm_N_Avg", "BulkEC_75cm_N_Avg",
              "VWC_100cm_N_Avg", "Ka_100cm_N_Avg", "T_100cm_N_Avg", "BulkEC_100cm_N_Avg", "Ka_5cm_S_Avg", "T_5cm_S_Avg",
              "BulkEC_5cm_S_Avg", "VWC_10cm_S_Avg", "Ka_10cm_S_Avg", "T_10cm_S_Avg", "BulkEC_10cm_S_Avg",
              "VWC_20cm_S_Avg", "Ka_20cm_S_Avg", "T_20cm_S_Avg", "BulkEC_20cm_S_Avg", "VWC_30cm_S_Avg", "Ka_30cm_S_Avg",
              "T_30cm_S_Avg", "BulkEC_30cm_S_Avg", "VWC_40cm_S_Avg", "Ka_40cm_S_Avg", "T_40cm_S_Avg",
              "BulkEC_40cm_S_Avg", "VWC_50cm_S_Avg", "Ka_50cm_S_Avg", "T_50cm_S_Avg", "BulkEC_50cm_S_Avg",
              "VWC_60cm_S_Avg", "Ka_60cm_S_Avg", "T_60cm_S_Avg", "BulkEC_60cm_S_Avg", "VWC_75cm_S_Avg", "Ka_75cm_S_Avg",
              "T_75cm_S_Avg", "BulkEC_75cm_S_Avg", "VWC_100cm_S_Avg", "Ka_100cm_S_Avg", "T_100cm_S_Avg"]

math_soils_v2 = ["SWC_3_1_1", "SWC_4_1_1", "K_3_1_1", "TS_3_1_1", "EC_3_1_1", "SWC_3_2_1", "K_3_2_1", "TS_3_2_1",
                 "EC_3_2_1", "SWC_3_3_1", "K_3_3_1", "TS_3_3_1", "EC_3_3_1", "SWC_3_4_1", "K_3_4_1", "TS_3_4_1",
                 "EC_3_4_1", "SWC_3_5_1", "K_3_5_1", "TS_3_5_1", "EC_3_5_1", "SWC_3_6_1", "K_3_6_1", "TS_3_6_1",
                 "EC_3_6_1", "SWC_3_7_1", "K_3_7_1", "TS_3_7_1", "EC_3_7_1", "SWC_3_8_1", "K_3_8_1", "TS_3_8_1",
                 "EC_3_8_1", "SWC_3_9_1", "K_3_9_1", "TS_3_9_1", "EC_3_9_1", "K_4_1_1", "TS_4_1_1", "EC_4_1_1",
                 "SWC_4_2_1", "K_4_2_1", "TS_4_2_1", "EC_4_2_1", "SWC_4_3_1", "K_4_3_1", "TS_4_3_1", "EC_4_3_1",
                 "SWC_4_4_1", "K_4_4_1", "TS_4_4_1", "EC_4_4_1", "SWC_4_5_1", "K_4_5_1", "TS_4_5_1", "EC_4_5_1",
                 "SWC_4_6_1", "K_4_6_1", "TS_4_6_1", "EC_4_6_1", "SWC_4_7_1", "K_4_7_1", "TS_4_7_1", "EC_4_7_1",
                 "SWC_4_8_1", "K_4_8_1", "TS_4_8_1", "EC_4_8_1", "EC_4_9_1", "SWC_4_9_1", "K_4_9_1", "TS_4_9_1",
                 "TS_1_1_1", "TS_2_1_1", "SWC_1_1_1", "SWC_2_1_1"]

well_soils = ["SWC_3_1_1", "SWC_4_1_1", "K_3_1_1", "TS_3_1_1", "EC_3_1_1",
              "SWC_3_2_1", "K_3_2_1", "TS_3_2_1", "EC_3_2_1",
              "SWC_3_3_1", "K_3_3_1", "TS_3_3_1", "EC_3_3_1",
              "SWC_3_4_1", "K_3_4_1", "TS_3_4_1", "EC_3_4_1",
              "SWC_3_5_1", "K_3_5_1", "TS_3_5_1", "EC_3_5_1",
              "SWC_3_6_1", "K_3_6_1", "TS_3_6_1", "EC_3_6_1",
              "SWC_3_7_1", "K_3_7_1", "TS_3_7_1", "EC_3_7_1",
              "SWC_3_8_1", "K_3_8_1", "TS_3_8_1", "EC_3_8_1",
              "SWC_3_9_1", "K_3_9_1", "TS_3_9_1", "EC_3_9_1", ]

bflat = list(filter(lambda item: item not in ('TA_1_1_4', 'TS_1_1_2', 'SWC_1_1_2'), default))
wellington = list(filter(lambda item: item not in 'TS_1_1_1', default))
big_math = wellington[:-10] + math_soils + wellington[-10:] + ['T_CANOPY']
big_math_v2 = wellington[:-10] + math_soils_v2 + wellington[-7:] + ['T_CANOPY']
big_math_v2_filt = list(filter(lambda item: item not in 'TA_1_1_4', big_math_v2))

big_well = list(filter(lambda item: item not in 'TA_1_1_4', default)) + well_soils
header_dict = {60: default, 57: bflat, 59: wellington, 96: big_well, 131: big_math, 132: big_math_v2_filt}


def check_header(csv_file):
    """
    Check if the given CSV file has a header row with the column name "TIMESTAMP_START".

    :param csv_file: The path to the CSV file.
    :type csv_file: str
    :return: True if the CSV file has a header row with column name "TIMESTAMP_START", False otherwise.
    :rtype: bool
    """
    df = pd.read_csv(csv_file, header=None, nrows=1)
    # If the first row contains "TIMESTAMP_START", assume it's a header row.
    if "TIMESTAMP_START" in df.iloc[0].values:
        return 1
    elif 'TOA5' in df.iloc[0].values:
        return 2
    else:
        return False


def dataframe_from_file(file):
    """
    :param file: File path of the csv file to read the data from.
    :return: pandas DataFrame containing the data read from the file.

    This method reads data from a csv file and returns a pandas DataFrame.
    If the file has a header, it is assumed that the first row contains column names.
    If the file does not have a header, the number of columns in the first row is counted
    and matched with a corresponding header from a predefined dictionary.

    If any error occurs during the process, an error message is printed and None is returned.

    Example usage:
        >>> df = dataframe_from_file('data.csv')
        >>> print(df.head())

    """
    try:
        na_vals = ['-9999', 'NAN', 'NaN','nan']
        # check to see if a header is present, if not, assign one using the header_dict dictionary
        if check_header(file) == 1:

            df = pd.read_csv(file, na_values=na_vals)
        elif check_header(file) == 2:

            df = pd.read_csv(file, na_values=na_vals, skiprows=[0, 2, 3])
            print(file)
            if "TIMESTAMP" in df.columns:
                df.drop(['TIMESTAMP'], axis=1, inplace=True)
        else:
            # count the number of columns in the first row of headerless data
            first_line = pd.read_csv(file, header=None, nrows=1, na_values=na_vals)
            fll = len(first_line.columns)
            # match header count (key) to key in dictionary
            header = header_dict[fll]
            df = pd.read_csv(file, na_values=na_vals, names=header)
        return df
    # If no header match is found, an exception occurs
    except Exception as e:
        print(f'Encountered an error with file {file}: {str(e)}')
        return None

def is_int(element: any) -> bool:
    #If you expect None to be passed:
    if element is None:
        return False
    try:
        int(element)
        return True
    except ValueError:
        return False

def raw_file_compile(raw_fold, station_folder_name, search_str="*Flux_AmeriFluxFormat*.dat"):
    """
    Compiles raw AmeriFlux datalogger files into a single dataframe.

    :param raw_fold: Path to the root folder of raw datalogger files
    :type raw_fold: pathlib.Path
    :param station_folder_name: Name of the station folder containing the raw datalogger files
    :type station_folder_name: str
    :return: Dataframe containing compiled AmeriFlux data, or None if no valid files found
    :rtype: pandas.DataFrame or None
    """
    amflux = {}
    station_folder = raw_fold / station_folder_name
    # iterate through specified folder of raw datalogger files (.dat); Match to AmeriFlux datalogger files
    for file in station_folder.rglob(search_str):
        # get the base number of the raw ameriflux file
        baseno = file.name.split(".")[0]

        if baseno.split("_")[-1][0] == 'A':
            file_number = 9999
        elif baseno.split("_")[-1][0] == '(':
            file_number = 9999
        elif is_int(baseno.split("_")[-1]):
            file_number = int(baseno.split("_")[-1])
        else:
            file_number = 9999
        if file_number >= 0:
            df = dataframe_from_file(file)
            if df is not None:
                #print(file)
                amflux[baseno] = df
        else:
            print("Error: File number is too high")
    if amflux:
        # concat dataframes that were successfully read in
        et_data = pd.concat(amflux, axis=0).reset_index()
        et_data = et_data.drop(columns=['level_0', 'level_1'])
    else:
        et_data = None
    return et_data


def remove_extra_soil_params(df):
    """
    Removes extra soil parameters from the given dataframe.

    :param df: A pandas dataframe containing soil parameters.
    :type df: pandas.DataFrame
    :return: The input dataframe with extra soil parameters removed.
    :rtype: pandas.DataFrame
    """
    # get a list of columns and split them into parts (parameter number number number)
    for col in df.columns:
        collist = col.split('_')
        main_var = collist[0]

        # get rid of columns that don't follow the typical pattern
        if len(collist) > 3 and collist[2] not in ['N', 'S']:
            depth_var = int(collist[2])
            if main_var in ['SWC', 'TS', 'EC', 'K'] and (depth_var >= 1 and int(collist[1]) >= 3):
                df.drop(col, axis=1, inplace=True)
        # drop cols from a specified list math_soils_v2
        elif col in math_soils_v2[:-4]:
            df.drop(col, axis=1, inplace=True)
        elif main_var in ['VWC', 'Ka'] or 'cm_N' in col or 'cm_S' in col:
            df.drop(col, axis=1, inplace=True)
    return df


class Reformatter(object):
    """
    Class for reformatting raw Utah Flux Network data into the acceptable Ameriflux format.

    Methods:
        __init__ : Initialize the Reformatter object.
        datefixer : Fixes the date and time format in the given data.
        update_dataframe_column : Replaces an old column with a new one in the DataFrame.
        name_changer : Changes column names in the DataFrame based on a dictionary of column name mappings.
        scale_and_convert : Scales and converts values in a column to float type.
        ssitc_scale : Scales the values in the SSITC columns of the DataFrame.
    """
    # Variables that are important, but do not have an int following the underscore
    othervar = [
        'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT', 'T_SONIC', 'T_CANOPY', 'FETCH_MAX', 'FETCH_90', 'FETCH_55',
        'FC_SSITC_TEST', 'ET_SSITC_TEST', 'LE_SSITC_TEST', 'H_SSITC_TEST', 'TAU_SSITC_TEST', 'CO2_SIGMA',
        'T_SONIC_SIGMA', 'H2O_SIGMA', 'WS_MAX', 'MO_LENGTH', 'U_SIGMA', 'V_SIGMA', 'W_SIGMA'
    ]

    # dictionary to fix naming convention issues with EasyFluxDL;
    # https://ameriflux.lbl.gov/wp-content/uploads/2015/10/AmeriFlux_DataVariables.pdf
    col_name_match = {"TA_1_1_2": "TA_1_2_1",
                      "RH_1_1_2": "RH_1_2_1",
                      "T_DP_1_1_2": "T_DP_1_2_1",
                      "TA_1_1_3": "TA_1_3_1",
                      "RH_1_1_3": "RH_1_3_1",
                      "T_DP_1_1_3": "T_DP_1_3_1",
                      "TA_2_1_1": "TA_1_2_1",
                      "RH_2_1_1": "RH_1_2_1",
                      "T_DP_2_1_1": "T_DP_1_2_1",
                      "TA_3_1_1": "TA_1_3_1",
                      "RH_3_1_1": "RH_1_3_1",
                      "TA_1_1_4": "TA_1_4_1",
                      "T_DP_3_1_1": "T_DP_1_3_1",
                      "PBLH": "PBLH_F",
                      "TS_1_1_2": "TS_2_1_1",
                      "SWC_1_1_2": "SWC_2_1_1"}

    # Variables to despike
    despikey = ['CO2', 'H2O', 'FC', 'LE', 'ET', 'H', 'G', 'SG', 'FETCH_MAX', 'FETCH_90', 'FETCH_55', 'FETCH_40',
                'WS',
                'USTAR', 'TAU', 'MO_LENGTH', 'U', 'V', 'W', 'PA', 'TA_1_1_1', 'RH_1_1_1', 'T_DP_1_1_1',
                'TA_1_2_1', 'RH_1_2_1', 'T_DP_1_2_1', 'TA_1_3_1', 'RH_1_3_1', 'T_DP_1_3_1', 'VPD', 'T_SONIC',
                'PBLH', 'TS_1_1_1', 'TS_2_1_1', 'SWC_1_1_1', 'SWC_2_1_1', 'ALB', 'NETRAD', 'SW_IN', 'SW_OUT',
                'LW_IN', 'LW_OUT']

    def __init__(self, et_data, drop_soil=True, data_path = None):
        # read in variable limits
        if data_path is None:
            try:
                data_path = pathlib.Path('../data/FP_variable_20220810.csv')
                self.varlimits = pd.read_csv(data_path, index_col='Name')
            except FileNotFoundError:
                try:
                    data_path = pathlib.Path('data/FP_variable_20220810.csv')
                    self.varlimits = pd.read_csv(data_path, index_col='Name')
                except FileNotFoundError:
                    data_path = pathlib.Path(
                        '/content/drive/Shareddrives/UGS_Flux/Data_Processing/Jupyter_Notebooks/Micromet/data/FP_variable_20220810.csv')
                    self.varlimits = pd.read_csv(data_path, index_col='Name')
        else:
            data_path = pathlib.Path(data_path)

        # fix datetimes
        self.et_data = self.datefixer(et_data)

        # change variable names
        self.name_changer()

        # set variable limits
        self.et_data = self.extreme_limiter(self.et_data)

        # despike variables
        for var in self.despikey:
            if var in self.et_data.columns:
                self.et_data[var] = self.despike(self.et_data[var])

        # switch tau sign
        self.tau_fixer()

        # turn decimals in SWC to percent
        self.fix_swc_percent()

        # rescale the quality values to a 0-2 scale
        self.ssitc_scale()

        self.et_data = self.et_data.fillna(value=int(-9999))
        #self.et_data = self.et_data.dropna(subset=['TIMESTAMP_START', 'TIMESTAMP_END'])

        self.et_data = self.et_data.drop(['datetime_end'], axis=1)

        if drop_soil:
            self.et_data = remove_extra_soil_params(self.et_data)

        # cdf.drop('station',axis=1,inplace=True)
        for col in self.et_data:
            if col in ['MO_LENGTH']:
                self.et_data[col] = self.et_data[col].astype(np.float64)
            elif col in ['TIMESTAMP_START', 'TIMESTAMP_END', 'RECORD']:
                self.et_data[col] = self.et_data[col].astype(np.int64)
            elif "SSITC" in col:
                self.et_data[col] = self.et_data[col].astype(np.int16)
            else:
                self.et_data[col] = self.et_data[col].astype(np.float32)

    def datefixer(self, et_data):
        """
        Fixes the date and time format in the given data.

        :param et_data: A pandas DataFrame containing the data to be fixed.
        :return: The fixed pandas DataFrame.

        The `datefixer` method takes a pandas DataFrame `et_data` as input and performs the following operations to fix the date and time format:
        1. Converts the 'TIMESTAMP_START' and 'TIMESTAMP_END' columns to datetime format using the format "%Y%m%d%H%M" and assigns them to 'datetime_start' and 'datetime_end' columns in `et_data`.
        2. Removes duplicate rows in `et_data` based on the 'TIMESTAMP_START' and 'TIMESTAMP_END' columns.
        3. Sets the 'datetime_start' column as the index of `et_data` and sorts the DataFrame based on this index.
        4. Removes any duplicate rows in `et_data` based on the 'datetime_start' column, keeping only the first occurrence.
        5. Resamples the DataFrame at 30-minute intervals and interpolates missing values using linear method.
        6. Returns the fixed pandas DataFrame.

        Example usage:
        ```python
        import pandas as pd

        # Create the input DataFrame et_data
        et_data = pd.DataFrame({
            'TIMESTAMP_START': ['202201011200', '202201011300', '202201020900'],
            'TIMESTAMP_END': ['202201011400', '202201011500', '202201021000']
        })

        # Create an instance of the class containing the datefixer method
        date_fixer = DateFixer()

        # Call the datefixer method to fix the date and time format in et_data
        fixed_data = date_fixer.datefixer(et_data)

        print(fixed_data)
        ```
        """
        # create datetime fields to conduct datetime operations on dataset
        et_data['datetime_start'] = pd.to_datetime(et_data['TIMESTAMP_START'], format="%Y%m%d%H%M")
        et_data['datetime_end'] = pd.to_datetime(et_data['TIMESTAMP_END'], format="%Y%m%d%H%M")
        et_data = et_data.drop_duplicates(subset=['TIMESTAMP_START', 'TIMESTAMP_END'])
        et_data = et_data.set_index(['datetime_start']).sort_index()

        # drop rows with duplicate datetimes
        et_data = et_data[~et_data.index.duplicated(keep='first')]

        # eliminate implausible dates that are set in the future
        et_data = et_data[et_data.index <= datetime.datetime.today() + pd.Timedelta(days=1)]

        # fix time offsets to harmonize sample frequency to 30 min
        et_data = et_data.resample('15min').asfreq().interpolate(method='linear', limit=2).resample(
            '30min').asfreq()

        # remake timestamp marks to match datetime index, filling in NA spots
        et_data['TIMESTAMP_START'] = et_data.index.strftime('%Y%m%d%H%M').astype(np.int64)
        et_data['TIMESTAMP_END'] = (et_data.index + pd.Timedelta('30min')).strftime('%Y%m%d%H%M').astype(np.int64)
        return et_data

    def update_dataframe_column(self, new_column, old_column):
        """
        Given the old column name and the new column name, replace the old column with
        the new one in the DataFrame self.et_data.
        The new column will be a combination of the maximum values of the old column and itself.
        """
        self.et_data[new_column] = self.et_data[[old_column, new_column]].max(axis=1)
        self.et_data = self.et_data.drop(old_column, axis=1)

    def name_changer(self):
        """
        Changes column names in the DataFrame based on the given dictionary of column name mappings.
        :return: None
        """
        for old_column, new_column in self.col_name_match.items():
            if old_column in self.et_data.columns:
                if new_column in self.et_data.columns:
                    self.update_dataframe_column(new_column, old_column)
                else:
                    self.et_data[new_column] = self.et_data[old_column]
                    self.et_data = self.et_data.drop(old_column, axis=1)

    def scale_and_convert(self, column):
        """
        Scale the values and then convert them to float type
        :param column: Pandas series or dataframe column
        :return: Scaled and converted dataframe column
        """
        column = column.apply(lambda i: self.rating(i))
        return pd.to_numeric(column, downcast='float')

    def ssitc_scale(self):
        """
        Scale the values in the SSITC columns of et_data dataframe.
        :return: None
        """
        ssitc_columns = ['FC_SSITC_TEST', 'LE_SSITC_TEST', 'ET_SSITC_TEST', 'H_SSITC_TEST', 'TAU_SSITC_TEST']
        for column in ssitc_columns:
            if column in self.et_data.columns:
                if (self.et_data[column] > 3).any():
                    self.et_data[column] = self.scale_and_convert(self.et_data[column])

    @staticmethod
    def rating(x):
        """
        Convert a value into a rating category.

        :param x: The value to be converted.
        :type x: int or float
        :return: The rating category based on the value.
        :rtype: int

        The `rating` method takes a numeric value, `x`, as input and categorizes it into one of
        three rating categories. For values less than or equal to 3, the method returns 0. For values
        between 4 and 6 (inclusive), the method returns 1. For all other values, the method returns 2.
        """
        if 0 <= x <= 3:
            x = 0
        elif 4 <= x <= 6:
            x = 1
        else:
            x = 2
        return x

    def _extract_variable_name(self, variable):
        """
        Extracts the variable name based on given variable.

        :param variable: The variable to extract the variable name from.
        :type variable: str
        :return: The extracted variable name.
        :rtype: str
        """
        if variable in self.othervar:
            varlimvar = variable
        elif any(variable in x for x in self.othervar):
            temp = variable.split('_')[:2]
            varlimvar = '_'.join(temp)
        else:
            varlimvar = variable.split("_")[0]

        return varlimvar

    def replace_out_of_range_with_nan(self, df, variable, replace_w=np.nan):
        """
        Replace values in a specified column with np.nan if they exceed a given range.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name where the replacement should be applied.
        lower_bound (float): The lower bound of the range.
        upper_bound (float): The upper bound of the range.

        Returns:
        pd.DataFrame: The DataFrame with values replaced by np.nan.
        """

        varlimvar = self._extract_variable_name(variable)
        if varlimvar in self.varlimits.index:
            if variable in df.columns:
                upper_bound = self.varlimits.loc[varlimvar, 'Max']
                lower_bound = self.varlimits.loc[varlimvar, 'Min']
                df[variable] = pd.to_numeric(df[variable], errors='coerce')
                df[variable] = df[variable].apply(lambda x: replace_w if x < lower_bound or x > upper_bound else x)
        return df

    def extreme_limiter(self, df, replace_w=np.nan):
        """
        :param df: pandas DataFrame object containing the data to be processed
        :param replace_w: value to replace the extreme values with (default is np.nan)
        :return: None

        This method takes a pandas DataFrame object (df) and a value (replace_w) as parameters. It iterates through each column of the DataFrame and converts the values to numeric type using `pd.to_numeric` function, with errors set to 'coerce' to replace invalid values with NaN.

        The method then extracts the variable name from the column name using the `_extract_variable_name` method. Finally, it calls the `_check_and_replace` method to check for extreme values and replace them with the given value (replace_w).
        """
        for variable in df.columns:
            df = self.replace_out_of_range_with_nan(df, variable, replace_w)
        return df

    def despike(self, arr, nstd: float = 4.5):
        """Removes spikes from an array of values based on a specified deviation from the mean.

        Args:
            arr: array of values with spikes
            nstd: number of standard deviations from mean; default is 4.5

        Returns:
            Array of despiked values
        Notes:
            * Not windowed.
            * This method is fast but might be too agressive if the spikes are small relative to seasonal variability.
        """

        stdd = np.nanstd(arr) * nstd
        avg = np.nanmean(arr)
        avgdiff = stdd - np.abs(arr - avg)
        y = np.where(avgdiff >= 0, arr, np.nan)
        #nans, x = np.isnan(y), lambda z: z.nonzero()[0]
        #if len(x(~nans)) > 0:
        #    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    def tau_fixer(self):
        """
        Fixes the values in the 'TAU' column of the et_data dataframe by multiplying them by -1.

        :return: None
        """
        if "TAU" in self.et_data.columns:
            self.et_data['TAU'] = -self.et_data['TAU']

    def fix_swc_percent(self):
        """
        Applies a fix to the SWC columns in the et_data DataFrame.
        This method identifies columns containing "SWC" in their names,
        and checks whether the maximum value is less than 1.5. If it is, the
        values in the column are multiplied by 100, except for any values
        that are -9999.
        :return: None
        """
        for col in self.et_data.columns:
            if "SWC" in col and self.et_data[col].max() < 1.5:
                self.et_data.loc[self.et_data[col] > -9999, col] *= 100

    def timestamp_reset(self):
        self.et_data['TIMESTAMP_START'] = self.et_data.index
        self.et_data['TIMESTAMP_END'] = self.et_data.index + pd.Timedelta(minutes=30)

    def despike_ewma_fb(self, df_column, span, delta):
        """Apply forwards, backwards exponential weighted moving average (EWMA) to df_column.
        Remove data from df_spikey that is > delta from fbewma.

        Args:
            df_column: pandas Series of data with spikes
            span: size of window of spikes
            delta: threshold of spike that is allowable

        Returns:
            despiked data

        Notes:
            https://stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python
        """
        # Forward EWMA.
        fwd = pd.Series.ewm(df_column, span=span).mean()
        # Backward EWMA.
        bwd = pd.Series.ewm(df_column[::-1], span=span).mean()
        # Add and take the mean of the forwards and backwards EWMA.
        stacked_ewma = np.vstack((fwd, bwd[::-1]))
        np_fbewma = np.mean(stacked_ewma, axis=0)
        np_spikey = np.array(df_column)
        # np_fbewma = np.array(fb_ewma)
        cond_delta = (np.abs(np_spikey - np_fbewma) > delta)
        np_remove_outliers = np.where(cond_delta, np.nan, np_spikey)
        return np_remove_outliers


main_header_part = ['TIMESTAMP_START', 'TIMESTAMP_END', 'CO2', 'CO2_SIGMA', 'H2O', 'H2O_SIGMA', 'FC', 'FC_SSITC_TEST',
                    'LE',
                    'LE_SSITC_TEST', 'ET', 'ET_SSITC_TEST', 'H', 'H_SSITC_TEST', 'G', 'SG', 'FETCH_MAX', 'FETCH_90',
                    'FETCH_55',
                    'FETCH_40', 'WD', 'WS', 'WS_MAX', 'USTAR', 'ZL', 'TAU', 'TAU_SSITC_TEST', 'MO_LENGTH', 'U',
                    'U_SIGMA', 'V',
                    'V_SIGMA', 'W', 'W_SIGMA', 'PA', 'TA_1_1_1', 'RH_1_1_1', 'T_DP_1_1_1', ]

bet_part = ['TA_1_2_1', 'RH_1_2_1', 'T_DP_1_2_1',
            'TA_1_3_1', 'RH_1_3_1', 'T_DP_1_3_1', 'VPD', 'T_SONIC', 'T_SONIC_SIGMA', 'PBLH', 'TS_1_1_1', 'SWC_1_1_1',
            'ALB', 'NETRAD', 'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT', 'P']
bet_header = main_header_part + bet_part

met_headers = ["TIMESTAMP_START", "TIMESTAMP_END", "CO2", "CO2_SIGMA", "H2O", "H2O_SIGMA", "FC", "FC_SSITC_TEST", "LE",
               "LE_SSITC_TEST", "ET", "ET_SSITC_TEST", "H", "H_SSITC_TEST", "G", "SG", "FETCH_MAX", "FETCH_90",
               "FETCH_55",
               "FETCH_40", "WD", "WS", "WS_MAX", "USTAR", "ZL", "TAU", "TAU_SSITC_TEST", "MO_LENGTH", "U", "U_SIGMA",
               "V",
               "V_SIGMA", "W", "W_SIGMA", "PA", "TA_1_1_1", "RH_1_1_1", "T_DP_1_1_1", "TA_1_1_2", "RH_1_1_2",
               "T_DP_1_1_2",
               "TA_1_1_3", "RH_1_1_3", "T_DP_1_1_3", "VPD", "T_SONIC", "T_SONIC_SIGMA", "PBLH", "TS_1_1_1", "TS_1_1_2",
               "SWC_1_1_1", "SWC_1_1_2", "ALB", "NETRAD", "SW_IN", "SW_OUT", "LW_IN", "LW_OUT", "P"]

bet_spikey = ['CO2', 'H2O', 'FC', 'LE', 'ET', 'H', 'G', 'SG', 'FETCH_MAX', 'FETCH_90', 'FETCH_55', 'FETCH_40', 'WS',
              'USTAR',
              'TAU', 'MO_LENGTH', 'U', 'V', 'W', 'PA', 'TA_1_1_1', 'RH_1_1_1', 'T_DP_1_1_1',
              'TA_1_2_1', 'RH_1_2_1', 'T_DP_1_2_1', 'TA_1_3_1', 'RH_1_3_1', 'T_DP_1_3_1', 'VPD', 'T_SONIC',
              'T_SONIC_SIGMA',
              'PBLH_F', 'TS_1_1_1', 'SWC_1_1_1', 'ALB', 'NETRAD', 'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT']


def load_data():
    df = pd.read_csv('./data/FP_variable_20220810.csv')
    return df

def outfile(df, stationname, out_dir):
    """Generates a file name based on the ameriflux file nameing standards

    """
    first_index = pd.to_datetime(df['TIMESTAMP_START'][0], format ='%Y%m%d%H%M')
    last_index = pd.to_datetime(df['TIMESTAMP_END'][-1], format ='%Y%m%d%H%M')#df.index[-1], format ='%Y%m%d%H%M')
    filename = stationname  + f"_HH_{first_index.strftime('%Y%m%d%H%M')}_{last_index.strftime('%Y%m%d%H%M')}.csv" #{last_index_plus_30min.strftime('%Y%m%d%H%M')}.csv"
    df.to_csv(out_dir + stationname + "/" + filename)

if __name__ == '__main__':
    data = load_data()
