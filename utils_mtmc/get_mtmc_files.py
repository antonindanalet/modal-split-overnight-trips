import pandas as pd
from pathlib import Path
import yaml
config_path = Path('config.yml')
CONFIG = yaml.safe_load(open(config_path))


def get_zp(year, selected_columns=None):

    if year == 2015:
        path_2_zielpersonen = Path(CONFIG['path_to_zp'])
        if path_2_zielpersonen.exists():
            with open(path_2_zielpersonen, 'r', encoding='latin1') as zielpersonen_file:
                if selected_columns is None:
                    df_zp = pd.read_csv(zielpersonen_file)
                else:
                    df_zp = pd.read_csv(zielpersonen_file,
                                        dtype={'HHNR': int},
                                        usecols=selected_columns)
                return df_zp
        else:
            raise Exception('File "zielpersonen.csv" not in the folder "data/input". Please copy it there.')
    else:
        raise Exception('Year not well defined')


def get_overnight_trips(year, selected_columns=None):
    if year == 2015:
        path_2_reisenmueb = Path(CONFIG['path_to_overnight_trips'])
        if path_2_reisenmueb.exists():
            with open(path_2_reisenmueb, 'r', encoding='latin1') as trips_with_overnight_file:
                df_trips_with_overnight = pd.read_csv(trips_with_overnight_file,
                                                      delimiter=',',
                                                      usecols=selected_columns)
                return df_trips_with_overnight
        else:
            Exception('File "reisenmueb.csv" not in the folder "data/input". Please copy it there.')
    else:
        raise Exception('Year not well defined')