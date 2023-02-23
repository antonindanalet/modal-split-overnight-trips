# import pandas as pd
# import numpy as np
from utils_mtmc.get_mtmc_files import *
from mtmc2015.utils2015.compute_confidence_interval import get_weighted_avg_and_std


def run():
    # df_zp = get_zp_renamed()
    # select only people who were asked questions about trips with overnights (module 1b, encoded 2)
    # df_zp = df_zp[df_zp['module_attributed_to_the_respondent'] == 2]
    # df_zp.drop(columns=['module_attributed_to_the_respondent'], inplace=True)
    # select only people who said the number of trips with overnights they made
    # df_zp['with_trips'] = df_zp['nb_trips_with_overnights'] > 0
    # df_zp = df_zp[~df_zp['nb_trips_with_overnights'] < 0]
    # sum of weights of the declared trips, including those without details, in particular distance
    # weight_declared_trips = df_zp[df_zp['with_trips']]['WP'].sum()
    # select overnight trips whose distance is known
    df_overnight_trips = get_overnight_trips_in_2015_renamed()  # contains all trips with overnights
    df_overnight_trips = df_overnight_trips[df_overnight_trips['trip_distance'] >= 0]  # only trips with valid distance
    # only keep trips with final destination NOT Switzerland
    df_overnight_trips = df_overnight_trips[df_overnight_trips['destination_country'] != 8100]
    # only keep trips with known transport mode
    # df_overnight_trips = df_overnight_trips[(df_overnight_trips['main_transport_mode'] != -97) &
    #                                         (df_overnight_trips['main_transport_mode'] != -98)]
    # get number of detailed trips (with distance) by person
    df_overnight_trips_count_nb = df_overnight_trips[['HHNR', 'trip_distance']].groupby('HHNR').count()
    df_overnight_trips_count_nb = df_overnight_trips_count_nb.rename(columns={'trip_distance': 'nb_detailed_trips'})
    # df_zp = pd.merge(df_zp, df_overnight_trips_count_nb, left_on='HHNR', right_index=True, how='left')
    transport_mean2mode = {1: 'Autre',
                           2: 'Autre',
                           3: 'TIM',
                           4: 'TIM',
                           5: 'TIM',
                           6: 'TIM',
                           7: 'TIM',
                           8: 'TIM',
                           9: 'TP',
                           10: 'TP',
                           11: 'TP',
                           12: 'TP',
                           13: 'Autre',
                           14: 'Autocar',
                           15: 'Autre',
                           16: 'Autre',
                           17: 'Avion',
                           18: 'Autre',
                           19: 'Autre',
                           20: 'Autre',
                           21: 'Autre',
                           95: 'Autre',
                           -97: 'Autre',  # Is this correct ???
                           -98: 'Autre'}  # Is this correct ???
    df_overnight_trips['main_transport_mode_agg'] = df_overnight_trips['main_transport_mode'].map(transport_mean2mode)
    # Remove people who said they did a trip, but whose distance are not valid
    # For HHNR=333950, the distance is known, but not the country of destination.
    # For HHNR=488907, the destination is known but is the same as home.
    df_overnight_trips = df_overnight_trips[~df_overnight_trips['HHNR'].isin([333950, 488907])]  # Is this correct ??
    print('Basis:', len(df_overnight_trips),
          'overnight trips collected with details, in particular with a valid information about the distance, and '
          'whose destination is abroad.')
    """ --- """
    # get total distance per person by plane
    df_overnight_trips['trip_distance_by_plane'] = \
        df_overnight_trips['trip_distance'] * (df_overnight_trips['main_transport_mode_agg'] == 'Avion')
    df_overnight_trips['trip_distance_by_pt'] = \
        df_overnight_trips['trip_distance'] * (df_overnight_trips['main_transport_mode_agg'] == 'TP')
    df_overnight_trips['trip_distance_by_car'] = \
        df_overnight_trips['trip_distance'] * (df_overnight_trips['main_transport_mode_agg'] == 'TIM')
    df_overnight_trips['trip_distance_by_autocar'] = \
        df_overnight_trips['trip_distance'] * (df_overnight_trips['main_transport_mode_agg'] == 'Autocar')
    df_overnight_trips['trip_distance_by_other'] = \
        df_overnight_trips['trip_distance'] * (df_overnight_trips['main_transport_mode_agg'] == 'Autre')
    # # Sum of weights of the detailed trips, without those missing details, in particular distance
    # weight_detailed_trips = df_zp[df_zp['with_trips']]['WP'].sum()
    # # Correction factor for people declaring they did trips, but without detailing them
    # correction_factor_declared_detailed_trips = weight_declared_trips / weight_detailed_trips
    # df_zp['WP_corrected'] = np.where(df_zp['with_trips'],
    #                                  df_zp['WP'] * correction_factor_declared_detailed_trips,
    #                                  df_zp['WP'])
    # df_overnight_trips = pd.merge(df_overnight_trips, df_zp[['HHNR', 'WP_corrected']], on='HHNR', how='left')
    dict_column_weighted_avg_and_std, sample = get_weighted_avg_and_std(df_overnight_trips,
                                                                        weights='WP', percentage=True,
                                                                        list_of_columns=['trip_distance_by_plane',
                                                                                         'trip_distance_by_pt',
                                                                                         'trip_distance_by_car',
                                                                                         'trip_distance_by_autocar',
                                                                                         'trip_distance_by_other'])
    print(dict_column_weighted_avg_and_std)
    # {'trip_distance_by_plane': [0.8053367838633771, 0.008889927779091157],
    # 'trip_distance_by_pt': [0.033646414422791676, 0.00404860068056136],
    # 'trip_distance_by_car': [0.1402710874821314, 0.007797086742249514],
    # 'trip_distance_by_autocar': [0.01270778049473439, 0.0025149260950734055],
    # 'trip_distance_by_other': [0.008037933736965309, 0.0020048748170338566]}


def get_overnight_trips_in_2015_renamed():
    selected_columns = ['HHNR', 'WP', 'RENR', 'reisenr', 'f70801', 'f71300', 'f71400_01', 'f71600b',
                        'f70700_01', 'f71700b', 'RZ_LND']
    df_overnight_trips = get_overnight_trips(year=2015, selected_columns=selected_columns)
    # Rename variables
    df_overnight_trips = df_overnight_trips.rename(columns={'f71600b': 'trip_distance',
                                                            'f71700b': 'trip_distance_in_CH',
                                                            'f70700_01': 'trip_goal',
                                                            'f70801': 'main_transport_mode',
                                                            'RZ_LND': 'destination_country'})
    return df_overnight_trips


def get_zp_renamed():
    selected_columns = ['HHNR', 'WP', 'dmod', 'f70100', 'alter']
    df_zp = get_zp(year=2015, selected_columns=selected_columns)
    # Rename variables
    df_zp = df_zp.rename(columns={'dmod': 'module_attributed_to_the_respondent',
                                  'f70100': 'nb_trips_with_overnights',
                                  'alter': 'age'})
    return df_zp


if __name__ == '__main__':
    run()
