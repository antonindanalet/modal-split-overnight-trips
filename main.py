# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geopandas
import unicodedata
import re
from utils_mtmc.get_mtmc_files import *
from mtmc2015.utils2015.compute_confidence_interval import get_weighted_avg_and_std
from mtmc2015.utils2015.codes import code2country


def run():
    df_zp = get_zp_renamed()
    # select only people who were asked questions about trips with overnights (module 1b, encoded 2)
    df_zp = df_zp[df_zp['module_attributed_to_the_respondent'] == 2]
    df_zp.drop(columns=['module_attributed_to_the_respondent'], inplace=True)
    # select only people who said the number of trips with overnights they made
    df_zp['with_trips'] = df_zp['nb_trips_with_overnights'] > 0
    df_zp = df_zp[~df_zp['nb_trips_with_overnights'] < 0]
    # sum of weights of the declared trips, including those without details, in particular distance
    weight_declared_trips = df_zp[df_zp['with_trips']]['WP'].sum()
    # select overnight trips whose distance is known
    df_overnight_trips = get_overnight_trips_in_2015_renamed()  # contains all trips with overnights
    # Select only trips that have been selected on the phone interview and that contain details
    df_overnight_trips = df_overnight_trips[df_overnight_trips['number_of_selected_trip'] > 0]
    # only keep trips going abroad from Switzerland or starting abroad going to Switzerland
    df_overnight_trips = df_overnight_trips[(df_overnight_trips['destination_country'] != 8100) |
                                            (df_overnight_trips['origin_country'] != 8100) |
                                            ~(df_overnight_trips['station1_country'].isin([-99, 8100])) |
                                            ~(df_overnight_trips['station2_country'].isin([-99, 8100])) |
                                            ~(df_overnight_trips['station3_country'].isin([-99, 8100]))]
    # only keep trips with known transport mode
    df_overnight_trips = df_overnight_trips[(df_overnight_trips['main_transport_mode'] != -97) &
                                            (df_overnight_trips['main_transport_mode'] != -98)]
    # Keep only trips with a distance
    df_overnight_trips.loc[df_overnight_trips['trip_distance_in_CH'] == -99,
                           'trip_distance_in_CH'] = df_overnight_trips['trip_distance']
    df_overnight_trips = df_overnight_trips[df_overnight_trips['trip_distance_in_CH'] > 0]
    df_overnight_trips = df_overnight_trips[df_overnight_trips['trip_distance'] > 0]
    # get number of detailed trips (with distance) by person
    df_overnight_trips_count_nb = df_overnight_trips[['HHNR', 'trip_distance']].groupby('HHNR').count()
    df_overnight_trips_count_nb = df_overnight_trips_count_nb.rename(columns={'trip_distance': 'nb_detailed_trips'})
    df_zp = pd.merge(df_zp, df_overnight_trips_count_nb, left_on='HHNR', right_index=True, how='left')
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
                           95: 'Autre'}
    df_overnight_trips['main_transport_mode_agg'] = df_overnight_trips['main_transport_mode'].map(transport_mean2mode)
    # Remove people who said they did a trip, but whose distance are not valid
    # For HHNR=333950, the distance is known, but not the country of destination.
    # For HHNR=488907, the destination is known but is the same as home.
    df_overnight_trips = df_overnight_trips[~df_overnight_trips['HHNR'].isin([333950, 488907])]  # Is this correct ??
    print('Basis:', len(df_overnight_trips),  #
          'overnight trips collected with details, in particular with a valid information about the distance, and '
          'whose destination is abroad.')
    """ --- """
    # get total distance per person by transport mode
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
    df_overnight_trips_count_sum = df_overnight_trips[['HHNR', 'trip_distance',
                                                       'trip_distance_by_plane',
                                                       'trip_distance_by_pt',
                                                       'trip_distance_by_car',
                                                       'trip_distance_by_autocar',
                                                       'trip_distance_by_other']].groupby('HHNR').sum()
    df_overnight_trips_count_sum = \
        df_overnight_trips_count_sum.rename(columns={'trip_distance': 'total_distance',
                                                     'trip_distance_by_plane': 'total_distance_by_plane',
                                                     'trip_distance_by_pt': 'total_distance_by_pt',
                                                     'trip_distance_by_car': 'total_distance_by_car',
                                                     'trip_distance_by_autocar': 'total_distance_by_autocar',
                                                     'trip_distance_by_other': 'total_distance_by_other'})
    df_zp_with_trips_temp = pd.merge(df_zp[df_zp['with_trips']], df_overnight_trips_count_sum,
                                     left_on='HHNR', right_index=True, how='inner')
    df_zp = pd.concat([df_zp[~df_zp['with_trips']], df_zp_with_trips_temp])
    # extrapolation for all trips based on declared trips
    df_zp['total_distance_extrapolated'] = \
        df_zp['total_distance'] * df_zp['nb_trips_with_overnights'] / \
        df_zp['nb_detailed_trips']
    df_zp['total_distance_by_plane_extrapolated'] = \
        df_zp['total_distance_by_plane'] * df_zp['nb_trips_with_overnights'] / \
        df_zp['nb_detailed_trips']
    df_zp['total_distance_by_pt_extrapolated'] = \
        df_zp['total_distance_by_pt'] * df_zp['nb_trips_with_overnights'] / \
        df_zp['nb_detailed_trips']
    df_zp['total_distance_by_car_extrapolated'] = \
        df_zp['total_distance_by_car'] * df_zp['nb_trips_with_overnights'] / \
        df_zp['nb_detailed_trips']
    df_zp['total_distance_by_autocar_extrapolated'] = \
        df_zp['total_distance_by_autocar'] * df_zp['nb_trips_with_overnights'] / \
        df_zp['nb_detailed_trips']
    df_zp['total_distance_by_other_extrapolated'] = \
        df_zp['total_distance_by_other'] * df_zp['nb_trips_with_overnights'] / \
        df_zp['nb_detailed_trips']
    # Sum of weights of the detailed trips, without those missing details, in particular distance
    weight_detailed_trips = df_zp[df_zp['with_trips']]['WP'].sum()
    # Correction factor for people declaring they did trips, but without detailing them
    correction_factor_declared_detailed_trips = weight_declared_trips / weight_detailed_trips
    df_zp['WP_corrected'] = np.where(df_zp['with_trips'],
                                     df_zp['WP'] * correction_factor_declared_detailed_trips,
                                     df_zp['WP'])
    df_overnight_trips = pd.merge(df_overnight_trips, df_zp[['HHNR', 'WP_corrected',
                                                             'total_distance_by_plane_extrapolated',
                                                             'total_distance_by_pt_extrapolated',
                                                             'total_distance_by_car_extrapolated',
                                                             'total_distance_by_autocar_extrapolated',
                                                             'total_distance_by_other_extrapolated']],
                                  on='HHNR', how='left')
    dict_column_weighted_avg_and_std, \
        sample = get_weighted_avg_and_std(df_overnight_trips,
                                          weights='WP_corrected', percentage=True,
                                          list_of_columns=['total_distance_by_plane_extrapolated',
                                                           'total_distance_by_pt_extrapolated',
                                                           'total_distance_by_car_extrapolated',
                                                           'total_distance_by_autocar_extrapolated',
                                                           'total_distance_by_other_extrapolated'])
    print(dict_column_weighted_avg_and_std)
    # {'total_distance_by_plane_extrapolated': [6392.478257075662, 287.2957478297803],     5924.3606645
    #  'total_distance_by_pt_extrapolated': [328.1650071027181, 36.031431845065406],       260.04529228
    #  'total_distance_by_car_extrapolated': [1091.1292114849887, 41.469835378121275],      1017.5690028
    #  'total_distance_by_autocar_extrapolated': [89.17073372467753, 12.631761681253586],    90.013509965
    #  'total_distance_by_other_extrapolated': [54.36982363494436, 14.084073815662288]}
    # {'total_distance_by_plane_extrapolated': [0.8035482991731556, 0.008918194339649594],   81 NOT OK
    #  'total_distance_by_pt_extrapolated': [0.04125104892044915, 0.0044638849347312624],    4 OK
    #  'total_distance_by_car_extrapolated': [0.13715729437115104, 0.007721795805898882],    14 OK
    #  'total_distance_by_autocar_extrapolated': [0.01120895348234876, 0.002363076810386391],    1 OK
    #  'total_distance_by_other_extrapolated': [0.006834404052895455, 0.0018492865080878213]}    1 NOT OK

    get_modalsplit_by_country(df_overnight_trips)
    get_modalsplit_by_nuts(df_overnight_trips)


def get_modalsplit_by_nuts(df_overnight_trips):
    # Removing places where NUTS are not defined or not relevant
    df_overnight_trips = df_overnight_trips[(df_overnight_trips['destination_country'] < 8300) &  # Keep only European
                                            (df_overnight_trips['destination_country'] != 8264) &  # Not  Russia
                                            (df_overnight_trips['destination_country'] != 8265) &  # Not Ukraine
                                            (df_overnight_trips['destination_country'] != 8252) &  # Not Bosnia
                                            (df_overnight_trips['destination_country'] != 8256) &  # Not Kosovo
                                            (df_overnight_trips['destination_country'] != 8100)]
    # Removing places where the quality of the coordinates of the destination are bad (i.e., country level only)
    df_overnight_trips = df_overnight_trips[df_overnight_trips['quality_dest_coord'].isin([3, 4])]
    list_modes = ['motorisierter Individualverkehr', 'öffentlicher Verkehr', 'Reisecar', 'Flugzeug', 'übrige']
    list_columns = ['Land']
    for mode in list_modes:
        list_columns.extend([mode, mode + ' (+/-)'])
    list_columns.append('Basis')
    df_for_csv = pd.DataFrame(columns=list_columns)

    'Add NUTS data'
    gdf_overnight_trips = geopandas.GeoDataFrame(df_overnight_trips,
                                                 geometry=geopandas.points_from_xy(df_overnight_trips.dest_x_coord,
                                                                                   df_overnight_trips.dest_y_coord),
                                                 crs='epsg:4326')
    # Read the shape file containing the NUTS data
    NUTS_folder_path = Path('data/inputs/NUTS_RG_01M_2021_3035.shp/')
    df_NUTS = geopandas.read_file(NUTS_folder_path / 'NUTS_RG_01M_2021_3035.shp')
    df_NUTS = df_NUTS[df_NUTS['LEVL_CODE'] == 1]
    df_NUTS.to_crs(epsg=4326, inplace=True)
    gdf_overnight_trips = geopandas.sjoin(gdf_overnight_trips, df_NUTS[['NAME_LATN', 'geometry']],
                                          how='left', predicate='intersects')
    gdf_overnight_trips.drop(['index_right'], axis=1, inplace=True)
    # Manuel corrections
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 143631) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Este'
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 152791) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = "Provence-Alpes-Côte d’Azur"  # Monaco
    gdf_overnight_trips.loc[((gdf_overnight_trips['HHNR'] == 155056) | (gdf_overnight_trips['HHNR'] == 244225) |
                             (gdf_overnight_trips['HHNR'] == 270084) | (gdf_overnight_trips['HHNR'] == 288473) |
                             (gdf_overnight_trips['HHNR'] == 305761) | (gdf_overnight_trips['HHNR'] == 313206) |
                             (gdf_overnight_trips['HHNR'] == 319623) | (gdf_overnight_trips['HHNR'] == 333400) |
                             (gdf_overnight_trips['HHNR'] == 356805) | (gdf_overnight_trips['HHNR'] == 358702) |
                             (gdf_overnight_trips['HHNR'] == 393455) | (gdf_overnight_trips['HHNR'] == 395357) |
                             (gdf_overnight_trips['HHNR'] == 402588) | (gdf_overnight_trips['HHNR'] == 415120) |
                             (gdf_overnight_trips['HHNR'] == 415548) | (gdf_overnight_trips['HHNR'] == 451879) |
                             (gdf_overnight_trips['HHNR'] == 458488) | (gdf_overnight_trips['HHNR'] == 482734) |
                             (gdf_overnight_trips['HHNR'] == 483636) | (gdf_overnight_trips['HHNR'] == 484244) |
                             (gdf_overnight_trips['HHNR'] == 496601) | (gdf_overnight_trips['HHNR'] == 331263)) &
                            (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = "Akdeniz"  # Antalya, 31.38852 36.76672
    gdf_overnight_trips.loc[((gdf_overnight_trips['HHNR'] == 210400) | (gdf_overnight_trips['HHNR'] == 222129) |
                             (gdf_overnight_trips['HHNR'] == 239507) | (gdf_overnight_trips['HHNR'] == 266268) |
                             (gdf_overnight_trips['HHNR'] == 279298) | (gdf_overnight_trips['HHNR'] == 302999) |
                             (gdf_overnight_trips['HHNR'] == 362051) | (gdf_overnight_trips['HHNR'] == 343824) |
                             (gdf_overnight_trips['HHNR'] == 322389) | (gdf_overnight_trips['HHNR'] == 390773) |
                             (gdf_overnight_trips['HHNR'] == 392617) | (gdf_overnight_trips['HHNR'] == 413971) |
                             (gdf_overnight_trips['HHNR'] == 445200) | (gdf_overnight_trips['HHNR'] == 319731)) &
                            (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Kýpros'  # 34.00183 34.98213
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 217997) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Södra Sverige'
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 228851) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Região Autónoma dos Açores'
    gdf_overnight_trips.loc[((gdf_overnight_trips['HHNR'] == 246703) | (gdf_overnight_trips['HHNR'] == 258329) |
                             (gdf_overnight_trips['HHNR'] == 267142) | (gdf_overnight_trips['HHNR'] == 303078) |
                             (gdf_overnight_trips['HHNR'] == 339084) | (gdf_overnight_trips['HHNR'] == 380182) |
                             (gdf_overnight_trips['HHNR'] == 421540) | (gdf_overnight_trips['HHNR'] == 441755)) &
                            (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = "Provence-Alpes-Côte d’Azur"  # Monaco, 7.43701 43.74965
    gdf_overnight_trips.loc[((gdf_overnight_trips['HHNR'] == 283584) | (gdf_overnight_trips['HHNR'] == 324292) |
                             (gdf_overnight_trips['HHNR'] == 395375)) &
                            (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Norge'
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 293599) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Ireland'  # In Irish waters...
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 306501) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Bretagne'
    gdf_overnight_trips.loc[((gdf_overnight_trips['HHNR'] == 327375) | (gdf_overnight_trips['HHNR'] == 415131)) &
                            (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Hrvatska'  # 14.76083 44.75694
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 339290) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Nord - Est'
    gdf_overnight_trips.loc[((gdf_overnight_trips['HHNR'] == 356156) | (gdf_overnight_trips['HHNR'] == 443830)) &
                            (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Schleswig - Holstein'  # SPO
    gdf_overnight_trips.loc[(gdf_overnight_trips['HHNR'] == 388734) & (gdf_overnight_trips['NAME_LATN'].isna()),
                            'NAME_LATN'] = 'Manner - Suomi'

    gdf_overnight_trips.drop(gdf_overnight_trips[(gdf_overnight_trips['HHNR'] == 283064) &
                                                 (gdf_overnight_trips['NAME_LATN'].isna())].index,
                             inplace=True)  # South Atlantic ocean, middle of nowhere

    gdf_overnight_trips['destination_country_name'] = gdf_overnight_trips['destination_country'].map(code2country)
    for NUTS1 in gdf_overnight_trips.NAME_LATN.unique():
        print(NUTS1)
        df_NUTS1 = gdf_overnight_trips.loc[gdf_overnight_trips['NAME_LATN'] == NUTS1, :]
        basis = len(df_NUTS1)
        if basis >= 10:
            dict_column_weighted_avg_and_std, \
                sample = get_weighted_avg_and_std(df_NUTS1, weights='WP_corrected', percentage=True,
                                                  list_of_columns=['total_distance_by_plane_extrapolated',
                                                                   'total_distance_by_pt_extrapolated',
                                                                   'total_distance_by_car_extrapolated',
                                                                   'total_distance_by_autocar_extrapolated',
                                                                   'total_distance_by_other_extrapolated'])

            df_for_figure = pd.DataFrame([[dict_column_weighted_avg_and_std['total_distance_by_car_extrapolated'][0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_pt_extrapolated'][0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_autocar_extrapolated'][
                                               0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_plane_extrapolated'][
                                               0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_other_extrapolated'][
                                               0] * 100]],
                                         columns=[list_modes])
            fig, ax = plt.subplots(figsize=(10, 4))
            df_for_figure.plot(kind='barh', stacked=True, ax=ax, color=['#000000',  # by car
                                                                        '#E33B3B',  # by PT
                                                                        'g',  # Autocar
                                                                        'y',
                                                                        '0.8'])
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['motorisierter Individualverkehr',
                                               'öffentlicher Verkehr',
                                               'Reisecar',
                                               'Flugzeug',
                                               'übrige'])
            sns.move_legend(ax, bbox_to_anchor=(1.01, 1.02), loc='upper left')
            ax.set_yticks([])
            n = 0
            if len(ax.patches) == 5:
                for p in ax.patches:
                    h, w, x, y = p.get_height(), p.get_width(), p.get_x(), p.get_y()
                    text = f'{w:0.0f}%'
                    if n == 4:
                        text_color = 'black'
                    else:
                        text_color = 'white'
                    ax.annotate(text=text, xy=(x + w / 2, y + h / 2), ha='center', va='center', color=text_color, size=16)
                    n = n + 1
            plt.xlim([0, 100])
            plt.title('Verkehrsmittelwahl bei Reisen mit Übernachtungen: ' + NUTS1.split('/')[0] +
                      ' (' + df_NUTS1['destination_country_name'].iloc[0] + ')')
            plt.annotate(text='Basis: ' + str(basis), xy=(70, -0.37), ha='center', va='center',
                         color='black', size=12)
            plt.tight_layout()
            if basis < 50:
                name_NUTS1 = '(' + slugify(NUTS1) + ')'
            else:
                name_NUTS1 = slugify(NUTS1)
            plt.savefig(Path('data/outputs/figures/by_NUTS1/' + name_NUTS1 + '.png'))
            plt.close()

            list_row = [name_NUTS1,
                        dict_column_weighted_avg_and_std['total_distance_by_car_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_car_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_pt_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_pt_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_autocar_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_autocar_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_plane_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_plane_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_other_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_other_extrapolated'][1], basis]
            df_for_csv.loc[len(df_for_csv)] = list_row
    df_for_csv.to_csv(Path('data/outputs/tables/by_NUTS1.csv'), index=False, sep=',', encoding='iso-8859-1')


def get_modalsplit_by_country(df_overnight_trips):
    list_modes = ['motorisierter Individualverkehr', 'öffentlicher Verkehr', 'Reisecar', 'Flugzeug', 'übrige']
    list_columns = ['Land']
    for mode in list_modes:
        list_columns.extend([mode, mode + ' (+/-)'])
    list_columns.append('Basis')
    df_for_csv = pd.DataFrame(columns=list_columns)

    df_overnight_trips['destination_country_name'] = df_overnight_trips['destination_country'].map(code2country)
    for country in ['ALBANIEN', 'BELGIEN', 'BULGARIEN', 'DAENEMARK', 'DEUTSCHLAND', 'FINNLAND', 'FRANKREICH',
                    'GRIECHENLAND', 'GROSSBRITANIEN', 'IRLAND', 'ITALIEN',
                    'LUXEMBURG', 'MALTA', 'MONACO', 'NIEDERLANDE', 'NORWEGEN', 'OESTERREICH', 'POLEN', 'PORTUGAL',
                    'RUMAENIEN', 'SCHWEDEN', 'SPANIEN', 'TUERKEI', 'UNGARN', 'ZYPERN', 'SLOWAKEI',
                    'TSCHECHISCHE REPUBLIK', 'SERBIEN', 'KROATIEN', 'SLOWENIEN', 'BOSNIEN UND HERZEGOWINA',
                    'MONTENEGRO', 'MAZEDONIEN', 'KOSOVO', 'RUSSLAND']:
        df_country = df_overnight_trips.loc[df_overnight_trips['destination_country_name'] == country, :]
        basis = len(df_country)
        if basis >= 10:
            dict_column_weighted_avg_and_std, \
                sample = get_weighted_avg_and_std(df_country, weights='WP_corrected', percentage=True,
                                                  list_of_columns=['total_distance_by_plane_extrapolated',
                                                                   'total_distance_by_pt_extrapolated',
                                                                   'total_distance_by_car_extrapolated',
                                                                   'total_distance_by_autocar_extrapolated',
                                                                   'total_distance_by_other_extrapolated'])

            df_for_figure = pd.DataFrame([[dict_column_weighted_avg_and_std['total_distance_by_car_extrapolated'][0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_pt_extrapolated'][0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_autocar_extrapolated'][
                                               0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_plane_extrapolated'][
                                               0] * 100,
                                           dict_column_weighted_avg_and_std['total_distance_by_other_extrapolated'][
                                               0] * 100]],
                                         columns=[list_modes])
            fig, ax = plt.subplots(figsize=(10, 4))
            df_for_figure.plot(kind='barh', stacked=True, ax=ax, color=['#000000',  # by car
                                                                        '#E33B3B',  # by PT
                                                                        'g',  # Autocar
                                                                        'y',
                                                                        '0.8'])
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=['motorisierter Individualverkehr',
                                               'öffentlicher Verkehr',
                                               'Reisecar',
                                               'Flugzeug',
                                               'übrige'])
            sns.move_legend(ax, bbox_to_anchor=(1.01, 1.02), loc='upper left')
            ax.set_yticks([])
            n = 0
            if len(ax.patches) == 5:
                for p in ax.patches:
                    h, w, x, y = p.get_height(), p.get_width(), p.get_x(), p.get_y()
                    text = f'{w:0.0f}%'
                    if n == 4:
                        text_color = 'black'
                    else:
                        text_color = 'white'
                    ax.annotate(text=text, xy=(x + w / 2, y + h / 2), ha='center', va='center', color=text_color, size=16)
                    n = n + 1
            plt.xlim([0, 100])
            plt.title('Verkehrsmittelwahl bei Reisen mit Übernachtungen: ' + country)
            plt.annotate(text='Basis: ' + str(basis), xy=(70, -0.37), ha='center', va='center',
                         color='black', size=12)
            plt.tight_layout()
            if basis < 50:
                name_country = '(' + country + ')'
            else:
                name_country = country
            plt.savefig(Path('data/outputs/figures/by_country/' + name_country + '.png'))
            plt.close()

            list_row = [name_country,
                        dict_column_weighted_avg_and_std['total_distance_by_car_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_car_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_pt_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_pt_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_autocar_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_autocar_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_plane_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_plane_extrapolated'][1],
                        dict_column_weighted_avg_and_std['total_distance_by_other_extrapolated'][0],
                        dict_column_weighted_avg_and_std['total_distance_by_other_extrapolated'][1], basis]
            df_for_csv.loc[len(df_for_csv)] = list_row
    df_for_csv.to_csv(Path('data/outputs/tables/by_country.csv'), index=False, sep=',', encoding='iso-8859-1')


def get_overnight_trips_in_2015_renamed():
    selected_columns = ['HHNR', 'WP', 'RENR', 'reisenr', 'f70801', 'f71300', 'f71400_01', 'f71600b', 'RZ_QAL',
                        'f70700_01', 'f71700b', 'RZ_LND', 'RZ_X', 'RZ_Y', 'RS_LND', 'RS1_LND', 'RS2_LND', 'RS3_LND']
    df_overnight_trips = get_overnight_trips(year=2015, selected_columns=selected_columns)
    # Rename variables
    df_overnight_trips = df_overnight_trips.rename(columns={'f71600b': 'trip_distance',
                                                            'f71700b': 'trip_distance_in_CH',
                                                            'f70700_01': 'trip_goal',
                                                            'f70801': 'main_transport_mode',
                                                            'RZ_LND': 'destination_country',
                                                            'RS_LND': 'origin_country',
                                                            'RZ_X': 'dest_x_coord',
                                                            'RZ_Y': 'dest_y_coord',
                                                            'RS1_LND': 'station1_country',
                                                            'RS2_LND': 'station2_country',
                                                            'RS3_LND': 'station3_country',
                                                            'RENR': 'number_of_selected_trip',
                                                            'RZ_QAL': 'quality_dest_coord'})
    return df_overnight_trips


def get_zp_renamed():
    selected_columns = ['HHNR', 'WP', 'dmod', 'f70100', 'alter']
    df_zp = get_zp(year=2015, selected_columns=selected_columns)
    # Rename variables
    df_zp = df_zp.rename(columns={'dmod': 'module_attributed_to_the_respondent',
                                  'f70100': 'nb_trips_with_overnights',
                                  'alter': 'age'})
    return df_zp


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


if __name__ == '__main__':
    run()
