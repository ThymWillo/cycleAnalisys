#! /usr/bin/env python3


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button
import math

search_range = 3
'''19'''
rolling_order = 7
'''3 for -35 kPa, 13 for -25 kPa ?'''
smoothing_order = 3

def open_csv(filepath):
    original_file_dataframe = pd.read_csv(filepath, sep='\t')                # Ouverture du fichier csv avec le séparteur "tabulation"
    clean_data_dataframe = original_file_dataframe[0:].astype(float)         # Convertion de toute les données de la datafram en float
    return clean_data_dataframe

directory_path = ""
file_name = "brossage_127_28-3-2020_5:25"
filepath = directory_path + file_name + ".csv"

pressure_df = open_csv(filepath)

## Affichage du graph de pression de tous les devices

column_name_list = []                       # Liste contenant le nom de chaque device
for column_name in pressure_df.columns :    # Remplissage de la liste
    column_name_list.append(column_name)

for column_name in column_name_list[1:]:                                 # Plot de toutes les courbes de pression
    plt.plot(pressure_df[column_name_list[0]],pressure_df[column_name])

plt.show()



"""
def compute_extremums(df, r_order):
    df['Pressure']   = df['Pressure'] * 50 - 150
    df['Averaged'] = df.Pressure.rolling(smoothing_order, center=True).sum() / smoothing_order

    df['min'] = df.iloc[argrelextrema(df.Averaged.values, np.less_equal, order=r_order)[0]]['Averaged']
    df['max'] = df.iloc[argrelextrema(df.Averaged.values, np.greater_equal, order=r_order)[0]]['Averaged']

    for index, row in df.iterrows():
        if index > (search_range + 28) and index < (df.shape[0] - search_range + 1):
            if not pd.isna(row.max):

                for x in range(index - search_range, index + search_range):
                    if row['max'] < df['Pressure'][x]:
                        df['max'][index] = df['Pressure'][x]
                        row['max'] = df['Pressure'][x]

            if not pd.isna(row.min):
                for x in range(index - search_range, index + search_range):
                    if row['min'] > df['Pressure'][x]:
                        df['min'][index] = df['Pressure'][x]
                        row['min'] = df['Pressure'][x]
    return df


pressure_df = compute_extremums(pressure_df, rolling_order)

fig = plt.figure()
ax0 = fig.add_subplot(311)
ax0.plot(pressure_df.Time, pressure_df.Pressure, c='b', label='Pressure')
ax0.plot(pressure_df.Time, pressure_df.Averaged, c='g', label='Averaged')
ax0.scatter(pressure_df.Time, pressure_df['min'], c='r')
ax0.scatter(pressure_df.Time, pressure_df['max'], c='g')
'''ax0.xlabel('Pressure (kPa)')
ax0.ylabel('Time (s)')'''

ax1 = fig.add_subplot(312)
ax2 = fig.add_subplot(313)


def compute_min_frequency(df):
    minimum_data_dataframe = pd.DataFrame(columns=['Time', 'Pressure', 'Min'])
    minimum_data_dataframe = df.filter(['Time', 'Pressure', 'min'], axis=1)
    minimum_data_dataframe.dropna(inplace=True)
    minimum_data_dataframe['Duration'] = minimum_data_dataframe['Time'].diff()
    minimum_data_dataframe['Frequency'] = 1.0 / minimum_data_dataframe['Duration']
    is_relevant = minimum_data_dataframe['Frequency'] < 10
    minimum_data_dataframe = minimum_data_dataframe[is_relevant]
    minimum_data_dataframe['MinFreq'] = minimum_data_dataframe['Frequency'].min()
    minimum_data_dataframe['MaxFreq'] = minimum_data_dataframe['Frequency'].max()
    minimum_data_dataframe['MeanFreq'] = minimum_data_dataframe['Frequency'].mean()
    return minimum_data_dataframe

def compute_max_frequency(df):
    maximum_data_dataframe = pd.DataFrame(columns=['Time', 'Pressure', 'Max'])
    maximum_data_dataframe = df.filter(['Time', 'Pressure', 'max'], axis=1)
    maximum_data_dataframe.dropna(inplace=True)
    maximum_data_dataframe['Duration'] = maximum_data_dataframe['Time'].diff()
    maximum_data_dataframe['Frequency'] = 1.0 / maximum_data_dataframe['Duration']
    is_relevant = maximum_data_dataframe['Frequency'] < 10
    maximum_data_dataframe = maximum_data_dataframe[is_relevant]
    maximum_data_dataframe['MinFreq'] = maximum_data_dataframe['Frequency'].min()
    maximum_data_dataframe['MaxFreq'] = maximum_data_dataframe['Frequency'].max()
    maximum_data_dataframe['MeanFreq'] = maximum_data_dataframe['Frequency'].mean()
    return maximum_data_dataframe

def analyze_data(min_df, max_df):
    # Dropping first and last point of min and max to prevent edge effect
    min_df.drop(min_df.index[1])
    min_df.drop(min_df.index[-1])
    max_df.drop(max_df.index[1])
    max_df.drop(max_df.index[-1])
    statistical_data_df = pd.DataFrame(index=['0'], columns=['nb_cycle', 'f_min', 'f_max', 'f_mean', 'f_variance', 'p_min', 'average_p_min', 'variance_p_min', 'p_max', 'average_p_max', 'variance_p_max'])
    statistical_data_df['nb_cycle'][0] = (min_df.shape[0] + max_df.shape[0])/2.0
    if min_df['Frequency'].min() < max_df['Frequency'].min():
        statistical_data_df['f_min'][0] = min_df['Frequency'].min()
    else:
        statistical_data_df['f_min'][0] = max_df['Frequency'].min()

    if max_df['Frequency'].max() > min_df['Frequency'].max():
        statistical_data_df['f_max'][0] = max_df['Frequency'].max()
    else:
        statistical_data_df['f_max'][0] = min_df['Frequency'].max()

    temp_df = max_df.append(min_df)
    statistical_data_df['f_mean'][0] = temp_df['Frequency'].mean()

    statistical_data_df['p_min'][0] = min_df['Pressure'].min()
    statistical_data_df['average_p_min'][0] = min_df['Pressure'].mean()
    statistical_data_df['p_max'][0] = max_df['Pressure'].max()
    statistical_data_df['average_p_max'][0] = max_df['Pressure'].mean()

    acc = 0
    for index, row in temp_df.iterrows():
        acc += math.pow((row.Frequency - statistical_data_df['f_mean'][0]), 2)

    statistical_data_df['f_variance'] = acc / temp_df.shape[0]

    acc = 0
    for index, row in min_df.iterrows():
        acc += math.pow((row.Pressure - statistical_data_df['average_p_min'][0]), 2)

    statistical_data_df['variance_p_min'] = acc / temp_df.shape[0]

    acc = 0
    for index, row in max_df.iterrows():
        acc += math.pow((row.Pressure - statistical_data_df['average_p_max'][0]), 2)

    statistical_data_df['variance_p_max'] = acc / temp_df.shape[0]

    return statistical_data_df, min_df, max_df

def onselect(xmin, xmax):
    indmin, indmax = np.searchsorted(pressure_df['Time'], (xmin, xmax))
    indmax = min(len(pressure_df['Time']) - 1, indmax)
    global subdata_df
    subdata_df = pd.DataFrame(columns=['Time', 'Pressure', 'Averaged', 'min', 'max'])
    subdata_df['Time'] = pressure_df.Time[indmin:indmax]
    subdata_df['Pressure'] = pressure_df.Pressure[indmin:indmax]
    subdata_df['Averaged'] = pressure_df.Averaged[indmin:indmax]
    subdata_df['min'] = pressure_df['min'][indmin:indmax]
    subdata_df['max'] = pressure_df['max'][indmin:indmax]
    '''subdata_df['min'] = subdata_df.iloc[argrelextrema(subdata_df.Averaged.values, np.less_equal, order=rolling_order)[0]]['Averaged']
    subdata_df['max'] = subdata_df.iloc[argrelextrema(subdata_df.Averaged.values, np.greater_equal, order=rolling_order)[0]]['Averaged']'''

    minimum_data_df = compute_min_frequency(subdata_df)
    maximum_data_df = compute_max_frequency(subdata_df)

    extremum_data_df = minimum_data_df
    extremum_data_df.append(maximum_data_df)

    global output_data_df
    output_data_df, minimum_data_df, maximum_data_df = analyze_data(minimum_data_df, maximum_data_df)

    ax1.clear()
    ax1.plot(subdata_df.Time, subdata_df.Pressure, c='b', label='Pressure')
    ax1.plot(subdata_df.Time, subdata_df.Averaged, c='g', label='Averaged')
    #ax1.scatter(subdata_df.Time, subdata_df.min, c='r')
    #ax1.scatter(subdata_df.Time, subdata_df.max, c='g')

    '{:06.2f}'.format(3.141592653589793)
    output_text = 'cycle number : ' + '{:2.0f}'.format(output_data_df['nb_cycle'][0]) + '\n'
    output_text += 'min frequency : ' + '{:2.3f}'.format(output_data_df['f_min'][0]) + '\n'
    output_text += 'max frequency : ' + '{:2.3f}'.format(output_data_df['f_max'][0]) + '\n'
    output_text += 'mean frequency : ' + '{:2.3f}'.format(output_data_df['f_mean'][0]) + '\n'
    output_text += 'min pressure : ' + '{:2.3f}'.format(output_data_df['p_min'][0]) + '\n'
    output_text += 'average min pressure : ' + '{:2.3f}'.format(output_data_df['average_p_min'][0]) + '\n'
    output_text += 'min pressure variance : ' + '{:2.3f}'.format(output_data_df['variance_p_min'][0]) + '\n'
    output_text += 'max pressure : ' + '{:2.3f}'.format(output_data_df['p_max'][0]) + '\n'
    output_text += 'average max pressure : ' + '{:2.3f}'.format(output_data_df['average_p_max'][0]) + '\n'
    output_text += 'max pressure variance : ' + '{:2.3f}'.format(output_data_df['variance_p_max'][0]) + '\n'
    ax2.clear()
    ax2.text(0.01, 0.01, output_text, verticalalignment='bottom', horizontalalignment='left', transform=ax2.transAxes, color='black', fontsize=8)

span = SpanSelector(ax0, onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'))

def save_to_csv(self):
    output_file_name = directory_path + file_name + "_Ouptut.csv"
    output_data_df.to_csv(output_file_name, header=True, mode='a')
    subdata_df.to_csv(output_file_name, header=True, mode='a')
    '''Nothing to do, yet'''
    print('We are saving !')


axsave = plt.axes([0.90, 0.01, 0.05, 0.05])
bsave = Button(axsave, 'Save')
bsave.on_clicked(save_to_csv)

plt.show()
"""