#! /usr/bin/env python3


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from matplotlib.widgets import SpanSelector
from matplotlib.widgets import Button
import math
from scipy import signal

search_range = 3
'''19'''
rolling_order = 7
'''3 for -35 kPa, 13 for -25 kPa ?'''
smoothing_order = 3

def open_csv(filepath, separateur = '\t'):
    original_file_dataframe = pd.read_csv(filepath, sep= separateur)                # Ouverture du fichier csv avec le séparteur "tabulation"
    clean_data_dataframe = original_file_dataframe[0:].astype(float)         # Convertion de toute les données de la datafram en float
    return clean_data_dataframe


def plot_df(df):
## Affichage du graph de pression de tous les devices
    column_name_list = []                       # Liste contenant le nom de chaque device
    for column_name in df.columns :    # Remplissage de la liste
        column_name_list.append(column_name)
    fig_list = []
    for column_name in column_name_list[1:]:                                 # Plot de toutes les courbes de pression clear dans une figure différente
        fig = plt.figure(column_name)
        ax0 = fig.add_subplot(111)
        ax0.plot(df[column_name_list[0]],df[column_name])
        fig_list.append(fig)
    plt.show()
    return 0

def plot_2_df(df1,df2):   # Permet de comparer 2 Dataframe

    column_name_list1 = []                       # Liste contenant le nom de chaque device
    column_name_list2 = []

    for column_name in df1.columns :    # Remplissage de la liste
        column_name_list1.append(column_name)

    for column_name in df2.columns :    # Remplissage de la liste
        column_name_list2.append(column_name)
    fig_list = []
    for  i in range(len(column_name_list1)):                                 # Plot de toutes les courbes fr df1 dans une figure différente
        if i == 0:
            pass
        else:
            fig = plt.figure(column_name_list1[i]+", "+column_name_list2[i])
            ax0 = fig.add_subplot(111)
            ax0.plot(df1[column_name_list1[0]],df1[column_name_list1[i]],color = 'blue', label = column_name_list1[i])
            ax0.plot(df2[column_name_list2[0]],df2[column_name_list2[i]],color = 'red', label = column_name_list2[i])
            plt.legend()
            fig_list.append(fig)

    plt.show()

    return 0

def df_average_filter(df,smoothing_order):
    column_name_list = []                       # Liste contenant le nom de chaque device
    for column_name in df.columns :    # Remplissage de la liste
        column_name_list.append(column_name)
    filtered_df = pd.DataFrame()
    filtered_df[column_name_list[0]] = df[column_name_list[0]]

    for column_name in column_name_list[1:]:
        filtered_df[column_name+"_average_filter_"+str(smoothing_order)] = df[column_name].rolling(smoothing_order, center=True).sum() / smoothing_order

    return filtered_df


def plot_min_df(original_df, min_df):

    column_name_list_original = []                       # Liste contenant le nom de chaque device
    column_name_list_min = []

    for column_name in original_df.columns :    # Remplissage de la liste
        column_name_list_original.append(column_name)

    for column_name in min_df.columns :    # Remplissage de la liste
        column_name_list_min.append(column_name)


    for  i in range(len(column_name_list_original)):
        if i == 0:
            pass
        else:
            fig = plt.figure(column_name_list_original[i]+", "+column_name_list_min[i])
            ax0 = fig.add_subplot(111)
            ax0.plot(original_df[column_name_list_original[0]],original_df[column_name_list_original[i]],color = 'blue', label = column_name_list_original[i])
            ax0.scatter(min_df[column_name_list_min[0]],min_df[column_name_list_min[i]],color = 'red', label = column_name_list_min[i])
            plt.legend()
            #fig_list.append(fig)


    plt.show()


    return 0

def plot_max_df(original_df, max_df):

    column_name_list_original = []                       # Liste contenant le nom de chaque device
    column_name_list_max = []

    for column_name in original_df.columns :    # Remplissage de la liste
        column_name_list_original.append(column_name)

    for column_name in max_df.columns :    # Remplissage de la liste
        column_name_list_max.append(column_name)


    for  i in range(len(column_name_list_original)):
        if i == 0:
            pass
        else:
            fig = plt.figure(column_name_list_original[i]+", "+column_name_list_max[i])
            ax0 = fig.add_subplot(111)
            ax0.plot(original_df[column_name_list_original[0]],original_df[column_name_list_original[i]],color = 'blue', label = column_name_list_original[i])
            ax0.scatter(max_df[column_name_list_max[0]],max_df[column_name_list_max[i]],color = 'red', label = column_name_list_max[i])
            plt.legend()
    plt.show()


    return 0

def plot_extremum_df(original_df, min_df, max_df):

    column_name_list_original = []                       # Liste contenant le nom de chaque device
    column_name_list_max = []
    column_name_list_min = []

    for column_name in original_df.columns :    # Remplissage de la liste
        column_name_list_original.append(column_name)

    for column_name in max_df.columns :    # Remplissage de la liste
        column_name_list_max.append(column_name)

    for column_name in min_df.columns :    # Remplissage de la liste
        column_name_list_min.append(column_name)


    for  i in range(len(column_name_list_original)):
        if i == 0:
            pass
        else:
            fig = plt.figure(column_name_list_original[i]+", "+column_name_list_max[i])
            ax0 = fig.add_subplot(111)
            ax0.plot(original_df[column_name_list_original[0]],original_df[column_name_list_original[i]],color = 'blue', label = column_name_list_original[i])
            ax0.scatter(min_df[column_name_list_min[0]],min_df[column_name_list_min[i]],color = 'm', label = column_name_list_min[i])
            ax0.scatter(max_df[column_name_list_max[0]],max_df[column_name_list_max[i]],color = 'red', label = column_name_list_max[i])
            plt.legend()
    plt.show()


    return 0

def compute_extremums(df, r_order,min_condition = -15, max_condition = -5):
       # Pour etre un min la pression doit etre en dessous de min_conditon
      # Pour etre un max la pression doit etre au dessus de max_condition
    column_name_list = []                       # Liste contenant le nom de chaque device
    for column_name in df.columns :    # Remplissage de la liste
        column_name_list.append(column_name)

    df_min = pd.DataFrame()
    df_min[column_name_list[0]] = df[column_name_list[0]]
    df_max = pd.DataFrame()
    df_max[column_name_list[0]] = df[column_name_list[0]]

    for column_name in column_name_list[1:]:
        df_min[column_name+'_min(kPa)'] = df.iloc[argrelextrema(df[column_name].values, np.less_equal, order=r_order)[0]][column_name]
        df_max[column_name+'_max(kPa)'] = df.iloc[argrelextrema(df[column_name].values, np.greater_equal, order=r_order)[0]][column_name]

    column_name_list_min = []                       # Liste contenant le nom de chaque device
    for column_name in df_min.columns :    # Remplissage de la liste
        column_name_list_min.append(column_name)
    column_name_list_max = []                       # Liste contenant le nom de chaque device
    for column_name in df_max.columns :    # Remplissage de la liste
        column_name_list_max.append(column_name)


    for column_number in range(len(column_name_list_min[1:])):
        for index in range(df_min.shape[0]):
            if df_min[column_name_list_min[column_number+1]][index] > min_condition:
                df_min[column_name_list_min[column_number+1]][index] = math.nan

        for index in range(df_max.shape[0]):
            if df_max[column_name_list_max[column_number+1]][index] < max_condition:
                df_max[column_name_list_max[column_number+1]][index] = math.nan


    return df_min, df_max


"""
    column_name_list_max = []                       # Liste contenant le nom de chaque device
    for column_name in df_max.columns :    # Remplissage de la liste
        column_name_list_max.append(column_name)

    search_range = 10

    for i in range(len(column_name_list_max[1:])):
        for index in df_max.iterrows():
            if (index[0] > search_range) and index[0] < (df.shape[0] - search_range + 1):
                if not pd.isna(df_max[column_name_list_max[i+1]][index[0]]):
                    for x in range(index[0] - search_range, index[0] + search_range):
                        if df_max[column_name_list_max[i+1]][index[0]] < df[column_name_list[i+1]][x]:
                            df_max[column_name_list_max[i+1]][index[0]] = df[column_name_list[i+1]][x]
                            df_max[column_name_list_max[i+1]][index[0]] = df[column_name_list[i+1]][x]
"""


def butterwoth_filter(df, fc = 1500, fe = 200):
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    f_niquist = fc/fe
    b, a = signal.butter(2, f_niquist,  btype='lowpass',analog=False,fs=fe)

    column_name_list = []                       # Liste contenant le nom de chaque device
    for column_name in df.columns :             # Remplissage de la liste
        column_name_list.append(column_name)


    df_filtered = pd.DataFrame()
    df_filtered[column_name_list[0]] = df[column_name_list[0]]
    for columnn_name in column_name_list[1:]:
        df_filtered[columnn_name+"_BW_filter"] = signal.filtfilt(b, a, df[columnn_name])

    print(df_filtered)
    return df_filtered


def cycleAnalysis(df):

#  f_mean, p_min, average_p_min, variance_p_min, cycle duration 

    return




"""



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


