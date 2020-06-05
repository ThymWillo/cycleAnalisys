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

#plot histogramme extremum
#verifier filtre butterworth

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

def compute_extremums(df, r_order=50,min_condition = -15, max_condition = -5):
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
    return df_min, df_max





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

    return df_filtered


def cycleAnalysis(df):

#  f_mean, p_min, p_minminmin, average_p_min, variance_p_min, cycle_duration

    return

def cycle_duration(df):
    df_min, df_max = compute_extremums(df)
    duration_list = []

    column_name_list_min = []                       # Liste contenant le nom de chaque device
    for column_name in df_min.columns :             # Remplissage de la liste
        column_name_list_min.append(column_name)

    column_name_list_max = []                       # Liste contenant le nom de chaque device
    for column_name in df_max.columns :             # Remplissage de la liste
        column_name_list_max.append(column_name)
    for i in range(df_min.shape[1]):
        if i == 0:
            pass
        else :
            #find the first min
            first_min = math.nan
            index_first = 0

            while math.isnan(first_min):
                index_first = index_first +1
                first_min = df_min[column_name_list_min[i]][df.index[index_first]]
            #print(str(df_min[column_name_list_min[0]][df.index[index]])+"        "+str(df_min[column_name_list_min[1]][df.index[index]]))

            #find the first max
            first_max = math.nan
            while math.isnan(first_max):
                index_first = index_first - 1
                first_max = df_max[column_name_list_max[i]][df.index[index_first]]
            #print(str(df_max[column_name_list_max[0]][df.index[index]])+"        "+str(df_max[column_name_list_max[1]][df.index[index]]))


            #find the last min
            last_min = math.nan
            index_last = df_min.shape[0]

            while math.isnan(last_min):
                index_last = index_last -1
                last_min = df_min[column_name_list_min[i]][df.index[index_last]]
            #print(str(df_min[column_name_list_min[0]][df.index[index]])+"        "+str(df_min[column_name_list_min[1]][df.index[index]]))

            #find the last max
            last_max = math.nan
            while math.isnan(last_max):
                index_last = index_last + 1
                last_max = df_max[column_name_list_max[i]][df.index[index_last]]
            #print(str(df_max[column_name_list_max[0]][df.index[index]])+"        "+str(df_max[column_name_list_max[1]][df.index[index]]))
            duration_list.append(df_max[column_name_list_max[0]][df.index[index_last]]-df_max[column_name_list_max[0]][df.index[index_first]])
    return duration_list



