#! /usr/bin/env python3

from cycleAnalysis import *
import argparse

parser = argparse.ArgumentParser(description='Analyse some pressure data from csv file')
parser.add_argument('csv_file', type=str, help='CSV file to analyse')
args = parser.parse_args()




def main(terminal_arg):
    directory_path = ""
    #file_name = "simple"
    file_name = terminal_arg
    filepath = directory_path + file_name

    pressure_df = open_csv(filepath,',')
    print(pressure_df)

    #plot_df(pressure_df)

    #pressure_df_filtered = df_average_filter(pressure_df,smoothing_order)
 
    pressure_df_filtered = butterwoth_filter(pressure_df,30,1000,traceBode = True)
    #plot_df(pressure_df_filtered)
    plot_2_df(pressure_df,pressure_df_filtered)


    #plot_df(pressure_df_filtered)
    #min_df,df_max = compute_extremums(pressure_df_filtered)
    #plot_min_df(pressure_df_filtered,min_df)
    #plot_max_df(pressure_df_filtered,df_max)
    #plot_extremum_df(pressure_df_filtered,min_df,df_max)
    #duration = cycle_duration(pressure_df_filtered)
    #print(duration)
    #plot_df(pressure_df_filtered)


try:
    main(args.csv_file)
except KeyboardInterrupt:
    print("     sKeyboard interrupt on ferme tout !")