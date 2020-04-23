#! /usr/bin/env python3

from cycleAnalysis import *




def main():
    directory_path = ""
    #file_name = "simple"
    file_name = "brossage_127_28-3-2020_5:25"
    filepath = directory_path + file_name + ".csv"

    pressure_df = open_csv(filepath)

    #plot_df(pressure_df)



    rolling_order = 50
    smoothing_order = 10



    #pressure_df_filtered = df_average_filter(pressure_df,smoothing_order)
    pressure_df_filtered = butterwoth_filter(pressure_df)
    plot_2_df(pressure_df,pressure_df_filtered)
    #plot_df(pressure_df_filtered)
    #min_df,df_max = compute_extremums(pressure_df_filtered,rolling_order,-150,50)
    #plot_min_df(pressure_df_filtered,min_df)
    #plot_max_df(pressure_df_filtered,df_max)
    #plot_extremum_df(pressure_df_filtered,min_df,df_max)



try:
    main()
except KeyboardInterrupt: 
    print("     sKeyboard interrupt on ferme tout !")