#!/usr/bin/env python3

import networkx as nx
import blocksci
import zarr
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict
import sys, os, os.path, socket
import logging
import matplotlib.pyplot as plt
import matplotlib.dates
from util import SYMBOLS, DIR_PARSED, SimpleChrono
import random
import math
import json
import graph_tool.all as gt
import operator
import concurrent.futures
from scipy.stats import entropy
# from NEMtropy import DirectedGraph
# from NEMtropy import matrix_generator as mg
# from NEMtropy.network_functions import build_adjacency_from_edgelist
#TODO: rewrite so it rabdomizes 10 times ....

def parse_command_line():
    import sys, optparse

    parser = optparse.OptionParser()

    parser.add_option("--curr", action='store', dest="currency", type='str',
                                              default=None, help="name of the currency")
    parser.add_option("--heur", action='store', dest="heuristic", type='str',
                                                  default=None, help="heuristics to apply")
    #parser.add_option("--overwrite", action='store_true', dest = "overwrite" )
#    parser.add_option("--period",  action='store', dest="period",
#                       default = None , help = "minimum block number to process" )
    parser.add_option("--start", action="store", dest="start_date",
                       default = None, help= "starting date for network creation in YYYY-MM-DD format")
    parser.add_option("--end", action="store", dest="end_date",
                       default = None, help = "ending date for network creation in YYYY-MM-DD format")
    parser.add_option("--freq", action="store", dest="frequency",
                       default = "day", help = "time aggregation of networks - choose between day, week, 2weeks, 4weeks")

    options, args = parser.parse_args()

    options.currency = SYMBOLS[options.currency]

#    options.period = [0,-1] if options.period == None else list( map( int, options.period.split(",")))
#    assert len(options.period) == 2

    switcher = {"day":1, "week":7, "2weeks":14, "4weeks":28}


    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}"
    options.blocks_folder = f"{DIR_PARSED}/{options.currency}/heur_all_data"
    options.networks_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_networks_{options.frequency}"
    options.frequency = switcher[options.frequency]
    
    # if not os.path.exists(options.networks_folder):
    #     os.mkdir(options.networks_folder)


    return options, args             
                
def daterange(date1, date2, by=1):
    return [  date1 + timedelta(n) for n in range(0, int((date2 - date1).days)+1, by) ]         
        
def calculate_and_return_metrics(data, label):
    data_arr = np.array(list(data.values()), dtype=float)
    
    # Calculate and print Gini Coefficient
    gini_coefficient = gini(data_arr)
    # print(f'Gini Coefficient for {label}: {gini_coefficient}')

    # Calculate and print Entropy
    data_arr /= data_arr.sum()
    e = entropy(data_arr)
    # print(f'Entropy for {label}: {e}')
    
    # Create a dictionary for percentiles
    data_arr = np.array(list(data.values()), dtype=float)
    percentiles = [0, 25, 50, 75, 100]
    values = np.percentile(data_arr, percentiles)
    percentile_dict = {p: v for p, v in zip(percentiles, values)}
    # for p, v in percentile_dict.items():
    #     print(f'{p}th percentile for {label}: {v}')
    
    return gini_coefficient, e, percentile_dict

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = np.sort(array) # values must be sorted
    index = np.arange(1, array.shape[0]+1) # index per array element
    n = array.shape[0] # number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))  #Gini coefficient

def community_inequality_analysis(date):

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}
    
    logging.info(f'Analyzing Inequality for communities for the week of:{date} has started')

    start_time = datetime.now()
    
    chrono.add_tic("net")

    networks_path = f"/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/heur_{options.heuristic}_networks_full_shifted_community/{switcherback[options.frequency]}"
    unit_graph_file = f"{networks_path}/{date.strftime('%Y-%m-%d')}.graphml.bz2"

    if not os.path.exists(unit_graph_file):
        logging.info(f'community building the date:{date} is unsuccesful since original network does not exist')
        return 0.0,0.0,0.0,0.0,0.0,[0.0,0.0]
    
    # Load Graph
    try:
        g = gt.load_graph(unit_graph_file)      
    except OSError:
        logging.info(f'community building of the date:{date} is unsuccesful because of OSError')
        return 
    
    # Initialize an empty dictionary to store the dark_ratio, dark_assets, and current_assets values for each block
    block_dark_assets = {}
    block_darkness_ratios = {}
    block_current_assets = {}

    # Initialize an empty list to store the dark_ratio, dark_assets, and current_assets values for each vertex
    vertex_dark_assets = {}
    vertex_darkness_ratios = {}
    vertex_current_assets = {}

    # Traverse the nodes in the graph
    for v in g.vertices():
        # Extract the block, dark_ratio, dark_assets and current_assets properties
        block = g.vp.block[v]
        dark_ratio = g.vp.dark_ratio[v]
        dark_assets = g.vp.dark_assets[v]
        current_assets = g.vp.current_assets[v]

        # Append the respective properties to the appropriate list in the dictionary
        if block in block_dark_assets:
            block_dark_assets[block].append(dark_assets)
            block_darkness_ratios[block].append(dark_ratio)
            block_current_assets[block].append(current_assets)
        else:
            block_dark_assets[block] = [dark_assets]
            block_darkness_ratios[block] = [dark_ratio]
            block_current_assets[block] = [current_assets]
        
        # Append the respective properties to the appropriate list for vertex level analysis
        # vertex_dark_assets[block] = [dark_assets]
        # vertex_darkness_ratios[block] = [dark_ratio]
        # vertex_current_assets[block] = [current_assets]
        if v not in vertex_dark_assets:
            vertex_dark_assets[v] = [dark_assets]
            vertex_darkness_ratios[v] = [dark_ratio]
            vertex_current_assets[v] = [current_assets]
        else:
            vertex_dark_assets[v].append(dark_assets)
            vertex_darkness_ratios[v].append(dark_ratio)
            vertex_current_assets[v].append(current_assets)


    # Compute the total dark assets, average darkness ratio, and total current assets for each block
    block_total_dark_assets = {block: np.sum(assets) for block, assets in block_dark_assets.items()}
    block_avg_darkness_ratios = {block: np.mean(ratios) for block, ratios in block_darkness_ratios.items()}
    block_total_current_assets = {block: np.sum(assets) for block, assets in block_current_assets.items()}

    # Compute the total dark assets, average darkness ratio, and total current assets for each block
    vertex_total_dark_assets = {v: np.sum(assets) for v, assets in vertex_dark_assets.items()}
    vertex_avg_darkness_ratios = {v: np.mean(ratios) for v, ratios in vertex_darkness_ratios.items()}
    vertex_total_current_assets = {v: np.sum(assets) for v, assets in vertex_current_assets.items()}


    # Calculate and return metrics for block level
    block_dark_assets_metrics = calculate_and_return_metrics(block_total_dark_assets, "Block Level Total Dark Assets")
    block_darkness_ratios_metrics = calculate_and_return_metrics(block_avg_darkness_ratios, "Block Level Average Darkness Ratios")
    block_current_assets_metrics = calculate_and_return_metrics(block_total_current_assets, "Block Level Total Current Assets")

    # Calculate and return metrics for vertex level
    vertex_dark_assets_metrics = calculate_and_return_metrics(vertex_total_dark_assets, "Vertex Level Dark Assets")
    vertex_darkness_ratios_metrics = calculate_and_return_metrics(vertex_avg_darkness_ratios, "Vertex Level Darkness Ratios")
    vertex_current_assets_metrics = calculate_and_return_metrics(vertex_total_current_assets, "Vertex Level Current Assets")

    logging.info(f'Building for the date:{date} has finished with t={datetime.now() - start_time} finished:')
    logging.info(f"     Original graph: {g}")


    tqdm_bar.set_description(f"{switcherback[options.frequency]} of '{date.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)

    return {
    "block": {
        "dark_assets": block_dark_assets_metrics,
        "darkness_ratios": block_darkness_ratios_metrics,
        "current_assets": block_current_assets_metrics
    },
    "vertex": {
        "dark_assets": vertex_dark_assets_metrics,
        "darkness_ratios": vertex_darkness_ratios_metrics,
        "current_assets": vertex_current_assets_metrics
    }
}

def process_timeunit(timeunit):
    # Run randomizer + assortativity builder and store result
    metrics_dict = community_inequality_analysis(timeunit)

    # Add values to data dictionaries
    date = timeunit.strftime('%Y-%m-%d')
    x_values.append(date)
    x_values.sort()
    
    # for level, level_data in metrics_dict.items():  # Level can be 'block' or 'vertex'
    #     for metric, metric_data in level_data.items():  # Metric can be 'dark_assets', 'darkness_ratios', 'current_assets'
    #         gini_coefficient, entropy, percentile_dict = metric_data

    #         # Construct the key for data_dicts
    #         key = f"{level}_{metric}_gini"
    #         if key not in data_dicts:
    #             data_dicts[key] = {}
    #         data_dicts[key][date] = gini_coefficient

    #         key = f"{level}_{metric}_entropy"
    #         if key not in data_dicts:
    #             data_dicts[key] = {}
    #         data_dicts[key][date] = entropy

    #         for percentile, value in percentile_dict.items():
    #             key = f"{level}_{metric}_{percentile}th_percentile"
    #             if key not in data_dicts:
    #                 data_dicts[key] = {}
    #             data_dicts[key][date] = value
    
    for level, level_data in metrics_dict.items():  # Level can be 'block' or 'vertex'
        for metric, metric_data in level_data.items():  # Metric can be 'dark_assets', 'darkness_ratios', 'current_assets'
            gini_coefficient, entropy, percentile_dict = metric_data

            # Construct the key for data_dicts
            key_prefix = level if level == 'block' else 'vertex'  # Append 'vertex' to vertex level keys
            key = f"{key_prefix}_{metric}_gini"
            if key not in data_dicts:
                data_dicts[key] = {}
            data_dicts[key][date] = gini_coefficient

            key = f"{key_prefix}_{metric}_entropy"
            if key not in data_dicts:
                data_dicts[key] = {}
            data_dicts[key][date] = entropy

            for percentile, value in percentile_dict.items():
                key = f"{key_prefix}_{metric}_{percentile}th_percentile"
                if key not in data_dicts:
                    data_dicts[key] = {}
                data_dicts[key][date] = value

    # Save data dictionaries after each process
    for key, data_dict in data_dicts.items():
        sorted_data = dict(sorted(data_dict.items(), key=operator.itemgetter(0)))
        file_path = os.path.join(f'jsonResults_v3/h{options.heuristic}/community_inequality', f'{key}_2009-01-03_{end_date}.json')
        with open(file_path, 'w') as f:
            save_json = json.dumps(sorted_data)
            f.write(save_json)

    return timeunit

if __name__ == "__main__":   
    options, args = parse_command_line()

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    logging.basicConfig(level=logging.DEBUG, filename=f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/community_inequality_logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    chrono      = SimpleChrono()
    # chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}_2022.cfg")

    chrono.print(message="init")

    chrono.add_tic('proc')

    start_date = datetime.strptime(options.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(options.end_date, "%Y-%m-%d").date()
    print(f'start_date is set as: {start_date}')
    print(f'end_date is set as: {end_date}')
    
    datelist = daterange(start_date, end_date, by=options.frequency)
    tqdm_bar = tqdm(datelist, desc="processed files")

    x_values = []
    data_dicts = {}

    # Create a ThreadPoolExecutor
    with tqdm(total=len(datelist)) as progress:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Define a helper function to update the progress bar
            def update_progress(future):
                progress.update()
                progress.set_description(f"week of '{timeunit.strftime('%Y-%m-%d')} took {chrono.elapsed('proc')} sec", refresh=True)

            # Process timeunits in parallel
            futures = []
            for timeunit in datelist:
                future = executor.submit(process_timeunit, timeunit)
                future.add_done_callback(update_progress)
                futures.append(future)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)
    
    # # Load the data from the saved JSON files
    # data = {}
    # for key in data_dicts.keys():
    #     file_path = os.path.join(f'jsonResults_v3/h{options.heuristic}/community_new', f'{key}_2009-01-03_{end_date}.json')
    #     with open(file_path, 'r') as f:
    #         data[key] = json.load(f)

    # dates = matplotlib.dates.date2num(x_values)
    # fig = matplotlib.pyplot.figure(figsize=(16, 9), dpi=100)
    # matplotlib.pyplot.style.use('seaborn-darkgrid')
    # matplotlib.pyplot.legend(loc="upper left")
    # # Plot the data from the loaded JSON files
    # matplotlib.pyplot.plot_date(dates, list(data["real_DR"].values()), '-', linewidth=4, color='black', label="real_DR_modularity")
    # matplotlib.pyplot.plot_date(dates, list(data["real_SBM"].values()), '-', linewidth=4, color='dimgray', label="real_SBM_modularity")
    # matplotlib.pyplot.plot_date(dates, list(data["random_DR"].values()), '-', linewidth=4, color='gray', label="random_DR_modularity")
    # matplotlib.pyplot.plot_date(dates, list(data["random_SBM"].values()), '-', linewidth=4, color='lightgray', label="random_SBM_modularity")
    # matplotlib.pyplot.plot_date(dates, list(data["maximum_modularity"].values()), '-', linewidth=4, color='whitesmoke', label="maximum_modularity")
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.gca().set_title("Modularity Scores")
    # matplotlib.pyplot.savefig(f'jsonResults_v3/h{options.heuristic}/community/Modularity_Plot.png', dpi=100)
    # plt.close(fig)

    print('Process terminated, graphs and attributes created.')
    print(f"Graphs created in {chrono.elapsed('proc', format='%H:%M:%S')}")

        
