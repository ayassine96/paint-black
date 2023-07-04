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
import gc
from util import SYMBOLS, DIR_PARSED, SimpleChrono
import math
import matplotlib.pyplot as plt
import matplotlib.dates
import json


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
    
    if not os.path.exists(options.networks_folder):
        os.mkdir(options.networks_folder)


    return options, args

            
    
def build_CA_dictionary(asset):

    current_assets_zarr = zarr.load(asset)
    
    current_assets_dict = defaultdict(lambda: 0, dict(zip(current_assets_zarr["current_assets_index"], current_assets_zarr["current_assets_values"])))
        
    return current_assets_dict

def build_DA_dictionary(asset):

    dark_assets_zarr = zarr.load(asset)
    
    dark_assets_dict = defaultdict(lambda: 0, dict(zip(dark_assets_zarr["dark_assets_index"], dark_assets_zarr["dark_assets_values"])))
        
    return dark_assets_dict
             
def build_DR_dictionary(ratio):

    dark_ratio_zarr = zarr.load(ratio)
    
    dark_ratio_dict = defaultdict(lambda: 0, dict(zip(dark_ratio_zarr["dark_ratio_index"], dark_ratio_zarr["dark_ratio_values"])))
    
    return dark_ratio_dict

def load_dictionaries(date, heur, freq):

    
    if freq == "day":
        freq = "daily"
    elif freq == "week":
        freq = "weekly"

    loadlocation = f"/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{heur}_data_v3/{freq}"

    current_assets_file = f"{loadlocation}/current_assets/current_assets_{date}.zarr"

    dark_assets_file = f"{loadlocation}/dark_assets/dark_assets_{date}.zarr"
    
    dark_ratio_file = f"{loadlocation}/dark_ratio/dark_ratio_{date}.zarr"
    
    if os.path.exists(current_assets_file):
        current_assets_dict = build_CA_dictionary(current_assets_file)

    if os.path.exists(dark_assets_file):
        dark_assets_dict = build_DA_dictionary(dark_assets_file)
    
    if os.path.exists(dark_ratio_file):
        dark_ratios_dict = build_DR_dictionary(dark_ratio_file)
    
    return current_assets_dict,dark_ratios_dict,dark_assets_dict
                    
                
                
def daterange(date1, date2, by=1):
    return [  date1 + timedelta(n) for n in range(0, int((date2 - date1).days)+1, by) ]         

def build_network_with_attributes(current_date):

    previous_date = current_date - timedelta(options.frequency)

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    savelocation = f"/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/heur_{options.heuristic}_networks_full_shifted/{switcherback[options.frequency]}"
    unitsavelocation = f"{savelocation}/{current_date.strftime('%Y-%m-%d')}.graphml.bz2"

    if os.path.exists(unitsavelocation):
        logging.info(f'building the date:{current_date} has started but the file exists so it will shutdown')
        return "Already exists"
    
    logging.info(f'Building attributes for the date:{current_date} has started')

    start_time = datetime.now()
    
    chrono.add_tic("net")
    g = nx.DiGraph()

    # Load current day network (nodes+edges)
    networks_path = f"/srv/consensus-2/blockchain_parsed/bitcoin/heur_{options.heuristic}_networks_{switcherback[options.frequency]}"
    unit_graph_file = f"{networks_path}/{current_date.strftime('%Y-%m-%d')}.graphml.bz2"

    if not os.path.exists(unit_graph_file):
        logging.info(f'building the date:{current_date} is unsuccesful since original network does not exist')
        return -2.0
    
    try:
        g = nx.read_graphml(unit_graph_file)
    except OSError:
        logging.info(f'assortativity building of the date:{current_date} is unsuccesful because of OSError')
        return -2.0

    current_assets_dict_full, dark_ratios_dict_full, dark_assets_dict_full = load_dictionaries(previous_date, options.heuristic, switcherback[options.frequency])

    current_assets_dict_full_new, dark_ratios_dict_full_new, dark_assets_dict_full_new = load_dictionaries(current_date, options.heuristic, switcherback[options.frequency])

    list_of_nodes = list(g.nodes)

    current_assets_dict_filtered = { k: current_assets_dict_full[int(k)] for k in list_of_nodes }
    dark_ratios_dict_filtered = { k: dark_ratios_dict_full[int(k)] for k in list_of_nodes }
    dark_assets_dict_filtered = { k: dark_assets_dict_full[int(k)] for k in list_of_nodes }

    # current_assets_dict_filtered_new = { k: current_assets_dict_full_new[int(k)] for k in list_of_nodes }
    dark_ratios_dict_filtered_new = { k: dark_ratios_dict_full_new[int(k)] for k in list_of_nodes }
    # dark_assets_dict_filtered_new = { k: dark_assets_dict_full_new[int(k)] for k in list_of_nodes }

    del current_assets_dict_full
    del dark_ratios_dict_full
    del dark_assets_dict_full
    del current_assets_dict_full_new
    del dark_ratios_dict_full_new
    del dark_assets_dict_full_new

    for node in g.nodes():

        if current_assets_dict_filtered[node] > 0:
        
            nx.set_node_attributes(g, {node:int(current_assets_dict_filtered[node])}, 'current_assets')

            nx.set_node_attributes(g, {node:int(dark_assets_dict_filtered[node])}, 'dark_assets')
            
            nx.set_node_attributes(g, {node:float(dark_ratios_dict_filtered[node])}, 'dark_ratio')

            if dark_ratios_dict_filtered[node] == 1.0:
                nx.set_node_attributes(g, {node:"black"}, 'color')
            elif 0.75 <= dark_ratios_dict_filtered[node] < 1.0 :
                nx.set_node_attributes(g, {node:"dark_grey"}, 'color')
            elif 0.5 <= dark_ratios_dict_filtered[node] < 0.75 :
                nx.set_node_attributes(g, {node:"grey"}, 'color')
            elif 0.25 <= dark_ratios_dict_filtered[node] < 0.5 :
                nx.set_node_attributes(g, {node:"light_grey"}, 'color')
            elif 0 < dark_ratios_dict_filtered[node] < 0.25 :
                nx.set_node_attributes(g, {node:"greyish_white"}, 'color')
            elif dark_ratios_dict_filtered[node] == 0.0 :
                nx.set_node_attributes(g, {node:"white"}, 'color')  
        
        elif current_assets_dict_filtered[node] == 0:

            average_DR = (dark_ratios_dict_filtered_new[node] + dark_ratios_dict_filtered[node]) / 2
            
            nx.set_node_attributes(g, {node:float(average_DR)}, 'dark_ratio')

            if average_DR == 1.0:
                nx.set_node_attributes(g, {node:"black"}, 'color')
            elif 0.75 <= average_DR < 1.0 :
                nx.set_node_attributes(g, {node:"dark_grey"}, 'color')
            elif 0.5 <= average_DR < 0.75 :
                nx.set_node_attributes(g, {node:"grey"}, 'color')
            elif 0.25 <= average_DR < 0.5 :
                nx.set_node_attributes(g, {node:"light_grey"}, 'color')
            elif 0 < average_DR < 0.25 :
                nx.set_node_attributes(g, {node:"greyish_white"}, 'color')
            elif average_DR == 0.0 :
                nx.set_node_attributes(g, {node:"white"}, 'color')
        else:
            continue

    del current_assets_dict_filtered
    del dark_ratios_dict_filtered
    del dark_assets_dict_filtered
    # del current_assets_dict_filtered_new
    del dark_ratios_dict_filtered_new
    # del dark_assets_dict_filtered_new
    gc.collect()

    logging.info(f'Building for the date:{current_date} has finished with t={datetime.now() - start_time} finished')

    DR_color_assortativity = nx.attribute_assortativity_coefficient(g, "color")
        
    if math.isnan(DR_color_assortativity):
        DR_color_assortativity = -2.0
    
    g.graph['DR_color_assortativity'] = DR_color_assortativity

    nx.write_graphml(g, unitsavelocation)

    logging.info(f'Computing assortativity for the date:{current_date} has finished with t={datetime.now() - start_time} finished')

    tqdm_bar.set_description(f"{switcherback[options.frequency]} of '{current_date.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)

    return DR_color_assortativity



if __name__ == "__main__":   
    options, args = parse_command_line()

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    logging.basicConfig(level=logging.DEBUG, filename=f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/shifted_networkbuilder_logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    chrono      = SimpleChrono()
    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}_2022.cfg")

    chrono.print(message="init")

    chrono.add_tic('proc')
    if options.start_date == None:
        start_date = datetime.fromtimestamp(chain.blocks[0].timestamp).date()
    else:
        start_date = datetime.strptime(options.start_date, "%Y-%m-%d").date()
    if options.end_date == None:
        end_date = datetime.fromtimestamp(chain.blocks[-1].timestamp).date()
    else:
        end_date = datetime.strptime(options.end_date, "%Y-%m-%d").date()

    print(f'start_date is set as: {start_date}')
    print(f'end_date is set as: {end_date}')
    
    datelist = daterange(start_date, end_date, by=options.frequency)
    tqdm_bar = tqdm(datelist, desc="processed files")

    x_values = []
    y_values_assortativity = []

    for timeunit in tqdm_bar:

        assort_result = build_network_with_attributes(timeunit)

        x_values.append(timeunit.strftime('%Y-%m-%d'))
        y_values_assortativity.append(float(assort_result))

        tqdm_bar.set_description(f"week of '{timeunit.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)

    # with open(f'jsonResults_v3/h{options.heuristic}/NEW_assortativity_2009-01-03_{end_date}.json', 'w') as f:
    #     results_dict = dict(zip(x_values, y_values_assortativity))
    #     save_json = json.dumps(results_dict)
    #     f.write(save_json)
    
    # dates = matplotlib.dates.date2num(x_values)
    # fig = matplotlib.pyplot.figure(figsize=(16, 9), dpi=100)
    # matplotlib.pyplot.style.use('seaborn-darkgrid')
    # matplotlib.pyplot.legend(loc="upper left")
    # matplotlib.pyplot.plot_date(dates, y_values_assortativity, 'kx', color='black', linewidth=3)
    # matplotlib.pyplot.legend()
    # matplotlib.pyplot.gca().set_title("DR Attribute Assortativity")
    # matplotlib.pyplot.savefig(f'jsonResults_v3/h{options.heuristic}/graphs/NEW_AssortativityPlot.png', dpi=100)
    # plt.close(fig)

    print('Process terminated, graphs and attributes created.')
    print(f"Graphs created in {chrono.elapsed('proc', format='%H:%M:%S')}")

        
