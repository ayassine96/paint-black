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

def community_modularity_analysis(date):

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    savelocation = f"/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/heur_{options.heuristic}_networks_full_shifted_community/{switcherback[options.frequency]}"
    unitsavelocation = f"{savelocation}/{date.strftime('%Y-%m-%d')}.graphml.bz2"

    # if os.path.exists(unitsavelocation):
    #     logging.info(f'building the date:{date} has started but the file exists so it will shutdown')
    #     return 0.0,0.0,0.0,0.0,0.0,0.0
    
    logging.info(f'Building communities for the week of:{date} has started')

    start_time = datetime.now()
    
    chrono.add_tic("net")

    networks_path = f"/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/heur_{options.heuristic}_networks_full/{switcherback[options.frequency]}"
    unit_graph_file = f"{networks_path}/{date.strftime('%Y-%m-%d')}.graphml.bz2"

    if not os.path.exists(unit_graph_file):
        logging.info(f'community building the date:{date} is unsuccesful since original network does not exist')
        return 0.0,0.0,0.0,0.0,0.0,[0.0,0.0]
    
    # Load Graph
    try:
        g = gt.load_graph(unit_graph_file)      
        ER = gt.GraphView(g, directed=True) # Get graph view for random version
    except OSError:
        logging.info(f'community building of the date:{date} is unsuccesful because of OSError')
        return 0.0,0.0,0.0,0.0,0.0,[0.0,0.0]
    
    
    state = gt.minimize_blockmodel_dl(g)
    blocks = state.get_blocks()
    

    # Iterate over the vertices and assign the block values as vertex attributes
    block_property = g.new_vertex_property("int")
    for vertex in g.vertices():
        block_property[vertex] = blocks[vertex]
        print(f'vertex:{vertex} , block {blocks[vertex]}')

    # Add the block_property as a vertex attribute
    g.vp["block"] = block_property

    # Calculate real modulariy
    real_DR_modularity = gt.modularity(g, g.vp["dark_ratio"], weight=g.ep["value"])
    real_SBM_modularity = gt.modularity(g, g.vp["block"], weight=g.ep["value"])

    # Randomize: Rewire the undirected graph using the random_rewire() function
    gt.random_rewire(ER, model="erdos")

    state_random = gt.minimize_blockmodel_dl(ER)
    blocks_random = state_random.get_blocks()

    # Iterate over the vertices and assign the block values as vertex attributes
    block_property_random = ER.new_vertex_property("int")
    for vertex in ER.vertices():
        block_property_random[vertex] = blocks_random[vertex]
        # print(f'vertex:{vertex} , block {blocks[vertex]}')

    # Add the block_property as a vertex attribute
    ER.vp["block"] = block_property_random

    # Calculate random modulariy
    random_DR_modularity = gt.modularity(ER, ER.vp["dark_ratio"], weight=ER.ep["value"])
    random_SBM_modularity = gt.modularity(ER, ER.vp["block"], weight=ER.ep["value"])

    # Perform modularity maximization algorithm for directed graphs
    state_maximal = gt.minimize_blockmodel_dl(g, state=gt.ModularityState)
    maximum_modularity = state_maximal.modularity()

    # calculate clusterring
    clustering = gt.global_clustering(g, weight=g.ep["value"])

    g.gp["real_DR_modularity"] = g.new_graph_property("float", real_DR_modularity)
    g.gp["real_SBM_modularity"] = g.new_graph_property("float", real_SBM_modularity)
    g.gp["random_DR_modularity"] = g.new_graph_property("float", random_DR_modularity)
    g.gp["random_SBM_modularity"] = g.new_graph_property("float", random_SBM_modularity)
    g.gp["maximum_modularity"] = g.new_graph_property("float", maximum_modularity)

    g.save(unitsavelocation)

    logging.info(f'Building for the date:{date} has finished with t={datetime.now() - start_time} finished:')
    logging.info(f"     Original graph: {g.vertices()} nodes, {g.num_edges()} edges")
    logging.info(f"     ER random graph: {ER.vertices()} nodes, {ER.num_edges()} edges")


    tqdm_bar.set_description(f"{switcherback[options.frequency]} of '{date.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)

    return real_DR_modularity, real_SBM_modularity, random_DR_modularity, random_SBM_modularity, maximum_modularity, clustering



if __name__ == "__main__":   
    options, args = parse_command_line()

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    logging.basicConfig(level=logging.DEBUG, filename=f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/community_logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
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
    real_DR_total, real_SBM_total, random_DR_total, random_SBM_total, maximum_modularity_total, clustering_total, clustering_std_total = ([] for i in range(7))

    for timeunit in tqdm_bar:

        # Run randomizer + assortativity builder and store result
        real_DR_unit, real_SBM_unit, random_DR_unit, random_SBM_unit, maximum_modularity_unit, clustering_unit = community_modularity_analysis(timeunit)

        x_values.append(timeunit.strftime('%Y-%m-%d'))

        real_DR_total.append(float(real_DR_unit))
        real_SBM_total.append(float(real_SBM_unit))
        random_DR_total.append(float(random_DR_unit))
        random_SBM_total.append(float(random_SBM_unit))
        maximum_modularity_total.append(float(maximum_modularity_unit))
        clustering_total.append(float(clustering_unit[0]))
        clustering_std_total.append(float(clustering_unit[1]))


        tqdm_bar.set_description(f"week of '{timeunit.strftime('%Y-%m-%d')} took {chrono.elapsed('proc')} sec", refresh=True)

    with open(f'jsonResults_v3/h{options.heuristic}/community/real_DR_modularity_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, real_DR_total))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/community/real_SBM_modularity_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, real_SBM_total))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/community/random_DR_modularity_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, random_DR_total))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/community/random_SBM_modularity_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, random_SBM_total))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/community/maximum_modularity_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, maximum_modularity_total))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/community/global_clustering_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, clustering_total))
        save_json = json.dumps(results_dict)
        f.write(save_json)
    
    dates = matplotlib.dates.date2num(x_values)
    fig = matplotlib.pyplot.figure(figsize=(16, 9), dpi=100)
    matplotlib.pyplot.style.use('seaborn-darkgrid')
    matplotlib.pyplot.legend(loc="upper left")
    matplotlib.pyplot.plot_date(dates, real_DR_total, 'k-', color='black', linewidth=4, label="real_DR_modularity")
    matplotlib.pyplot.plot_date(dates, real_SBM_total, 'k-', color='dimgray', linewidth=4, label="real_SBM_modularity")
    matplotlib.pyplot.plot_date(dates, random_DR_total, 'k-', color='gray', linewidth=4, label="random_DR_modularity")
    matplotlib.pyplot.plot_date(dates, random_SBM_total, 'k-', color='lightgray', linewidth=4, label="random_SBM_modularity")
    matplotlib.pyplot.plot_date(dates, maximum_modularity_total, 'k-', color='whitesmoke', linewidth=4, label="maximum_modularity")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.gca().set_title("Modularity Scores")
    matplotlib.pyplot.savefig(f'jsonResults_v3/h{options.heuristic}/community/Modularity_Plot.png', dpi=100)
    plt.close(fig)

    print('Process terminated, graphs and attributes created.')
    print(f"Graphs created in {chrono.elapsed('proc', format='%H:%M:%S')}")

        
