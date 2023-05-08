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
                
def daterange(date1, date2, by=1):
    return [  date1 + timedelta(n) for n in range(0, int((date2 - date1).days)+1, by) ]         

def build_random_network_with_attributes(date):

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    savelocation = f"/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/heur_{options.heuristic}_networks_full_random/{switcherback[options.frequency]}"
    unitsavelocation = f"{savelocation}/{date.strftime('%Y-%m-%d')}.graphml.bz2"

    if os.path.exists(unitsavelocation):
        logging.info(f'building the date:{date} has started but the file exists so it will shutdown')
        return -2.0
    
    logging.info(f'Building attributes for the date:{date} has started')

    start_time = datetime.now()
    
    chrono.add_tic("net")
    g = nx.DiGraph()

    networks_path = f"/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/heur_{options.heuristic}_networks_full/{switcherback[options.frequency]}"
    unit_graph_file = f"{networks_path}/{date.strftime('%Y-%m-%d')}.graphml.bz2"

    if not os.path.exists(unit_graph_file):
        logging.info(f'building the date:{date} is unsuccesful since original network does not exist')
        return -2.0
    
    try:
        g = nx.read_graphml(unit_graph_file)
    except OSError:
        logging.info(f'assortativity building of the date:{date} is unsuccesful because of OSError')
        return -2.0

    # Create a new empty directed graph
    ER = nx.DiGraph()

    # Copy over the nodes and their attributes from the original graph to the new graph
    for node, attrs in g.nodes(data=True):
        ER.add_node(node, **attrs)
    
    # Randomly assign one incoming or outgoing edge to each node
    while ER.number_of_edges() < g.number_of_edges():

        for node in ER.nodes():
            if ER.number_of_edges() >= g.number_of_edges():
                break

            if random.random() < 0.5:
                # Assign outgoing edge
                neighbors = list(ER.nodes() - {node})
                v = random.choice(neighbors)
                ER.add_edge(node, v)
            else:
                # Assign incoming edge
                neighbors = list(ER.nodes() - {node})
                u = random.choice(neighbors)
                ER.add_edge(u, node)

    logging.info(f'Building for the date:{date} has finished with t={datetime.now() - start_time} finished:')
    logging.info(f"     Original graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    logging.info(f"     ER random graph: {ER.number_of_nodes()} nodes, {ER.number_of_edges()} edges")

    DR_color_assortativity = nx.attribute_assortativity_coefficient(g, "color")
        
    if math.isnan(DR_color_assortativity):
        DR_color_assortativity = -2.0
    
    ER.graph['DR_color_assortativity'] = DR_color_assortativity

    logging.info(f'Computing assortativity for the date:{date} has finished with t={datetime.now() - start_time} finished')

    nx.write_graphml(ER, unitsavelocation)

    tqdm_bar.set_description(f"{switcherback[options.frequency]} of '{date.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)

    return DR_color_assortativity



if __name__ == "__main__":   
    options, args = parse_command_line()

    switcherback = {1:"day", 7:"week", 14:"2weeks", 28:"4weeks"}

    logging.basicConfig(level=logging.DEBUG, filename=f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/random_networkbuilder_logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
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

        # Run randomizer + assortativity builder and store result
        assort_result = build_random_network_with_attributes(timeunit)

        x_values.append(timeunit.strftime('%Y-%m-%d'))
        y_values_assortativity.append(float(assort_result))

        tqdm_bar.set_description(f"week of '{timeunit.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)

    with open(f'jsonResults_v3/h{options.heuristic}/random/random_assortativity_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, y_values_assortativity))
        save_json = json.dumps(results_dict)
        f.write(save_json)
    
    dates = matplotlib.dates.date2num(x_values)
    fig = matplotlib.pyplot.figure(figsize=(16, 9), dpi=100)
    matplotlib.pyplot.style.use('seaborn-darkgrid')
    matplotlib.pyplot.legend(loc="upper left")
    matplotlib.pyplot.plot_date(dates, y_values_assortativity, 'kx', color='black', linewidth=3)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.gca().set_title("DR Attribute Assortativity")
    matplotlib.pyplot.savefig(f'jsonResults_v3/h{options.heuristic}/graphs/Random_AssortativityPlot.png', dpi=100)
    plt.close(fig)

    print('Process terminated, graphs and attributes created.')
    print(f"Graphs created in {chrono.elapsed('proc', format='%H:%M:%S')}")

        
