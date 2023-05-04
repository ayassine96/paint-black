#!/usr/bin/env python3

# input:
#     - `{options.black_data_folder}/cluster_is_black_ground_truth.zarr` ground truth clusters from `ub_ground_truth.py`
#     - `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
#     - `{DIR_PARSED}/{options.currency}.cfg` blockchain data
# outputs:
#     * zarr file: `cluster_is_black_when_block.zarr` index is cluster id, value is int block when the cluster became black which can also represent time.

# results 22 json files, 3 figures

import blocksci
import sys, os, os.path, socket
import numpy as np
import zarr
import time
import pandas as pd
from collections import defaultdict
from decimal import Decimal
import math
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates
import json
import logging
from itertools import compress

from util import SYMBOLS, DIR_BCHAIN, DIR_PARSED, SimpleChrono, darknet

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]

def parse_command_line():
    import sys, optparse

    parser = optparse.OptionParser()

    parser.add_option("--curr", action='store', dest="currency", type='str',
                                              default=None, help="name of the currency")
    parser.add_option("--heur", action='store', dest="heuristic", type='str',
                                                  default=None, help="heuristics to apply")
    parser.add_option("--output", action='store', dest = "output_folder", 
                        default="uniform_black/", type='str', help='directory to save outputs in')
    parser.add_option("--start", action="store", dest="start_date",
                                   default = None, help= "starting date for network creation in YYYY-MM-DD format")
    parser.add_option("--end", action="store", dest="end_date",
                                       default = None, help = "ending date for network creation in YYYY-MM-DD format")
    parser.add_option("--freq", action="store", dest="frequency",
                       default = "day", help = "time aggregation of networks - choose between day, week, 2weeks, 4weeks")

    switcher = {"day":1, "week":7, "2weeks":14, "4weeks":28}

    options, args = parser.parse_args()

    options.currency = SYMBOLS[options.currency]

    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}/"

    options.cluster_data_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/"

    options.output_folder = f"{options.output_folder}/heur_{options.heuristic}_data/"

    options.frequency = switcher[options.frequency]

    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    # atm ground truth is in the output folder
    options.black_data_folder = options.output_folder

    return options, args


class AddressMapper():
    def __init__(self, chain):
        self.chain = chain

        self.__address_types = [blocksci.address_type.nonstandard, blocksci.address_type.pubkey,
                                blocksci.address_type.pubkeyhash, blocksci.address_type.multisig_pubkey,
                                blocksci.address_type.scripthash, blocksci.address_type.multisig,
                                blocksci.address_type.nulldata, blocksci.address_type.witness_pubkeyhash,
                                blocksci.address_type.witness_scripthash, blocksci.address_type.witness_unknown]

        self.__counter_addresses = { _:self.chain.address_count(_) for _ in self.__address_types }

        self.__offsets = {}
        offset = 0
        for _ in self.__address_types:
            self.__offsets[_] = offset
            offset += self.__counter_addresses[_]


        self.total_addresses = offset
        print(f"[INFO] #addresses: {self.total_addresses}")
#        print(self.__counter_addresses)


    def map_clusters(self,cm):
        cluster_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types }

        self.cluster = np.zeros(self.total_addresses, dtype=np.int64)
        offset = 0
        for _at in cluster_vector.keys():
            clusters = cluster_vector[_at]
            print(f"{_at}     -  {len(clusters)}")
            for _i, _add in enumerate(chain.addresses(_at)):
                clusters[_i] = cm.cluster_with_address(_add).index
        offset = 0
        for _ in cluster_vector.keys():
            v = cluster_vector[_]
            self.cluster[offset:offset + len(v)] = v
            offset += len(v)

    def dump_clusters(self, output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        zarr.save(f"{output_folder}/address_cluster_map.zarr", self.cluster)


    def load_clusters(self, input_folder):
        self.cluster = zarr.load(f"{input_folder}/address_cluster_map.zarr")

    def __getitem__(self,addr):
        return self.__offsets[addr.raw_type]+ addr.address_num-1

def daterange(date1, date2, by=1):
    return [  date1 + timedelta(n) for n in range(0, int((date2 - date1).days)+1, by) ]

if __name__ == "__main__":

    options, args = parse_command_line()

    # logging.basicConfig(level=logging.DEBUG, filename=f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/plotting_logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    # Start Chrono
    chrono = SimpleChrono()

    # Load chain and initialize address mapper
    
    if socket.gethostname() == 'abacus-1':
        chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}_2022.cfg")
    elif socket.gethostname() == 'consensus-2':
        chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}_2022.cfg")
    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}")

    # PRE-PROCESSING
    # define blocks range after given dates
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
    weeksList = daterange(start_date, end_date, by= options.frequency)

    tqdm_bar = tqdm(weeksList, desc="processed files")



    chrono.print(message="init")
    print(f"[INF] Below is the ground_truth_clust_id dark entities for heuristic {options.heuristic}...")
    df = pd.read_csv(f"uniform_black/heur_{options.heuristic}_data/ground_truth_clust_id.csv")
    cluster_table = df.groupby(['entity'])["cluster_id"].unique()
    print(cluster_table)

    print(f"[CALC] Starting the Plotting for heuristic {options.heuristic} ...")

    # x_values represents timeunit
    x_values = []
    # Collect total CA  
    y_values_total_CA = []
    # Collect total DA
    y_values_total_DA = []
    # Collect values from DR
    y_values_black = []
    y_values_dark_grey = []
    y_values_grey = []
    y_values_light_grey = []
    y_values_greyish_white = []
    y_values_white = []
    # Collect values for each darknet market
    y_values_darknets = defaultdict(list)

    # for each week load data to save in json
    for timeunit in tqdm_bar:
        chrono.add_tic("net")

        # CA Zarr file for current timeunit
        current_assets_zarr = zarr.load(f'/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/weekly/current_assets/current_assets_{timeunit}.zarr') #load gives an object array containing this dictionary:
        current_assets_zarr = dict(zip(current_assets_zarr["current_assets_index"], current_assets_zarr["current_assets_values"]))

        # DA Zarr file for current timeunit
        dark_assets_zarr = zarr.load(f'/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/weekly/dark_assets/dark_assets_{timeunit}.zarr') #load gives an object array containing this dictionary:
        dark_assets_zarr = dict(zip(dark_assets_zarr["dark_assets_index"], dark_assets_zarr["dark_assets_values"]))

        # DR Zarr file for current timeunit
        dark_ratio_zarr = zarr.load(f'/srv/abacus-1/bitcoin_darknet/grayscale_op_ali/heur_{options.heuristic}_data_v3/weekly/dark_ratio/dark_ratio_{timeunit}.zarr') #load gives an object array containing this dictionary:
        dark_ratio_zarr = dict(zip(dark_ratio_zarr["dark_ratio_index"], dark_ratio_zarr["dark_ratio_values"]))

        # x_values set to timeunit
        x_values.append(timeunit.strftime('%Y-%m-%d'))

        # Caclculate total CA in Satoshis
        y_values_total_CA.append(float(sum(current_assets_zarr.values())) * 0.00000001)

        # Caclculate total DA in Satoshis
        y_values_total_DA.append(float(sum(dark_assets_zarr.values())) * 0.00000001)

        # plot darknet market total assets
        for dnm, list in cluster_table.items():
            y_total = 0.0
            cluster_list = list
            for cluster in cluster_list:
                if cluster in current_assets_zarr.keys():
                    y_total = y_total + float(current_assets_zarr[cluster])
                else:
                    y_total += 0.0
            
            y_values_darknets[dnm].append(y_total)

        # count distribution over DR
        count_black = 0
        count_dark_grey = 0
        count_grey = 0
        count_light_grey = 0
        count_greyish_white = 0
        count_white = 0

        size = len(dark_ratio_zarr)

        for v in dark_ratio_zarr.values():
            if v == 1.0:
                count_black += 1
            elif 0.75 <= v < 1.0 :
                count_dark_grey += 1
            elif 0.5 <= v < 0.75 :
                count_grey += 1
            elif 0.25 <= v < 0.5 :
                count_light_grey += 1
            elif 0 < v < 0.25 :
                count_greyish_white += 1
            elif v == 0.0 :
                count_white += 1
        
        y_values_black.append(count_black / size)
        y_values_dark_grey.append(count_dark_grey / size)
        y_values_grey.append(count_grey / size)
        y_values_light_grey.append(count_light_grey / size)
        y_values_greyish_white.append(count_greyish_white / size)
        y_values_white.append(count_white / size)

        tqdm_bar.set_description(f"week of '{timeunit.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)


    # Saving in JSON
    # Save total CA in btc 
    with open(f'jsonResults_v3/h{options.heuristic}/Total_CA_2009-01-03_{end_date}.json', 'w') as f:
        # total_CA_btc = [float(value * 0.00000001) for value in y_values_total_CA] # convert
        results_dict = dict(zip(x_values, y_values_total_CA))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    # Save total DA in btc 
    with open(f'jsonResults_v3/h{options.heuristic}/Total_DA_2009-01-03_{end_date}.json', 'w') as f:
        # total_DA_btc = [float(value * 0.00000001) for value in y_values_total_DA] # convert
        results_dict = dict(zip(x_values, y_values_total_DA))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    # Save Dark ratio plots
    with open(f'jsonResults_v3/h{options.heuristic}/DR_black_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, y_values_black))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/DR_dark_grey_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, y_values_dark_grey))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/DR_grey_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, y_values_grey))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/DR_light_grey_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, y_values_light_grey))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/DR_greyish_white_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, y_values_greyish_white))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    with open(f'jsonResults_v3/h{options.heuristic}/DR_white_2009-01-03_{end_date}.json', 'w') as f:
        results_dict = dict(zip(x_values, y_values_white))
        save_json = json.dumps(results_dict)
        f.write(save_json)

    # save dnm plots
    for dnm, list in cluster_table.items():
        with open(f'jsonResults_v3/h{options.heuristic}/total_assets_{dnm}_2009-01-03_{end_date}.json', 'w') as f:
            results_dict = dict(zip(x_values, y_values_darknets[dnm]))
            save_json = json.dumps(results_dict)
            f.write(save_json)



    # Saving plotted figures/images
    # Plotting CA and DA on same figure
    x_values = []
    y_values = []
    y_values_DA = []

    with open(f'jsonResults_v3/h{options.heuristic}/Total_CA_{start_date}_{end_date}.json', 'r') as f:
        load_json = json.load(f)
        for iterator in load_json:
            x_values.append(iterator)
            y_values.append(float(load_json[iterator]))

    with open(f'jsonResults_v3/h{options.heuristic}/Total_DA_{start_date}_{end_date}.json', 'r') as f:
        load_json = json.load(f)
        for iterator in load_json:
            y_values_DA.append(float(load_json[iterator]))

    dates = matplotlib.dates.date2num(x_values)
    fig = matplotlib.pyplot.figure(figsize=(16, 9), dpi=100)
    matplotlib.pyplot.style.use('seaborn-darkgrid')
    matplotlib.pyplot.legend(loc="upper left")
    matplotlib.pyplot.plot_date(dates, y_values, 'k-', color='green', linewidth=3, label="Total Circulating Satoshis")
    matplotlib.pyplot.plot_date(dates, y_values_DA, 'k-', color='black', linewidth=3, label="Total Circulating Dark Satoshis")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.gca().set_title("Total Assets & Total Dark Assets")

    matplotlib.pyplot.savefig(f'jsonResults_v3/h{options.heuristic}/graphs/TotalAssets_plot.png', dpi=100)
    plt.close(fig)

    # Plotting DNMs on same figure
    fig, ax = plt.subplots(nrows=7, ncols=2, figsize=(15, 15))
    fig.tight_layout()
    plot_number = 1

    for dnm, list in cluster_table.items():
        x_values = []
        y_values = []
        with open(f'jsonResults_v3/h{options.heuristic}/total_assets_{dnm}_{start_date}_{end_date}.json', 'r') as f:
            load_json = json.load(f)
            for iterator in load_json:
                x_values.append(iterator)
                y_values.append(float(load_json[iterator]))
            
            dates = matplotlib.dates.date2num(x_values)
            plt.subplot(7, 2, plot_number)
            plt.gca().set_title(f'{dnm}')
            plt.plot_date(dates, y_values, 'k-')
            plot_number += 1
    
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
    plt.style.use('seaborn-darkgrid')
    fig.savefig(f'jsonResults_v3/h{options.heuristic}/graphs/DNMsPlot.png', dpi=100)
    plt.plot_date(dates, y_values, 'k-')
    plt.close(fig)

    # # Plotting DR percentages on same figure
    dates = matplotlib.dates.date2num(x_values)
    fig = matplotlib.pyplot.figure(figsize=(16, 9), dpi=100)
    matplotlib.pyplot.style.use('seaborn-darkgrid')
    matplotlib.pyplot.legend(loc="upper left")
    matplotlib.pyplot.plot_date(dates, y_values_black, 'k-', color='black', linewidth=4, label="dark ratio = 1")
    matplotlib.pyplot.plot_date(dates, y_values_dark_grey, 'k-', color='dimgray', linewidth=4, label="0.75 <= dark ratio < 1.0")
    matplotlib.pyplot.plot_date(dates, y_values_grey, 'k-', color='gray', linewidth=4, label="0.5 <= dark ratio < 0.75")
    matplotlib.pyplot.plot_date(dates, y_values_light_grey, 'k-', color='lightgray', linewidth=4, label="0.25 <= dark ratio < 0.5")
    matplotlib.pyplot.plot_date(dates, y_values_greyish_white, 'k-', color='whitesmoke', linewidth=4, label="0 < dark ratio < 0.25")
    matplotlib.pyplot.plot_date(dates, y_values_white, 'k-', color='white', linewidth=4, label="dark ratio = 0.0")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.gca().set_title("Dark Ratio Distribution")
    matplotlib.pyplot.savefig(f'jsonResults_v3/h{options.heuristic}/graphs/DarkRatioDistribution.png', dpi=100)
    plt.close(fig)

    chrono.print(message="took", tic="last")


