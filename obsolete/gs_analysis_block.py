#!/usr/bin/env python3
"""
TODO:
# This script uses the results from the last two scripts to creates large csv of interesting features 
- add total_clusters in the block as union of clusters input and clusters output
OUTPUT
------
    - csv file, by row
        'block', # for each block b
        'no_old_black_clusters', # number of "old" black clusters (before b is added)
        'no_new_black_clusters', # number of new black clusters created in this block
        'no_black_clusters_input',  # number of new black clusters which appear as input in b   
        'no_black_clusters_output', # number of new black clusters which appear as output on n
        'no_old_black_clusters_output', # number of old black clusters as output in b
        'no_new_black_clusters_output', # number of new black clusters as output in b
        'no_black_clusters_cumulative', # At block b, what is the cumulative black clusters we found
        'no_active_black_clusters', # At block b, what the number of active (I/O) black clusters we found
        'no_white_clusters_input', # number of clusters which is not black as input
        'no_white_clusters_output', # number of clusters which is not black as output
        'no_active_white_clusters', # number of active (I/O) clusters which is not black
        'no_clusters_input', # number of clusters which are used as input in b
        'no_clusters_output', # number of clusters which are used as output in b
        'no_clusters_cumulative'  # cumulative number of all clusters at b
        'no_active_clusters',  # number of active clusters in b 
        'black2black_no_links', # number of links between b2b
        'black2white_no_links', # number of links between b2w
        'white2black_no_links', # number of links between w2b
        'white2white_no_links', # number of links between w2w
        'no_links', # total number of links
        'vol_black_trxs', # actual ammount of black bitcoin exchanged in that block
        'vol_white_trxs' # actual ammount of white bitcoin exchanged in that block
        'no_black_trxs', # number of black transactions
        'no_white_trxs', # number of white transactions
        'total_trxs' # total number of transactions
"""
# this file is responsible for creating the features from the previous data. It is unfinished, and we can add more features if we want.

import blocksci

import os
import os.path
import numpy as np
# import networkx as nx
import zarr
# import time
# from tqdm import tqdm
# import pandas as pd
from csv import DictWriter
# import pickle as pkl
from datetime import datetime
from itertools import compress

from util import SYMBOLS, DIR_PARSED, SimpleChrono


def parse_command_line(): # more options than before
    import optparse

    parser = optparse.OptionParser()

    parser.add_option("--curr", action='store', dest="currency", type='str',
                      default=None, help="name of the currency")
    parser.add_option("--heur", action='store', dest="heuristic", type='str',
                      default=None, help="heuristics to apply")
    parser.add_option("--overwrite", action='store_true', dest="overwrite")
    parser.add_option("--input", action='store', dest="input_file", # clust_is_black_when generated in previous script
                      default = None,
                      type='str', help='input file')
    parser.add_option("--output", action='store', dest="output_folder",
                      default="uniform_black/", type='str',
                      help='directory to save outputs in')
    parser.add_option("--start", action="store", dest="start_date",
                      default=None,
                      help="starting date for network creation in YYYY-MM-DD format")
    parser.add_option("--end", action="store", dest="end_date",
                      default=None,
                      help="ending date for network creation in YYYY-MM-DD format")

    options, args = parser.parse_args()

    options.currency = SYMBOLS[options.currency]


    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}/"

    options.cluster_data_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/"

    options.output_folder = f"{options.output_folder}/heur_{options.heuristic}_data/"
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    if options.input is None:
        options.input = "{options.output_folder}/cluster_is_black_when.zarr",
    """
    options.output_active_folder = options.output_folder + f'black_nodes_by_block/'
    if not os.path.exists(options.output_active_folder):
        os.mkdir(options.output_active_folder)
    """

    options.output_csv = f"{options.output_folder}/diffusion_analysis_block.csv"

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

        self.__counter_addresses = { _:self.chain.address_count(_) for _ in self.__address_types}

        self.__offsets = {}
        offset = 0
        for _ in self.__address_types:
            self.__offsets[_] = offset
            offset += self.__counter_addresses[_]

        self.total_addresses = offset
        print(f"[INFO] #addresses: {self.total_addresses}")
#        print(self.__counter_addresses)

    def map_clusters(self, cm):
        # address_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types }
        cluster_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types}

        self.cluster = np.zeros(self.total_addresses, dtype=np.int64)
        offset = 0
        for _at in cluster_vector.keys():
            clusters = cluster_vector[_at]
            print(f"{_at}     -  {len(clusters)}")
#            addrs = address_vector[_at]
            for _i, _add in enumerate(chain.addresses(_at)):
#                addrs[_i] = _add.address_num
                clusters[_i] = cm.cluster_with_address(_add).index
                #max_addr_num = max(max_addr_num, addrs[_i])
#        pickle.dump(cluster_vector, open("cluster_dict.pickle","wb"))

        offset = 0
        for _ in cluster_vector.keys():
            v = cluster_vector[_]
            self.cluster[offset:offset + len(v)] = v
            offset += len(v)

    def dump_clusters(self, output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        zarr.save(f"{output_folder}/address_cluster_map.zarr", self.cluster)


#    def dump_offsets(self, output_folder):
#        if not os.path.exists(output_folder):
#            os.mkdir(output_folder)
#        pickle.dump(self.__offsets, open(f"{output_folder}/offsets.pickle", "wb") )

#    def load_offsets(self, output_folder):
#        if not os.path.exists(output_folder):
#            os.mkdir(output_folder)
#        self.__offsets = pickle.load( open(f"{output_folder}/offsets.pickle", "rb") )

    def load_clusters(self, input_folder):
        self.cluster = zarr.load(f"{input_folder}/address_cluster_map.zarr")

    def __getitem__(self, addr):
        return self.__offsets[addr.raw_type] + addr.address_num-1


if __name__ == "__main__":
    options, args = parse_command_line()

    # heur_file   = zarr.open_array(f"{options.cluster_folder}_data/address_cluster_map.zarr", mode = 'r')
    # memstore    = zarr.MemoryStore()
    # zarr.copy_store(heur_file.store, memstore)
    # heur_mem    = zarr.open_array(memstore)
    chrono = SimpleChrono()

    # LOAD STUFF
    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}.cfg")

    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}")
    # here we need to load both ground truth and black when arrays
    # black_cluster: index-cluster, bool value-true if cluster is black
    clust_is_black_ground = zarr.load(f"{options.black_data_folder}/cluster_is_black_ground_truth.zarr")
    # cluster is black when
    clust_is_black_when = zarr.load(options.input)

    # PRE-PROCESSING
    # csv prepping
    col_names = [
        'block',
        'no_old_black_clusters',
        'no_new_black_clusters',
        'no_black_clusters_input',
        'no_black_clusters_output',
        'no_old_black_clusters_output',
        'no_new_black_clusters_output',
        'no_black_clusters_cumulative',
        'no_active_black_clusters',
        'no_white_clusters_input'
        'no_white_clusters_output'
        'no_active_white_clusters',
        'no_clusters_input',
        'no_clusters_output',
        'no_clusters_cumulative'
        'no_active_clusters'
        'black2black_no_links',
        'black2white_no_links',
        'white2black_no_links',
        'white2white_no_links',
        'no_links',
        'vol_black_trxs',
        'vol_white_trxs'
        'no_black_trxs',
        'no_white_trxs',
        'total_trxs'
    ]
    csv_fout = DictWriter(open(options.output_csv, "w"), fieldnames=col_names)

    # write header
    csv_fout.writeheader()

    # define blocks range after given dates
    if options.start_date is None:
        start_date = datetime.fromtimestamp(chain.blocks[0].timestamp).date()
    else:
        start_date = datetime.strptime(options.start_date, "%Y-%m-%d").date()
    if options.end_date is None:
        end_date = datetime.fromtimestamp(chain.blocks[-1].timestamp).date()
    else:
        end_date = datetime.strptime(options.end_date, "%Y-%m-%d").date()

    blocks_range = chain.range(start_date, end_date)

    # set of black users 
    # node type should change to str because in graphs nodes ids are str
    # clust_is_black_ground_set = set([str(i) for i in range(len(clust_is_black_ground)) if clust_is_black_ground[i]])
    clust_is_black_ground_set = set(compress(range(len(clust_is_black_ground)), clust_is_black_ground))
    # clust_is_black_cum_set = set([])
    # clust_is_black_cum = np.zeros(len(clust_is_black_ground), dtype=bool)

    chrono.print(message="init")

    print("[CALC] starts black bitcoin diffusion...")

    # the part below is unfinished and might not be correct
    no_black_clusters_cumulative = 0
    # RUN ON ALL BLOCKS
    for b in blocks_range:
        _ = {}
        # create empty temp sets
        old_black_nodes = set([])
        new_black_nodes = set([])
        
        # initialize all the feature variables
        black_old2new_no_links = 0
        black_old2old_no_links = 0
        white2black_no_links = 0
        white2white_no_links = 0
        not_black_no_links = 0
        no_links = 0

        no_black_trxs = 0
        no_white_trxs = 0

        no_black_clusters_input = 0
        no_black_clusters_output = 0
        no_old_black_clusters_output = 0
        no_new_black_clusters_output = 0

        no_clusters_input = 0
        no_clusters_output = 0

        # on a single trx
        for trx in b.txes:
            loc_old_black_nodes = set([])
            loc_new_black_nodes = set([])

            loc_no_black_clusters_input = 0
            loc_no_black_clusters_output = 0
            loc_no_old_black_clusters_output = 0
            loc_no_new_black_clusters_output = 0

            flag_input_black = False

            if trx.is_coinbase: continue

            tup_inputs = {}
            tup_outputs = {}

            # following 2 loops are like 7-blocksci-buil-networks
            # Builds reduced representation of inputs
            trx_input_value = 0
            for inp in trx.inputs:
                    cluster, value= am.cluster[am[inp.address]], inp.value
                    if cluster in tup_inputs:
                        tup_inputs[cluster] += value
                    else:
                        tup_inputs[cluster] = value
                    trx_input_value += value
            # Builds reduced representation of outputs
            for out in trx.outputs:
                    cluster, value= am.cluster[am[out.address]], out.value
                    if cluster in tup_outputs:
                        tup_outputs[cluster] += value
                    else:
                        tup_outputs[cluster] = value

            for inp in tup_inputs:
                # if the input is already black
                if inp in clust_is_black_ground_set:
                    # at least on of the input is black
                    flag_input_black = True
                    # add the black input to old black nodes
                    loc_old_black_nodes.add(inp)
                    loc_no_black_clusters_input += 1

            # if at least one input is black
            if flag_input_black:
                for out in tup_outputs:
                    loc_no_black_clusters_output += 1
                    if out in clust_is_black_ground_set:
                        # the cluster is already black
                        loc_old_black_nodes.add(out)
                        loc_no_old_black_clusters_output += 1
                    else:
                        # the cluster was not black before
                        # add the cluster to new black nodes
                        loc_new_black_nodes.add(out)
                        # update clust_is_black_when
                        clust_is_black_when[out] = b.height
                        loc_no_new_black_clusters_output += 1
            else:
                for out in tup_outputs:
                    if out in clust_is_black_ground_set:
                        loc_old_black_nodes.add(out)
                        loc_no_black_clusters_output += 1
                        loc_no_old_black_clusters_output += 1
                        # print(loc_no_black_clusters_output)

            # all trxs have been cheked, now we have to collect the information 
            # clusters
            loc_no_clusters_input = len(tup_inputs)
            loc_no_clusters_output = len(tup_outputs)
            no_clusters_input += loc_no_clusters_input
            no_clusters_output += loc_no_clusters_output
            no_black_clusters_input += loc_no_black_clusters_input
            no_black_clusters_output += loc_no_black_clusters_output
            no_old_black_clusters_output += loc_no_old_black_clusters_output
            no_new_black_clusters_output += loc_no_new_black_clusters_output
            # links    
            black_old2new_no_links += loc_no_black_clusters_input*loc_no_new_black_clusters_output
            black_old2old_no_links += loc_no_black_clusters_input*loc_no_old_black_clusters_output
            white2black_no_links += (loc_no_clusters_input - loc_no_black_clusters_input)*(loc_no_black_clusters_output)
            white2white_no_links += (loc_no_clusters_input - loc_no_black_clusters_input)*(loc_no_clusters_output - loc_no_black_clusters_output) 
            no_links += loc_no_clusters_input*loc_no_clusters_output
            # trxs
            no_black_trxs += int(flag_input_black)
            no_white_trxs += int(not flag_input_black)

            # old and new nodes local update
            old_black_nodes.update(loc_old_black_nodes)
            new_black_nodes.update(loc_new_black_nodes)

        # block level
        # update black nodes and write them
        clust_is_black_ground_set.update(new_black_nodes)
        # clust_is_black_cum_set.update(old_black_nodes.union(new_black_nodes))
        no_old_black_clusters = len(old_black_nodes)
        no_new_black_clusters = len(new_black_nodes)
        no_black_clusters_cumulative += no_new_black_clusters

        # save in a new dictionary 
        _['block'] = b.height
        _['no_old_black_clusters'] = no_old_black_clusters
        _['no_new_black_clusters'] = no_new_black_clusters
        _['no_black_clusters_input'] = no_black_clusters_input 
        _['no_black_clusters_output'] = no_black_clusters_output
        _['no_old_black_clusters_output'] = no_old_black_clusters_output
        _['no_new_black_clusters_output'] = no_new_black_clusters_output
        _['no_black_clusters_cumulative'] = no_black_clusters_cumulative
        _['no_active_black_clusters'] = no_active_black_clusters
        _['no_white_clusters_input'] = no_white_clusters_input
        _['no_white_clusters_output'] = no_white_clusters_output
        _['no_active_white_clusters'] = no_active_white_clusters
        _['no_clusters_input'] = no_clusters_input
        _['no_clusters_output'] = no_clusters_output
        _['no_clusters_cumulative'] = no_clusters_cumulative
        _['no_active_clusters'] = no_active_clusters
        _['black2black_no_links'] = black2black_no_links
        _['black2white_no_links'] = black2white_no_links
        _['white2black_no_links'] = white2black_no_links
        _['white2white_no_links'] = white2white_no_links
        _['no_links'] = no_links
        _['vol_black_trxs'] = vol_black_trxs
        _['vol_white_trxs'] = vol_white_trxs
        _['no_black_trxs'] = no_black_trxs
        _['no_white_trxs'] = no_white_trxs
        _['total_trxs'] = total_trxs

        # save in csv
        csv_fout.writerow(_)

    chrono.print(message="took", tic="last")



   # addr_no_input_tx = zarr.load(f"{options.data_in_folder}/cluster_no_input_tx.zarr")
   # addr_no_output_tx = zarr.load(f"{options.data_in_folder}/cluster_no_output_tx.zarr")


