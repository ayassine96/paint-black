#!/usr/bin/env python3
"""
Note: we only have to run this once for each currency/heuristic pair, then they are saved.
input: these files are already on the server
- `{DIR_PARSED/bitcoin_darknet/ground_truth_id.csv}` # ground truth dataframe with addresses connected to criminal activities
- `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` # clustering data, think of them as a map from many addressess to a user for all users.
- `{DIR_PARSED}/{options.currency}.cfg` # blockchain data in a different format for blocksci  
output:
in `{options.output_folder}/heur_{options.heuristic}_data/` # inside this it will create two files.
* `cluster_is_black_ground_truth.zarr`:
	where index is cluster number, value is bool: True if black (originally) ground truth, False otherwise
* `ground_truth_clust_id.csv`:
	dataframe to relate entities, btc addresses and cluster ids
"""

import blocksci

import sys, os, os.path, socket
import numpy as np
import zarr
from tqdm import tqdm
import pandas as pd
from util import SYMBOLS, DIR_BCHAIN, DIR_PARSED, SimpleChrono, darknet



def parse_command_line(): #this is a function to pass options to the python script: currency, heuristics (4 different), etc.
    import sys, optparse

    parser = optparse.OptionParser()

    parser.add_option("--curr", action='store', dest="currency", type='str',
                                              default=None, help="name of the currency")
    parser.add_option("--heur", action='store', dest="heuristic", type='str',
                                                  default=None, help="heuristics to apply")
    parser.add_option("--overwrite", action='store_true', dest = "overwrite" )
    parser.add_option("--output", action='store', dest = "output_folder", 
                        default=None, type='str', help='directory to save outputs in')

    options, args = parser.parse_args()
    options.currency = SYMBOLS[options.currency] # resolved in util.py
    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}/"
    options.cluster_data_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/"

    print(f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}/")
    print(f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/")

    if options.output_folder is None:
        options.output_folder = options.cluster_data_folder
    else:
        options.output_folder = f"{options.output_folder}/heur_{options.heuristic}_data/"
        if not os.path.exists(options.output_folder):
            os.makedirs(options.output_folder)

    return options, args


# this class creates a map from addresses to the set {0 -> number of cluster}, in other words it helps to use arrays to store the information. 
class AddressMapper(): # This creates the map that clusters the addresses to actual users. Treat it a a blackbox for now.
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
        print(f"{input_folder}address_cluster_map.zarr")
        self.cluster = zarr.load(f"{input_folder}address_cluster_map.zarr")
    def __getitem__(self,addr):
        return self.__offsets[addr.raw_type]+ addr.address_num-1

def catch(address, am):
    try:
        return am[address] 
    except:
        return np.nan

if __name__ == "__main__":
    options, args = parse_command_line() # parse the options
    chrono = SimpleChrono() # measure time

    df = pd.read_csv(f"{DIR_PARSED}/bitcoin_darknet/ground_truth_id.csv") # TODO: add column names ?
    #pd.set_option("display.max_rows", None, "display.max_columns", None) TODO: remove
    #print(df)

    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}.cfg") # load the blockchain
    print(f"{DIR_PARSED}/{options.currency}.cfg")
    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}") #this data should already be on the server.

    no_clusters = max( am.cluster ) + 1 # print the number of clusters
    print(f"[INFO] #clusters: {no_clusters}")

    chrono.print(message="init")

    print("[CALC] who_is_black ... ") #
    clust_is_black = np.zeros(no_clusters, dtype=bool)
    print(clust_is_black.len)

    df = df.loc[df['entity'].isin(darknet)]  # drop not darknet entities from ground truth
    df["address_id"] = [catch(chain.address_from_string(a), am) for a in df.address]  # find address id
    df = df.dropna(subset=["address_id"])  # drop na address
    df = df.astype({'address_id' : 'int64'})  # address_id column must be int - type conversion
    df['cluster_id'] = am.cluster[df.address_id]  # use address mapper to add cluster id to entities address or cluster id

    c_indices = am.cluster[df.address_id]  # find black cluster indices
    clust_is_black[c_indices] = True  # taint black clusters in array format

    # test print
    print(c_indices)
    print(clust_is_black[c_indices[1:10]])
    print(clust_is_black.len)
    # save files
    df.to_csv(f"{options.output_folder}/ground_truth_clust_id.csv")
    zarr.save(f"{options.output_folder}/cluster_is_black_ground_truth.zarr", clust_is_black)
    del clust_is_black 

    chrono.print(message="took", tic="last")


