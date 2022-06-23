#!/usr/bin/env python3
"""
input:
- `{DIR_PARSED/bitcoin_darknet/ground_truth_id.csv}` ground trught dataframe
- `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
- `{DIR_PARSED}/{options.currency}.cfg` blockchain data
output:
in `{options.output_folder}/heur_{options.heuristic}_data/`
* `cluster_is_black_ground_truth.zarr`:
	index is cluster, value is bool: True if black ground truth, False otherwise
* `ground_truth_clust_id.csv`:
	dataframe to relate entities, btc addresses and cluster ids
"""


import blocksci

import sys, os, os.path, socket
import numpy as np
import zarr
import time
from tqdm import tqdm
import pandas as pd

from util import SYMBOLS, DIR_BCHAIN, DIR_PARSED, SimpleChrono, darknet



def parse_command_line():
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

    options.currency = SYMBOLS[options.currency]


    options.cluster_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}/"

    options.cluster_data_folder = f"{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/"

    if options.output_folder is None:
        options.output_folder = options.cluster_data_folder
    else:
        options.output_folder = f"{options.output_folder}/heur_{options.heuristic}_data/"
        if not os.path.exists(options.output_folder):
            os.makedirs(options.output_folder)


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
#        address_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types }
        cluster_vector = {_: np.zeros(self.__counter_addresses[_], dtype=np.int64) for _ in self.__address_types }

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



    def __getitem__(self,addr):
        return self.__offsets[addr.raw_type]+ addr.address_num-1

def catch(address, am):
    try:
        return am[address] 
    except:
        return np.nan

if __name__ == "__main__":
    options, args = parse_command_line()

    # heur_file   = zarr.open_array(f"{options.cluster_folder}_data/address_cluster_map.zarr", mode = 'r')
    # memstore    = zarr.MemoryStore()
    # zarr.copy_store(heur_file.store, memstore)
    # heur_mem    = zarr.open_array(memstore)
    chrono = SimpleChrono()

    df = pd.read_csv(f"{DIR_PARSED}/bitcoin_darknet/ground_truth_id.csv")
    display(df)

    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}.cfg")
    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}")

    no_clusters = max( am.cluster ) + 1
    print(f"[INFO] #clusters: {no_clusters}")


    chrono.print(message="init")


    print("[CALC] who_is_black ... ")
    clust_is_black = np.zeros(no_clusters, dtype=bool)

    df = df.loc[df['entity'].isin(darknet)]  # drop not darknet entities
    df["address_id"] = [catch(chain.address_from_string(a), am) for a in df.address]  # find address id
    df = df.dropna(subset=["address_id"])  # drop na address
    df = df.astype({'address_id' : 'int64'})  # address_id column must be int
    df['cluster_id'] = am.cluster[df.address_id]  # add cluster id to entities address

    c_indices = am.cluster[df.address_id]  # find black cluster indices
    clust_is_black[c_indices] = True  # taint black clusters

    df.to_csv(f"{options.output_folder}/ground_truth_clust_id.csv")
    zarr.save(f"{options.output_folder}/cluster_is_black_ground_truth.zarr", clust_is_black)
    del clust_is_black 

    chrono.print(message="took", tic="last")



   # addr_no_input_tx = zarr.load(f"{options.data_in_folder}/cluster_no_input_tx.zarr")
   # addr_no_output_tx = zarr.load(f"{options.data_in_folder}/cluster_no_output_tx.zarr")







#    chrono.print(message="load...")


