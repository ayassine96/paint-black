#!/usr/bin/env python3

import networkx as nx
import blocksci
import zarr
import time
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict
import bz2
import matplotlib.pyplot as plt

import sys, os, os.path, socket


from util import SYMBOLS, DIR_PARSED, SimpleChrono

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
        print(f"Total #Addresses: {self.total_addresses}" )
   #     print(self.__counter_addresses)
        
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

    def load_clusters(self, input_folder):
        self.cluster = zarr.load(f"{input_folder}/address_cluster_map.zarr")
        print(f"Total #Clusters: {max(self.cluster)}" )



    def __getitem__(self,addr):
        return self.__offsets[addr.raw_type]+ addr.address_num-1
 

    
def get_all_files(path):
    os.chdir(path)
    
    list_of_files = list()
    
    for file in os.listdir():
        list_of_files.append(path+file)
            
    return list_of_files
            
    
def g_add_trx(inp,outp, value):

    if inp == outp: return

    if g.has_edge(inp,outp):
            g[inp][outp]['value'] = g[inp][outp]['value'] + value
            g[inp][outp]['n_tx']  = g[inp][outp]['n_tx']  + 1
    else:
            g.add_edge(inp,outp)
            g[inp][outp]['value'] = value
            g[inp][outp]['n_tx']  = 1
    
def build_current_dictionary(asset, idx):

    current_assets_zarr = zarr.load(asset)
    current_assets_index_zarr = zarr.load(idx)
    
    current_assets_dict = dict(zip(current_assets_index_zarr, current_assets_zarr))
        
    return current_assets_dict
        
        
def build_dark_dictionary(ratio, idx):

    dark_ratio_zarr = zarr.load(ratio)
    dark_ratio_index_zarr = zarr.load(idx)
    
    dark_ratio_dict = dict(zip(dark_ratio_index_zarr, dark_ratio_zarr))
    
    return dark_ratio_dict

def load_dictionaries(day):
    current_assets_path = "/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_final_daily_weekly/daily/current_assets/"
    
    current_index_path = "/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_final_daily_weekly/daily/current_assets_index/"
        
    dark_ratio_path = "/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_final_daily_weekly/daily/dark_ratio/"
    
    dark_index_path = "/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/heur_1_data_final_daily_weekly/daily/dark_ratio_index/"
  

    current_assets_file = f"{current_assets_path}current_assets_values_{day}.zarr"
    current_index_file = f"{current_index_path}current_assets_index_{day}.zarr"
    
    dark_ratio_file = f"{dark_ratio_path}dark_ratio_values_{day}.zarr"
    dark_index_file = f"{dark_index_path}dark_ratio_index_{day}.zarr"
    
    if os.path.exists(current_assets_file) and os.path.exists(current_index_file):
        current_assets_dict = build_current_dictionary(current_assets_file, current_index_file)
    
    if os.path.exists(dark_ratio_file) and os.path.exists(dark_index_file):
        dark_ratios_dict = build_dark_dictionary(dark_ratio_file, dark_index_file)
    
    return current_assets_dict,dark_ratios_dict

def g_add_node_attributes(node):

    #print(current_assets_dict)
    # print(node)
    # print(current_assets_dict[node])
    # print(dark_ratios_dict[node])

    # if g.has_node(node):
    #     # g[node]['current_assets'] = current_assets_dict[node]
    #     # g[node]['dark_ratio']  = dark_ratios_dict[node]
    #     attrs_n = [node, {'current_assets': current_assets_dict[node], 'dark_ratio': dark_ratios_dict[node]}]
    #     nx.set_node_attributes(g, attrs_n)
    #     G.node[node]['Attributes']
    
    if g.has_node(node):
        g.nodes[node]['current_assets'] = current_assets_dict[node]
        g.nodes[node]['dark_ratio'] = dark_ratios_dict[node]
    # g.add_node(node, current_assets = current_assets_dict[node], dark_ratio = dark_ratios_dict[node])
                    
                
                
def daterange(date1, date2, by=1):
    return [  date1 + timedelta(n) for n in range(0, int((date2 - date1).days)+1, by) ]         

if __name__ == "__main__":   
    options, args = parse_command_line()
    chrono      = SimpleChrono()
    chain       = blocksci.Blockchain("/local/scratch/exported/blockchain_parsed/bitcoin_old.cfg")

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
    
    daylist = daterange(start_date, end_date, by=options.frequency)
    tqdm_bar = tqdm(daylist, desc="processed files")

    for day in tqdm_bar:
#       chrono.add_tic("net0")
        dayrange = [day, day + timedelta(days=options.frequency)]
        try:
            dayblocks = chain.range(dayrange[0], dayrange[1])
        except:
            print(dayrange[0], dayrange[1], "cannot be processed")
            continue
            
        chrono.add_tic("net")
        g = nx.DiGraph()

        networks_daily_path = "/local/scratch/exported/blockchain_parsed/bitcoin/heur_1_networks_day"
        daily_graph_file = f"{networks_daily_path}/{day.strftime('%Y-%m-%d')}.graphml.bz2"
        
        g = nx.read_graphml(daily_graph_file)

        current_assets_dict_full, dark_ratios_dict_full = load_dictionaries(day)

        list_of_nodes = list(g.nodes)

        # print(f'list of nodes of size {len(list_of_nodes)}: {list_of_nodes}')
        
        current_assets_dict_filtered = { k: current_assets_dict_full[int(k)] for k in list_of_nodes }
        dark_ratios_dict_filtered = { k: dark_ratios_dict_full[int(k)] for k in list_of_nodes }
        # print(current_assets_dict_filtered)
        # print(dark_ratios_dict_filtered)    


        # print(f"adding graph attributes for {day.strftime('%Y-%m-%d')}")
        for node in g.nodes():
            
            nx.set_node_attributes(g, {node:current_assets_dict_filtered[node]}, 'current_assets')
            
            nx.set_node_attributes(g, {node:dark_ratios_dict_filtered[node]}, 'dark_ratio')
            
        
        # print(f'size of g.nodes :{len(g.nodes(data=True))}')
        # print(g.nodes(data=True))
        
        # nx.draw(g)
        # plt.show()
        tqdm_bar.set_description(f"'{day.strftime('%Y-%m-%d')} b ' took {chrono.elapsed('net')} sec", refresh=True)
        #nx.readwrite.graphml.write_graphml(g,fname)
        tqdm_bar.set_description(f"'{day.strftime('%Y-%m-%d')} bs' took {chrono.elapsed('net')} sec", refresh=True)
    print('Process terminated, graphs created.')
    print(f"Graphs created in {chrono.elapsed('proc', format='%H:%M:%S')}")

        
