#!/usr/bin/env python3

# input:
#     - `{options.black_data_folder}/cluster_is_black_ground_truth.zarr` ground truth clusters from `ub_ground_truth.py`
#     - `{DIR_PARSED}/{options.currency}/heur_{options.heuristic}_data/` clustering data
#     - `{DIR_PARSED}/{options.currency}.cfg` blockchain data
# outputs:
#     * zarr file: `` index is cluster id, value is int block when the cluster became black which can also represent time.

# here in this script we replicate the diffusion and from ground-truth we see how users turn black block by block

import blocksci
from decimal import Decimal, getcontext
import sys, os, os.path, socket
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import zarr
from tqdm import tqdm
from datetime import datetime, timedelta
from itertools import compress
from collections import defaultdict
import logging
import math
from util import SYMBOLS, DIR_BCHAIN, DIR_PARSED, SimpleChrono

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

def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n

if __name__ == "__main__":

    options, args = parse_command_line()

    logging.basicConfig(level=logging.DEBUG, filename=f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/logfile", filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")

    # Start Chrono
    chrono = SimpleChrono()

    # Load chain and initialize address mapper
    chain = blocksci.Blockchain(f"{DIR_PARSED}/{options.currency}_old.cfg")
    am = AddressMapper(chain)
    am.load_clusters(f"{options.cluster_data_folder}")

    # black_cluster: index-cluster, bool value-true if cluster is black. We use the same file we got from ub_ground_truth.py file
    clust_is_black_ground = zarr.load(f"{options.black_data_folder}/cluster_is_black_ground_truth.zarr") 

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
    weeksList = daterange(start_date, end_date, by=7)

    tqdm_bar = tqdm(weeksList, desc="processed files")

    # set of black users, transform clust_is_black_ground into a set where we consider only black clusters.
    clust_is_black_ground_set = set(compress(range(len(clust_is_black_ground)), clust_is_black_ground))

    if options.start_date != None:
        savedDataLocation = f"/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/final/heur_{options.heuristic}_data_v3/weekly/"

        current_assets_zarr = zarr.load(savedDataLocation + f'current_assets/current_assets_{start_date.strftime("%Y-%m-%d")}.zarr')
        current_assets = defaultdict(lambda: 0, dict(zip(current_assets_zarr["current_assets_index"], current_assets_zarr["current_assets_values"])))

        dark_assets_zarr = zarr.load(savedDataLocation + f'dark_assets/dark_assets_{start_date.strftime("%Y-%m-%d")}.zarr')
        dark_assets = defaultdict(lambda: 0, dict(zip(dark_assets_zarr["dark_assets_index"], dark_assets_zarr["dark_assets_values"])))

        dark_ratio_zarr = zarr.load(savedDataLocation + f'dark_ratio/dark_ratio_{start_date.strftime("%Y-%m-%d")}.zarr')
        dark_ratio = defaultdict(lambda: 0, dict(zip(dark_ratio_zarr["dark_ratio_index"], dark_ratio_zarr["dark_ratio_values"])))
    else:
        current_assets = defaultdict(lambda: 0)
        dark_assets = defaultdict(lambda: 0)
        dark_ratio = defaultdict(lambda: 0.0)


    chrono.print(message="init")
    print(f"[CALC] Starting the grayscale diffusion for all the blockchain...")

    # used to track blocks to ensure non are repeated. to restart the simulation find out from logs and set last seen block from previous week 
    currentBlock = 0 

    for week in tqdm_bar:
        chrono.add_tic("net")
        weekrange = [week, week + timedelta(days=7)]
        try:
            daysList = daterange(weekrange[0], weekrange[1], by=1)
        except:
            print(weekrange[0], weekrange[1], "cannot be processed")
            continue

        skip_last_day = 0
        # RUN ON Days range
        for day in daysList:
            if skip_last_day == 7: # ensure a 7 day week
                continue

            dayrange = [day, day + timedelta(days=1)]
            try:
                dayblocks = chain.range(dayrange[0], dayrange[1])
            except:
                print(dayrange[0], dayrange[1], "cannot be processed")
                continue

            for block in dayblocks:

                if block.height < currentBlock:
                    continue
                
                #______________________________TRX level_____________________________________

                for trx in block.txes:
                    # set of clusters who happeared in the current trx
                    trx_clusters = set()
                    #______________________________Initialize Variables_____________________________________
                    # 
                    clustered_inputs_dict = defaultdict(lambda: 0)
                    clustered_outputs_dict = defaultdict(lambda: 0)
                    
                    total_trx_input_value = 0
                    weight = defaultdict(dict)

                    # if coinbase generate save reward in assets
                    if trx.is_coinbase:

                        for out in trx.outputs:
                            cluster, value = am.cluster[am[out.address]], out.value
                            current_assets[cluster] = int(current_assets[cluster]) + int(value)
                            trx_clusters.add(cluster)
                    else:
                        # loop over trx inputs to build a reduced representation of inputs
                        for inp in trx.inputs:
                            cluster, value = am.cluster[am[inp.address]], inp.value
                            clustered_inputs_dict[cluster] += value
                            total_trx_input_value += value
                        
                        # loop over trx outputs to build a reduced representation of inputs
                        for out in trx.outputs:
                            cluster, value = am.cluster[am[out.address]], out.value
                            clustered_outputs_dict[cluster] += value

                        # loop trought all inputs and all outputs to find wij
                        for out_sender, sender_value in clustered_inputs_dict.items():
                        
                            if total_trx_input_value == 0:
                                continue

                            for out_receiver, receiver_value in clustered_outputs_dict.items():
                                # Calculate the weight of the edge and add the edge to the graph
                                weight[out_sender][out_receiver] = Decimal(sender_value/total_trx_input_value*receiver_value)

                    # once we computed all the weights, we can finally compute the new assets
                    for out_sender, sender_value in clustered_inputs_dict.items():
                        if total_trx_input_value == 0:
                            continue

                        trx_clusters.add(out_sender)
                        for out_receiver, receiver_value in clustered_outputs_dict.items():

                            current_assets[out_sender] = int(current_assets[out_sender]) - int(weight[out_sender][out_receiver])
                            current_assets[out_receiver] = int(current_assets[out_receiver]) + int(weight[out_sender][out_receiver])

                            # if current_assets[out_sender] < 0:
                            #     current_assets[out_sender] = 0

                            if dark_ratio[out_sender] > 0 and dark_ratio[out_sender] <= 1 :
                                dark_assets[out_sender] = clamp(int(dark_assets[out_sender]) - int(weight[out_sender][out_receiver]*Decimal(str(dark_ratio[out_sender]))), int(0), int(current_assets[out_sender]))
                                dark_assets[out_receiver] = clamp(int(dark_assets[out_receiver]) + int(weight[out_sender][out_receiver]*Decimal(str(dark_ratio[out_sender]))), int(0), int(current_assets[out_receiver]))
                            
                            trx_clusters.add(out_receiver)

                    # trx level, all transactions have been analysed
                    # update dark ratio of all clusters appeared in current trx
                    for cluster in trx_clusters:
                        if cluster in clust_is_black_ground_set:
                            dark_assets[cluster] = int(current_assets[cluster])
                            dark_ratio[cluster] = float(Decimal(1.0))
                        else:
                            if current_assets[cluster] > 0 and dark_assets[cluster] > 0:
                                dark_ratio[cluster] = float(Decimal(dark_assets[cluster]) / Decimal(current_assets[cluster]))
                            else:
                                dark_ratio[cluster] = float(Decimal(0.0))


                    # Unusual Values monitoring
                    with open(f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/unusual_values.txt", "a") as f:
                        if dark_ratio[cluster] < 0 or dark_ratio[cluster] > 1 or math.isnan(dark_ratio[cluster]) or math.isinf(dark_ratio[cluster]):
                            print(f'error value of dark_ratio at week={week}, day={day}, block={block.height}, cluster={cluster}, value={dark_ratio[cluster]}', file=f)
                        
                        if current_assets[cluster] < 0 or  math.isnan(current_assets[cluster]) or math.isinf(current_assets[cluster]):
                            print(f'error value of current_assets at week={week}, day={day}, block={block.height}, cluster={cluster}, value={current_assets[cluster]}', file=f)
                        
                        if dark_assets[cluster] < 0 or  math.isnan(dark_assets[cluster]) or math.isinf(dark_assets[cluster]):
                            print(f'error value of dark_assets at week={week}, day={day}, block={block.height}, cluster={cluster}, value={dark_assets[cluster]}', file=f)
                            

                # Block level progress monitoring monitoring
                with open(f"logfiles/daily_weekly_final_heur_{options.heuristic}_v3/block_progress.txt", "a") as f:
                    print(f'finished block={block.height} where currentBlock={currentBlock} , day:{day} , week:{week} , time:{datetime.now()}', file=f)
                
                currentBlock += 1

            # Initialize and save per day
            current_assets_index, current_assets_values = zip(*current_assets.items())
            dark_assets_index, dark_assets_values = zip(*dark_assets.items())
            dark_ratio_index, dark_ratio_values = zip(*dark_ratio.items())

            savelocation = f"/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/final/heur_{options.heuristic}_data_v3/daily/"
            zarr.save(savelocation + f'current_assets/current_assets_{day.strftime("%Y-%m-%d")}.zarr', current_assets_values=current_assets_values, current_assets_index=current_assets_index)
            zarr.save(savelocation + f'dark_assets/dark_assets_{day.strftime("%Y-%m-%d")}.zarr', dark_assets_values=dark_assets_values, dark_assets_index=dark_assets_index)
            zarr.save(savelocation + f'dark_ratio/dark_ratio_{day.strftime("%Y-%m-%d")}.zarr', dark_ratio_values=dark_ratio_values, dark_ratio_index=dark_ratio_index)
            logging.info(f'results day:{day} with last block of the day being b={currentBlock - 1}')
            skip_last_day += 1

        # Initialize and save per week
        current_assets_index, current_assets_values = zip(*current_assets.items())
        dark_assets_index, dark_assets_values = zip(*dark_assets.items())
        dark_ratio_index, dark_ratio_values = zip(*dark_ratio.items())

        savelocation = f"/local/scratch/exported/blockchain_parsed/bitcoin_darknet/gs_group/grayscale_op_ali/final/heur_{options.heuristic}_data_v3/weekly/"
        zarr.save(savelocation + f'current_assets/current_assets_{week.strftime("%Y-%m-%d")}.zarr', current_assets_values=current_assets_values, current_assets_index=current_assets_index)
        zarr.save(savelocation + f'dark_ratio/dark_ratio_{week.strftime("%Y-%m-%d")}.zarr', dark_ratio_values=dark_ratio_values, dark_ratio_index=dark_ratio_index)
        zarr.save(savelocation + f'dark_assets/dark_assets_{week.strftime("%Y-%m-%d")}.zarr', dark_assets_values=dark_assets_values, dark_assets_index=dark_assets_index)
        logging.info(f'results week:{week} with last block of the week being b={currentBlock - 1}')

        tqdm_bar.set_description(f"week of '{week.strftime('%Y-%m-%d')} took {chrono.elapsed('net')} sec", refresh=True)
    

    chrono.print(message="took", tic="last")


