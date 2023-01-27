#!/usr/bin/env python3


import time
import smtplib
import email
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict

import sys, os, os.path, socket

darknet = [
    'AbraxasMarket',
    'AlphaBayMarket',
    'BabylonMarket',
    'BlackBankMarket',
    'BlueSkyMarketplace',
    'CannabisRoadMarket',
    'EvolutionMarket',
    'GreenRoadMarket',
    'MiddleEarthMarketplace',
    'NucleusMarket',
    'PandoraOpenMarket',
    'SheepMarketplace',
    'SilkRoad2Market',
    'SilkRoadMarketplace'
        ]

SYMBOLS = {
            "BTC": "bitcoin",
            "LTC": "litecoin",
            "DOGE": "dogecoin",
            "BCH": "bitcoin_cash",
            "BSV": "bitcoin_sv",
            "LCH": "litecoin_cash",
            "FTC": "feathercoin",
            "MONA": "monacoin",
            "PPC": "peercoin",
            "NMC": "namecoin"
          }

uzh_colors = {
            "uzh_darkblue": (5,85,158),
            "uzh_blue": (0,39,163),
            "uzh_lightblue": (167,180,217),
            "uzh_grey": (163,173,183),
            "uzh_lightgrey": (212,219,224),
            "uzh_darkgrey": (151,165,175),
            "uzh_red": (220,96,39),
            "uzh_darkred": (216,88,52),
            "uzh_lightred": (240,193,166),
            "uzh_green": (136,215,49),
            "uzh_darkgreen": (123,171,65),
            "uzh_darkyellow": (254,222,0),
            "uzh_lightgreen": (201,231,161)
            }

if socket.gethostname() == 'abacus':
    DIR_BCHAIN="/mnt/hdd_data/blockchain_data/"
    DIR_PARSED="/mnt/hdd_data/blockchain_parsed/"
elif socket.gethostname() == 'abacus-1':
    DIR_BCHAIN="/mnt/hdd_data/blockchain_data/"
    DIR_PARSED="/mnt/hdd_data/blockchain_parsed/"
elif socket.gethostname() == 'consensus-2':
    DIR_BCHAIN="/local/scratch/exported/blockchain_parsed"
    DIR_PARSED="/export/consensus-2/blockchain_parsed" # updated location


def sendEmail(username,sender,reciever,pasw,subject, msg):
    msg = email.message_from_string(msg)
    msg['From'] = sender
    msg['To'] = reciever
    msg['Subject'] = subject
    try:
        s = smtplib.SMTP("smtp.uzh.ch",587)
        s.starttls() #Puts connection to SMTP server in TLS mode
        s.login(username, pasw)
        s.sendmail(msg['From'], msg['To'], msg.as_string())
        print('Email sent')
    except Exception as e:
        print('Email not sent for the following error: ',e)
    finally:
        s.quit()    
    
class SimpleChrono:
    def __init__(self):
        n=time.perf_counter()
        self.tics = {0: n, "last":n}


    def add_tic(self, key):
        self.tics[key] = time.perf_counter()
        self.tics['last'] = self.tics[key]

    def elapsed(self, tic=0, format=None):
        n = time.perf_counter()
        if format == None:
            return "{:.3f}".format(n - self.tics[tic])
        else:
            return time.strftime(format, time.gmtime(n - self.tics[tic]))
        self.tics['last'] = n

    def print(self, tic=0, message=None, format=None):
        s = self.elapsed(tic, format)
        print(f"[CHRONO] {message}: {s}")
