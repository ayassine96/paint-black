#!/bin/bash

echo "heur 1"
python3 ub_diffusion_block.py --curr=BTC --heur=1 --output=uniform_black
echo "heur 1p"
python3 ub_diffusion_block.py --curr=BTC --heur=1p --output=uniform_black
echo "heur 2"
python3 ub_diffusion_block.py --curr=BTC --heur=2 --output=uniform_black
echo "heur 2p"
python3 ub_diffusion_block.py --curr=BTC --heur=2p --output=uniform_black
