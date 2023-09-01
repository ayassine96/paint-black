#!/bin/bash

echo "heur 2 Plotting Json & Figures"
python3 gs_plotting_weekly_final.py --curr=BTC --heur=2 --freq=week --end=2018-05-12 #--start=2016-07-16
echo "heur 2 Network Builder"
python3 network_builder.py --curr=BTC --heur=2 --freq=week --end=2018-05-12 #--start=2016-07-16
echo "heur 2 Assortativity"
python3 assortativity_builder.py --curr=BTC --heur=2 --freq=week --end=2018-05-12 #--start=2016-07-16