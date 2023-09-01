#!/bin/bash

# echo "heur 1p Plotting Json & Figures"
# python3 gs_plotting_weekly_final.py --curr=BTC --heur=1p --freq=week --end=2018-05-12 #--start=2016-07-16
echo "heur 1p Network Builder & assortativity Builder"
python3 network_builder.py --curr=BTC --heur=1p --freq=week --end=2018-05-12 --start=2009-01-10
# echo "heur 1p Assortativity"
# python3 assortativity_builder.py --curr=BTC --heur=1p --freq=week --end=2018-05-12 #--start=2016-07-16