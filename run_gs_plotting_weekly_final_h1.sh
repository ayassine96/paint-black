#!/bin/bash

# echo "heur 1 Plotting Json & Figures"
# python3 gs_plotting_weekly_final.py --curr=BTC --heur=1 --freq=week --end=2018-05-12 #--start=2016-07-16
# echo "heur 1 Network & assortativity Builder"
# python3 network_builder.py --curr=BTC --heur=1 --freq=week --end=2018-05-12 #--start=2009-01-10
# echo "heur 1 Assortativity"
# python3 assortativity_builder.py --curr=BTC --heur=1 --freq=week --end=2018-05-12 #--start=2009-01-10
# echo "heur 1 Random Network Builder" 
# python3 random_network_builder.py --curr=BTC --heur=1 --freq=week --end=2018-05-12 --start=2009-01-10
echo "heur 1 Community Builder" 
python3 community_builder.py --curr=BTC --heur=1 --freq=week --end=2018-05-12 --start=2016-07-30
