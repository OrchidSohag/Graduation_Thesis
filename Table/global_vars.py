import os, sys
#from decimal import Decimal
#This file contains global variables
global rough_min
global ss_opt,  rough_max, bin_count, span, db_size
#graph_wise weight
ss_opt = 1


#Sample Dataset
##rough_min = float(0.1)
##rough_max = float(0.9)
##bin_count = 4
##span = (rough_max-rough_min)/bin_count
##db_size = 4
##fname='data/smpl.txt'
##msup=0.7




#cgnorm Dataset
##rough_min = float(10)
##rough_max = float(140)
##bin_count = 4
##span = (rough_max-rough_min)/bin_count
##db_size = 422


#goodware Dataset
rough_min = float(0)
rough_max = float(4300)
bin_count = 5
span = (rough_max-rough_min)/bin_count
db_size = 10
fname='malware_graphs_10.txt'
msup=1


WDT = [[0 for x in range(db_size)] for y in range(bin_count)]

filepath = os.path.dirname(os.path.abspath(__file__))

