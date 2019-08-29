from __future__ import print_function
import os, sys
import numpy as np
import time


from algorithms import g_span as gSpan
from algorithms import load_graphs
from algorithms import read_data
from algorithms import weighted_subgraph_isomorphisms

from global_vars import rough_min
from global_vars import rough_max
from global_vars import span
from global_vars import bin_count
from global_vars import db_size
from global_vars import WDT
from global_vars import filepath
from global_vars import fname
from global_vars import msup



def main(filename=fname, min_sup=msup):
    start_time = time.time()
    filename = os.path.join(filepath, filename)
    graphs, maxW = load_graphs(filename, rough_min, span, db_size, bin_count)
    for wdt in WDT:
        print("*************")
        print(wdt)


    n = len(graphs)
    extensions = []


    canCount = 0
    gCount = 0
    fwsCount = 0
    oc = range(len(graphs))
    fwsCount, canCount, gCount = gSpan([], graphs, min_sup=min_sup, extensions=extensions, maxW=maxW, fwsCount = fwsCount, canCount = canCount, gCount = gCount, oc=oc)
    end_time = time.time()
    obj = open("output.txt", "w+")
    for i, ext in enumerate(extensions):
        obj.write('Pattern %d\n' % (i+1))
        for _c in ext:
            obj.write(str(_c))
            obj.write('\n')
        obj.write('')

    obj.write("--- %s seconds ---\n" % (end_time - start_time))
    obj.write(str(fwsCount))
    obj.write('\n')
    obj.write(str(canCount))
    obj.write('\n')
    obj.write(str(gCount))
    obj.write('\n')
    obj.close()
    
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("")
        print("Finds possible frequent and canonical extensions of C in D, using")
        print("min_sup as lowest allowed support value.")
        print("Usage: %s FILENAME minsup" % (sys.argv[0]))
        print("")
        print("FILENAME: Relative path of graph data file.")
        print("minsup:   Minimum support value.")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['filename'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['min_sup'] = sys.argv[2]
        if len(sys.argv) > 3:
            sys.exit("Not correct arguments provided. Use %s -h for more information" % (sys.argv[0]))
        main(**kwargs)


