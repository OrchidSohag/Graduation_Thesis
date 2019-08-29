from __future__ import print_function
import os, sys
import numpy as np
import re

from algorithms import load_graphs
from algorithms import read_data
from algorithms import subgraph_isomorphisms
filepath = os.path.dirname(os.path.abspath(__file__))



def main(filename='gd10.txt', min_sup=8):
    filename = os.path.join(filepath, filename)
    graphs = load_graphs(filename)

    C = list()
    with open("Malware_sub_Goodware.txt", "r") as ins:
        content = ins.read().splitlines()
        temp = list()
        for line in content:
            if line !='Pattern':
                line = re.sub(r'[\(\)]', '', line)
                u, v, L_u, L_v, L_uv =  line.split(", ")
                u, v, L_u, L_v, L_uv = int(u), int(v),L_u.strip("'"), L_v.strip("'"), L_uv.strip("'")
                #print("u= {} v= {} L_u= {} L_v= {} L_uv= {}\n".format(u,v,L_u,L_v,L_uv))
                temp.append((u, v, L_u, L_v, L_uv))
            elif line == 'Pattern':
                C.append(temp)
                temp =[]
    # for num, li in enumerate(C):
    #     for t in li:
    #         print("Element {}: {}\n".format(num,t[2]))

    for g, graph in enumerate(graphs):
        for p, li in enumerate(C):
            E = subgraph_isomorphisms(li,graph)
            if len(E) == 0:
                print ("Graph: {} , Pattern: {} Not Matched".format(g+1,p+1))
            else:
                print ("Graph: {} , Pattern: {} Matched".format(g+1,p+1))


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