"""
    Author: Lasse Regin Nielsen
"""

from __future__ import print_function
import os, csv
import numpy as np
#import matplotlib.pyplot as plt
#from decimal import Decimal

from global_vars import ss_opt
from global_vars import rough_min
from global_vars import rough_max
from global_vars import span
from global_vars import bin_count
from global_vars import db_size
from global_vars import WDT
from math import ceil

filepath = os.path.dirname(os.path.abspath(__file__))

def read_data(filename, has_header=True):
    """
        Read data from file.
        Will also return header if header=True
    """
    data, header = [], None
    with open(filename, 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        if has_header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def getWDT_index(minw ,span, x):
    x = float("{0:.4f}".format(x))
    x = x-minw
    k = ceil(x/span)   
    if k < 0:
        k = 0
    k = k - 1
    return int(k)


def getMaxWeightFromWDTindex(indx):
    return rough_min + ((indx+1)*span)
    

def load_graphs(filename, minw, span, db_size, bin_count):
    """
        Loads graphs from file
    """
##    weights = []
    edgecnt = 0
    data, _ = read_data(filename, has_header=False)
    graphs = []
    maxW = 0
    gc = -1
    global lgid
    for line in data:
        if line[0] == 't':
            lgid=int(line[2])
            gc += 1
            G = Graph(lgid)
            graphs.append(G)
            edgecnt = 0;
        else:
            if line[0] == 'v':
                v = Vertex(id=int(line[1]), label=line[2])
                graphs[len(graphs)-1].add_vertex(vertex=v)
            elif line[0] == 'e':
                edgecnt = edgecnt +  1
                weight = float(line[4].strip('"'))
##                weights.append(weight)
                if weight > maxW:
                    maxW = weight
                row = getWDT_index(minw, span, weight)
                #print(row)
                #print(lgid)
                #print(WDT)
                WDT[row][gc] = WDT[row][gc] + 1
                e = Edge(label=line[3],
                         from_vertex=graphs[len(graphs)-1].get_vertex(id=int(line[1])),
                         to_vertex=graphs[len(graphs)-1].get_vertex(id=int(line[2])), edge_weight = weight)
                if(edgecnt < 10):
                    graphs[len(graphs)-1].add_edge(edge=e)
                #graphs[len(graphs)-1].add_edge(edge=e)

##    n, x, _ = plt.hist(weights, bins=120, histtype=u'step')
##    bin_centers = 0.5*(x[1:]+x[:-1])
##    plt.plot(bin_centers,n)
##    plt.ylabel('Frequency')
##    plt.show()
    return graphs, maxW

#################################################
#                    Classes                    #
#################################################
class Queue(object):
    """
        Implementation of a simple queue data structure
    """
    def __init__(self, queue=None):
        if queue is None:
            self.queue = []
        else:
            self.queue = list(queue)
    def dequeue(self):
        return self.queue.pop(0)
    def enqueue(self, element):
        self.queue.append(element)
    def is_empty(self):
        return len(self.queue) == 0
    def empty(self):
        self.queue = []

class Vertex():
    """
        Implementation of an Vertex in a graph
    """
    visited = False
    dfs_id = 0
    def __init__(self, id, label):
        self.id = id
        self.label = label

class Edge():
    """
        Implementation of an Edge in a graph(mod: added edge_weight)
    """
    def __init__(self, label, from_vertex, to_vertex, edge_weight):
        self.label = label
        self.from_vertex = from_vertex
        self.to_vertex = to_vertex
        self.edge_weight = edge_weight

    def connected_to(self, vertex):
        return vertex.id == self.from_vertex.id or \
               vertex.id == self.to_vertex.id

class Graph():
    """
        Implementation of a Graph
    """
    edges, vertices = [], []
    def __init__(self, id):
        self.id = id
        self.edges = []
        self.vertices = []
    def add_vertex(self, vertex):
        self.vertices.append(vertex)
    def add_edge(self, edge):
        self.edges.append(edge)
    def get_vertex(self, id):
        for v in self.vertices:
            if v.id == id:
                return v
        raise KeyError('No vertex with the id was found in graph')
    def adjacent_edges(self, vertex):
        adj_edges = []
        for e in self.edges:
            if e.connected_to(vertex):
                adj_edges.append(e)
        return adj_edges
    def adjacent_vertices(self, vertex):
        adj_edges = self.adjacent_edges(vertex)
        adj_vertices = []
        for e in adj_edges:
            if e.from_vertex.id == vertex.id:
                adj_vertices.append(e.to_vertex)
            else:
                adj_vertices.append(e.from_vertex)
        return adj_vertices
    def adjacent_connections(self, vertex):
        adj_edges = self.adjacent_edges(vertex)
        adj_connections = []
        for e in adj_edges:
            if e.from_vertex.id == vertex.id:
                adj_connections.append((e, e.to_vertex))
            else:
                adj_connections.append((e, e.from_vertex))
        # Sort according to node index
        ids = [w.id for e,w in adj_connections]
        idx = np.argsort(ids)
        adj_connections = [adj_connections[i] for i in idx]
        return adj_connections
    def generate_vertices(self):
        for e in self.edges:
            for v in [e.from_vertex, e.to_vertex]:
                v.id = v.dfs_id
                if not v in self.vertices:
                    self.add_vertex(vertex=v)
    def get_max_vertex(self):
        ids = [v.id for v in self.vertices]
        idx = np.argsort(ids)[::-1]
        return self.vertices[idx[0]]
    def get_max_dfs_id_vertex(self):
        vertices_id = []
        for i, v in enumerate(self.vertices):
            if not v.dfs_id is None:
                vertices_id.append(i)
        if len(vertices_id) > 0:
            ids = [self.vertices[i].id for i in vertices_id]
            idx = np.argsort(ids)[::-1]
            return self.vertices[idx[0]]
        else:
            return []
    def get_min_vertex(self):
        ids = [v.id for v in self.vertices]
        idx = np.argsort(ids)
        return self.vertices[idx[0]]
    def contains_vertex_id(self, id):
        for v in self.vertices:
            if v.id == id:
                return True
        return False
    def contains_edge(self, from_id, to_id):
        for e in self.edges:
            if (e.from_vertex.id == from_id and e.to_vertex.id == to_id) or \
               (e.to_vertex.id == from_id and e.from_vertex.id == to_id):
               return True
        return False
    def reverse_graph(self):
        for e in self.edges:
            tmp_from = e.from_vertex
            e.from_vertex = e.to_vertex
            e.to_vertex = tmp_from
        self.edges = self.edges[::-1]
        self.vertices = self.vertices[::-1]
    def print_graph(self):
        DFScode = G2DFS(self)
        for line in DFScode:
            print(line)
    def get_edge(self, from_id, to_id):
        for e in self.edges:
            if (e.from_vertex.id == from_id and e.to_vertex.id == to_id) or \
               (e.to_vertex.id == from_id and e.from_vertex.id == to_id):
               return e
        return None
    def reset(self):
        for v in self.vertices:
            v.visited = False
            v.dfs_id = None

#################################################
#                   Functions                   #
#################################################
def DFS(G, v):
    """
        Depth-first search recursive algorithm:
        Input:
            G   Graph object containing vertices and edges
            v   Root vertex of the graph G (Vertex object)
        Output:
            p   Graph making a DFS spanning tree
    """
    G.reset() # Reset search parameters
    edges = []
    recursive_call_DFS(G, v, edges)
    p = Graph(-1)
    for e in edges:
        p.add_edge(e)
    p.generate_vertices()
    return p

def recursive_call_DFS(G, v, edges):
    """
        Helper function for recursive DFS
    """
    v.visited = True
    v.dfs_id = len(edges)
    neighbors = G.adjacent_connections(vertex=v)
    for e, w in G.adjacent_connections(vertex=v):
        if not w.visited:
            edges.append(e)
            recursive_call_DFS(G, w, edges)

def rightmost_path_BFS(G, v, v_target):
    """
        Get rightmost path using Breadth-First search algorithm on DFS path:
        Input:
            G           Graph object containing vertices and edges
            v           Root vertex of the graph G (Vertex object)
            v_target    Target vertex
        Output:
            p           Graph of shortest path from v to v_target
    """
    G.reset() # Reset search parameters
    for _v in G.vertices:
        _v.dfs_id = float('inf')
        _v.parent = None
    Q = Queue()
    v.dfs_id = 0
    Q.enqueue(v)
    while not Q.is_empty():
        current = Q.dequeue()
        for e, w in G.adjacent_connections(vertex=current):
            if w.dfs_id == float('inf'):
                w.dfs_id = current.dfs_id + 1
                w.parent = current
                Q.enqueue(w)
                if(w == v_target):
                    Q.empty()
                    break
    tmp = v_target
    p = Graph(id=-1)
    while tmp.parent is not None:
        e = Edge(label='_', from_vertex=tmp, to_vertex=tmp.parent, edge_weight=-1)
        p.add_edge(edge=e)
        p.add_vertex(vertex=tmp)
        tmp = tmp.parent
    p.add_vertex(vertex=tmp)
    return p

def get_rightmost_path(G):
    """
        Returns the rightmost-path of the graph G
    """
    v_root = G.get_min_vertex()
    v_target = G.get_max_vertex()
    T_G = DFS(G=G, v=v_root)
    v_target = G.get_max_dfs_id_vertex()
    R = rightmost_path_BFS(T_G, v_root, v_target)
    #for v in R.vertices:
    #    v.id = v.dfs_id
    R.reverse_graph()
    return R

def G2DFS(G):
    """
        Converts a graph object into a DFScode tuple sequence
    """
    DFScode = []
    for e in G.edges:
        DFScode.append((e.from_vertex.id, e.to_vertex.id,
            e.from_vertex.label, e.to_vertex.label, e.label, e.edge_weight))
    return DFScode

def DFS2G(C):
    """
        Converts a DFScode tuple sequence C into a graph G
    """
    G = Graph(id=-1)
    vertices = []
    for u,v,L_u,L_v,L_uv, edge_weight in C:
        for vertex, label in [(u, L_u), (v, L_v)]:
            if not (vertex, label) in vertices:
                vertices.append((vertex, label))
    for v_id, v_label in vertices:
        # Create and add vertex
        v = Vertex(id=v_id, label=v_label)
        G.add_vertex(vertex=v)
    # Add edges
    for t in C:
        # Expand tuple
        u, v, L_u, L_v, L_uv, edge_weight = t
        # Get vertices
        _u, _v = G.get_vertex(id=u), G.get_vertex(id=v)
        # Add edge
        e = Edge(label=L_uv, from_vertex=_u, to_vertex=_v, edge_weight = edge_weight)
        G.add_edge(edge=e)
    return G

def tuple_is_smaller(t1,t2):
    """
        Checks whether the tuple t1 is smaller than t2
    """
    t1_forward = t1[1] > t1[0]
    t2_forward = t2[1] > t2[0]
    i,j,x,y = t1[0], t1[1], t2[0], t2[1]
    # Edge comparison
    if t1_forward and t2_forward:
        if j < y or (j == y and i > x):
            return True
        elif j > y or (j == y and i < x):
            return False
    elif (not t1_forward) and (not t2_forward):
        if i < x or (i == x and j < y):
            return True
        elif i > x or (i == x and j > y):
            return False
    elif t1_forward and (not t2_forward):
        if j <= x:
            return True
        else:
            return False
    elif (not t1_forward) and t2_forward:
        if i < y:
            return True
        elif i > y: # Maybe something missing here
            return False
    # Lexicographic order comparison
    a1,b1,c1 = str(t1[2]), str(t1[3]), str(t1[4])
    a2,b2,c2 = str(t2[2]), str(t2[3]), str(t2[4])

##    if not a1.isdigit():
##        a1,b1,c1 = ord(a1),ord(b1),ord(c1)
##        a2,b2,c2 = ord(a2),ord(b2),ord(c2)
##    else:
##        a1,b1,c1 = int(a1),int(b1),int(c1)
##        a2,b2,c2 = int(a2),int(b2),int(c2)

    if a1 < a2:
        return True
    elif a1 == a2:
        if b1 < b2:
            return True
        elif b1 == b2:
            if c1 < c2:
                return True
    return False
        #raise KeyError('Wrong key type in tuple')

#def compare_DFScodes
def tuples_are_smaller(G1, G2):
    """
        Checks if tuples in G1 are less than tuples in G2
    """
    #print("tuples_are_smaller")
    #print(G1)
    #print(G2)
    DFScodes_1, DFScodes_2 = G1, G2
    if len(DFScodes_1) != len(DFScodes_2):
        raise Exception('Size of the two graphs are not equal')
    for i in range(0, len(DFScodes_1)):
        t1, t2 = DFScodes_1[i], DFScodes_2[i]
        is_smaller = tuple_is_smaller(t1,t2)
        if is_smaller:
            return True
    return False

def get_minimum_DFS(G_list):
    """
        Finds the graph with smallest DFS code i.e. the canonical graph
    """
    # Initialize first one as minimum
    min_G = G_list[0]
    min_idx = 0
    counts = np.zeros(len(G_list))
    for i in range(0, len(G_list)):
        for j in range(0, len(G_list)):
            #print(i)
            #print(j)
            if i == j:
                continue
            is_smaller = tuples_are_smaller(G_list[i], G_list[j])
            if not is_smaller:
                counts[i] += 1
    #print(counts)
    min_idx = np.argmin(counts)
    min_G = G_list[min_idx]
    return min_G, min_idx


def weighted_subgraph_isomorphisms(C, G):
    """
        Returns the set of all isomorphisms between C and G
    """
    # Initialize set of isomorphisms by mapping vertex 0 in C
    # to each vertex x in G that shares the same label as 0s
    #G.print_graph()
    phi_c = []
    phi_c_weight = []
    G_C = DFS2G(C)
    v0 = G_C.get_min_vertex()
    for v in G.vertices:
        if v.label == v0.label:
            phi_c.append([(v0.id, v.id)])
            phi_c_weight.append([])

    #first Edge
    #print(phi_c)
    for i, t in enumerate(C):
        u, v, L_u, L_v, L_uv, edge_weight = t       # Expand extended edge
        phi_c_prime = []                            # partial isomorphisms
        phi_c_weight_prime = []
        phi_c_weight_leni = 0
        for phi in phi_c:
            phi_weight = phi_c_weight[phi_c_weight_leni]
            phi_c_weight_leni += 1
            # phi is a list of transformations
            if v > u:
                # Forward edge
                try:
                    phi_u = transform_vertex(u, phi)
                except Exception as e:
                    continue
                # Find neighbors of transformed vertex
                vertex = G.get_vertex(phi_u)
                neighbors = G.adjacent_connections(vertex)
                for e, x in neighbors:
                    # Check if an inverse transformation exists
                    inv_trans_exists = check_inv_exists(x.id, phi)
                    if (not inv_trans_exists) and \
                        (x.label == L_v) and \
                        (e.label == L_uv):
                        phi_prime = list(phi)
                        phi_weight_prime = list(phi_weight)
                        phi_prime.append((v, x.id))
                        phi_weight_prime.append(e.edge_weight)
                        phi_c_prime.append(list(phi_prime))
                        phi_c_weight_prime.append(list(phi_weight_prime))
            else:
                # Backward edge
                try:
                    phi_u = transform_vertex(u, phi)
                    phi_v = transform_vertex(v, phi)
                except Exception as e:
                    continue
                # Find neighbors of transformed vertex
                vertex = G.get_vertex(phi_u)
                neighbors = G.adjacent_connections(vertex)
                for e, x in neighbors:
                    if phi_v == x.id:
                        phi_c_prime.append(list(phi))
                        phi_weight_prime = list(phi_weight)
                        phi_weight_prime.append(e.edge_weight)
                        phi_c_weight_prime.append(list(phi_weight_prime))
                        break
        phi_c = list(phi_c_prime)
        phi_c_weight = list(phi_c_weight_prime)
    return phi_c, phi_c_weight



def check_inv_exists(v, phi):
    """
        Given a vertex id u and a set of partial isomorphisms phi.
        Returns True if an inverse transformation exists for v
    """
    for _phi in phi:
        if _phi[1] == v:
            return True
    return False

def inv_transform_vertex(x, phi):
    """
        Given a vertex id x and a set of partial isomorphisms phi.
        Returns the inverse transformed vertex id
    """
    for _phi in phi:
        if _phi[1] == x:
            return _phi[0]
    raise Exception('Could not find inverse transformation')

def transform_vertex(u, phi):
    """
        Given a vertex id u and a set of partial isomorphisms phi.
        Returns the transformed vertex id
    """
    for _phi in phi:
        if _phi[0] == u:
            return _phi[1]
    raise Exception('u couldn\' be found in the isomorphisms')

def tuple_is_equal(t1, t2, ignoreWeight):
    x, y, lx, ly, lxy, wxy = t1
    i, j, li, lj, lij, wij = t2
    if x == i and y == j and lx == li and ly == lj and lxy == lij:
        if ignoreWeight == True or wxy == wij:
            return True
        else:
            return False
    else:
        return False

def List_is_equal(t1, t2, ignoreWeight):
    if t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2] and t1[3] == t2[3] and t1[4] == t2[4]:
        if ignoreWeight == True or t1[5] == t2[5]:
            return True
        else:
            return False
    else:
        return False


def getWeightFromPHI(G, C, phi):
    wgt_sum = 0
    rngs = np.zeros((bin_count,), dtype=int)
    for t in C:
        u, v, lu, lx, lxy, w = t
        ri = getWDT_index(rough_min ,span, w)
        u = transform_vertex(u, phi)
        v = transform_vertex(v, phi)
        edg = G.get_edge(u, v)
        ri = getWDT_index(rough_min ,span, edg.edge_weight)
        wgt_sum += edg.edge_weight
        rngs[ri] += 1
    return wgt_sum, rngs


def getWeightFromPHIweight(G, C, phi_weight):
    wgt_sum = 0
    rngs = np.zeros((bin_count,), dtype=int)
    for item in phi_weight:
        ri = getWDT_index(rough_min ,span, item)
        wgt_sum += item
        rngs[ri] += 1
    return wgt_sum, rngs



def getMaxWeight(indx, C, G, WeightSum, Phi_rngs):
    length = len(C)+1
    curAvgW = WeightSum/length
    for i in range(bin_count-1, -1, -1):
        #print(WDT[i][indx] - Phi_rngs[i])
        RemEdge = WDT[i][indx] - Phi_rngs[i] #Remaining Edge of ith range
        rngW = getMaxWeightFromWDTindex(i)
        if rngW > curAvgW:
            WeightSum = WeightSum + rngW*RemEdge
            length += RemEdge
            curAvgW = WeightSum/length
        else:
            break
    return curAvgW
        




def Len0RMPE(G, indx):
    # option 1
    temp = [] # extensions of one edge from Gi
    
    # add distinct label tuples in G_i as forward extensions
    for e in G.edges:
        L_x, L_y, L_xy, edge_weight = e.from_vertex.label, e.to_vertex.label, e.label, e.edge_weight
        
        ##option 1
        WDT_indx = getWDT_index(rough_min ,span, edge_weight)
        rngs = list(np.zeros((bin_count,), dtype=int))
        rngs[WDT_indx] += 1
        MaxWeight = getMaxWeight(indx, [], G, edge_weight, rngs)
        #print(WDT_indx)
        f = [0, 1, L_x, L_y, L_xy, edge_weight, 1, MaxWeight]
        #print(f)
        f1 = [0, 1, L_y, L_x, L_xy, edge_weight, 1, MaxWeight]
        #print(f1)

        flag = 0
        tempi = 0
        while tempi<len(temp):
             if List_is_equal(f, temp[tempi], True):
                 temp[tempi][5] = (temp[tempi][5] * temp[tempi][6] + f[5]) / (temp[tempi][6] + 1)
                 temp[tempi][6] = temp[tempi][6] + 1
                 if temp[tempi][7] < f[7]:
                     temp[tempi][7] = f[7]
                 flag = 1
                 break
             tempi += 1

        if flag == 0:
            temp.append(f)
            

        flag = 0
        tempi = 0
        while tempi<len(temp):
             if List_is_equal(f1, temp[tempi], True):
                 temp[tempi][5] = (temp[tempi][5] * temp[tempi][6] + f1[5]) / (temp[tempi][6] + 1)
                 temp[tempi][6] = temp[tempi][6] + 1
                 if temp[tempi][7] < f[7]:
                     temp[tempi][7] = f[7]
                 flag = 1
                 break
             tempi += 1

        if flag == 0:
            temp.append(f1)
    return temp


def getWeightFromPhi_rngs(phi_rngs):
    wgt_sum = 0
    for val in phi_rngs:
        wgt_sum += getMaxWeightFromWDTindex(val)
    return wgt_sum
    
    


def getSmallerPhi_rngs(phi_rngs1, phi_rngs2):
    prW1 = getWeightFromPhi_rngs(phi_rngs1)
    prW2 = getWeightFromPhi_rngs(phi_rngs2)
    if prW2 > prW1:
        return phi_rngs1
    else:
        return phi_rngs2





def PhiRMPE(indx, C, G, phi, phi_weight, weightSum, phi_rngs, temp, G_C, R, u_r, L_u_r):
    ############################################
    # Backward extensions from rightmost child #
    ############################################
    phi_u_r = transform_vertex(u_r, phi)
    # Find neighbors of transformed vertex
    vertex = G.get_vertex(phi_u_r)
    neighbors = G.adjacent_connections(vertex)
    for e, x in neighbors:
        if check_inv_exists(x.id, phi):
            v = inv_transform_vertex(x.id, phi)
            if R.contains_vertex_id(id=v) and \
               not G_C.contains_edge(from_id=u_r, to_id=v):
                _e = G.get_edge(transform_vertex(v, phi), phi_u_r)
                if _e is None:
                    raise Exception('Couldn\'t find edge')
                L_v = G_C.get_vertex(id=v).label
                WDT_indx = getWDT_index(rough_min ,span, _e.edge_weight)

                #To test
                copyOfPhi_rngs = list(phi_rngs)
                copyOfPhi_rngs[WDT_indx] += 1
                newWeightSum = _e.edge_weight + weightSum
                #print("Max Weight Called")
                MaxWeight = getMaxWeight(indx, C, G, newWeightSum, copyOfPhi_rngs)
                #f = [u_r, v, L_u_r, L_v, _e.label, newWeightSum, 1, copyOfPhi_rngs]
                f = [u_r, v, L_u_r, L_v, _e.label, newWeightSum, 1, MaxWeight]
                
                #f = [u_r, v, L_u_r, L_v, _e.label, (_e.edge_weight + weightSum), 1, WDT_indx]
                flag = 0
                tempi = 0
                while tempi<len(temp):
                     if List_is_equal(f, temp[tempi], True):
                         temp[tempi][5] = temp[tempi][5] + f[5]     #adds weight of all extended isomorphic patterns's weight
                         temp[tempi][6] = temp[tempi][6] + 1        #increases isomorphism count in same graph
                         #temp[tempi][7] = getSmallerPhi_rngs(f[7], temp[tempi][7])
                         if temp[tempi][7] < f[7]:
                             temp[tempi][7] = f[7]
                         flag = 1
                         break
                     tempi += 1

                if flag == 0:
                    temp.append(f)
    ###################################################
    # Forward extensions from nodes on rightmost path #
    ###################################################
    for u in R.vertices:
        phi_u = transform_vertex(u.id, phi)
        # Find neighbors of transformed vertex
        vertex = G.get_vertex(phi_u)
        neighbors = G.adjacent_connections(vertex)
        for e, x in neighbors:
            if not check_inv_exists(x.id, phi):
                WDT_indx = getWDT_index(rough_min ,span, e.edge_weight)
                copyOfPhi_rngs = list(phi_rngs)
                copyOfPhi_rngs[WDT_indx] += 1

                newWeightSum = e.edge_weight + weightSum
                #print("Max Weight Called")
                MaxWeight = getMaxWeight(indx, C, G, newWeightSum, copyOfPhi_rngs)
                
                #f = [u.id, u_r + 1, vertex.label, x.label, e.label, (e.edge_weight+weightSum), 1, copyOfPhi_rngs]
                f = [u.id, u_r + 1, vertex.label, x.label, e.label, newWeightSum , 1, MaxWeight]
                flag = 0
                tempi = 0
                while tempi<len(temp):
                     if List_is_equal(f, temp[tempi], True):
                         temp[tempi][5] = (temp[tempi][5] + f[5])
                         temp[tempi][6] = temp[tempi][6] + 1
                         #temp[tempi][7] = getSmallerPhi_rngs(f[7], temp[tempi][7])
                         if temp[tempi][7] < f[7]:
                             temp[tempi][7] = f[7]
                         flag = 1
                         break
                     tempi += 1

                if flag == 0:
                    temp.append(f)
    #print(temp)
    return temp


def RMPE(C, D, oc):
    """
        Implements the RightMostPath-Extensions algorithm.
        Given a frequent canonical DFS code C and a list of graphs D, a
        set of possible rightmost path extensions from C, along with
        their support values are computed.
    """
    # Create graph of C -> G(C)
    G_C = DFS2G(C=C)
    # Only if C is not empty
    if len(C) > 0:
        # Compute rightmost path
        R = get_rightmost_path(G_C)
        u_r = R.vertices[len(R.vertices)-1].id
        L_u_r = R.vertices[len(R.vertices)-1].label
    E = [] # set of extensions from C
    for i in oc:
        G = D[i]
    #for i, G in enumerate(D):
        if len(C) == 0: # If C is empty
            temp = Len0RMPE(G, i)
            tempie = 0
            while tempie < len(temp):
                E.append((i, temp[tempie][7], [temp[tempie][0], temp[tempie][1], temp[tempie][2], temp[tempie][3], temp[tempie][4], temp[tempie][5]]))
                tempie += 1           
        else:
            # Get subgraph isomorphisms
            phi_c_i, phi_c_weight_i = weighted_subgraph_isomorphisms(C, G)
            temp = []
            ec = len(C)
            phi_c_weight_i_leni = 0
            for phi in phi_c_i:
                phi_weight = phi_c_weight_i[phi_c_weight_i_leni]
                phi_c_weight_i_leni += 1
                weightSum, phi_rngs = getWeightFromPHIweight(G, C, phi_weight)
                temp = PhiRMPE(i, C, G, phi, phi_weight, weightSum, phi_rngs, temp, G_C, R, u_r, L_u_r)
                

            tempie = 0
            while tempie < len(temp):
                weight = (temp[tempie][5] / temp[tempie][6])/(ec+1)
                E.append((i, temp[tempie][7] , [temp[tempie][0], temp[tempie][1], temp[tempie][2], temp[tempie][3], temp[tempie][4], weight]))
                tempie += 1


    #print(E)
    extensions = []
    for item in E:
        gi, MaxWeight, te = item
        exti = 0
        flag = 0
        while exti < len(extensions):
            if extensions[exti][0] == te[0] and extensions[exti][1] == te[1] and extensions[exti][2] == te[2] and extensions[exti][3] == te[3] and extensions[exti][4] == te[4]:
                extensions[exti][5] = (extensions[exti][5] * len(extensions[exti][6]) + te[5]) /  (len(extensions[exti][6]) + 1)
                extensions[exti][6].append(gi)
                extensions[exti][7] += MaxWeight                
                flag = 1
                break
            exti += 1
        if flag == 0:
            oc_item = []
            oc_item.append(gi)
            ext_item = [te[0], te[1], te[2], te[3], te[4], te[5], oc_item, MaxWeight]
            extensions.append(ext_item)
    return extensions




def sort_tuples(E):
    """
        Sort a list of tuples using the get_minimum_DFS function.
    """
    sorted_tuples = []
    tuples = [[t] for t in E]
    for i in range(0, len(tuples)):
        min_G, min_idx = get_minimum_DFS(tuples)
        sorted_tuples.append(tuples[min_idx][0])
        del tuples[min_idx]
    return sorted_tuples

def compute_support(C, D):
    """
        Computes the support of subgraph C in set of graphs D
    """
    sup = 0
    for i, G in enumerate(D):
        phi_c_i = subgraph_isomorphisms(C, D[i])
        if len(phi_c_i) > 0:
            sup += 1
    return sup

def is_canonical(C):
    """
        Checks if C is canonical
    """
    D_C = [DFS2G(C)]    # graph corresponding to code C
    C_star = []         # initialize canonical DFScode
    k = len(C)
    for i in range(0, k):
        E = RMPE(C_star, D_C, [0])
        if len(E) == 0:
            break
        #print(E)
        #G_list = [/[_e[0]] for _e in E/]
        G_list = []
        for ext in E:
            t = (ext[0], ext[1], ext[2], ext[3], ext[4], ext[5])
            c_t = []
            c_t.append(t)
            G_list.append(c_t)
        #print(G_list)
        min_G, min_idx = get_minimum_DFS(G_list)
        tmp = E[min_idx]
        s_i = (tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])
        sup_s_i = len(tmp[6])
        if tuple_is_smaller(s_i, C[i]):
            #print("**************************************EndC******************")
            return False
        C_star.extend([s_i])
    return True # no smaller code exists -> C is canonical

def g_span(C, D, min_sup, extensions, maxW, fwsCount, canCount, gCount, oc):
    """
        Finds possible frequent and canonical extensions of C in D, using
        min_sup as lowest allowed support value.
        Results are stored in extensions
    """
    E = RMPE(C, D, oc)
    for ext in E:
        t = (ext[0], ext[1], ext[2], ext[3], ext[4], ext[5])
        weight_t = ext[5]
        oc_t = ext[6]
        MaxPWS = ext[7]
        sup_t = len(oc_t)
        C_prime = list(C)
        C_prime.extend([t])
        gCount = gCount + 1
        sup_C_prime = sup_t
        wsup = weight_t * sup_t
  
        if (wsup >= min_sup) and is_canonical(C_prime):
            extensions.append(C_prime)
            canCount = canCount + 1
            fwsCount = fwsCount + 1
            fwsCount, canCount, gCount = g_span(C_prime, D, min_sup, extensions, maxW, fwsCount, canCount, gCount, oc_t)
            
        elif (MaxPWS >= min_sup) and is_canonical(C_prime):
            canCount = canCount + 1
            fwsCount, canCount, gCount = g_span(C_prime, D, min_sup, extensions, maxW, fwsCount, canCount, gCount, oc_t)
    return fwsCount, canCount, gCount

