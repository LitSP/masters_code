import networkx as nx
import copy
from collections import defaultdict
import numpy as np
import pandas as pd

def read_trusted_ases(G, filename='./trusted.txt'):
    with open(filename) as file:
        for line in file:
            node, size, weight = line.rstrip().split()
            # print(node, size, weight)
            G.add_node(node, size=eval(size), weight=eval(weight))
            G.add_edge('trusted', node)

            
def read_semi_trusted_ases(G, filename='./semi-trusted.txt'):
    with open(filename) as file:
        for line in file:
            node, size, weight = line.rstrip().split()
            G.add_node(node, size=eval(size), weight=eval(weight))
            G.add_edge('semi-trusted', node)

            
def read_untrusted_ases(G, filename='./untrusted.txt'):
    with open(filename) as file:
        for line in file:
            node, size, weight = line.rstrip().split()
            G.add_node(node, size=eval(size), weight=eval(weight))
            G.add_edge('untrusted', node)

            
def read_links(G, filename='./links.txt'):
    with open(filename) as file:
        for line in file:
            
            link = line.rstrip()
            
            if len(link) > 4:
                
                link = eval(link)
                G.add_edge(str(link[0]), str(link[1]))


def simrank(G, C=0.9, max_iter=100, weights_flag=False):

    # init vars
    
    sim_old = defaultdict(list)
    sim = defaultdict(list)
    weights = dict(G.nodes(data='weight'))
    
    for n in G.nodes():
    
        sim[n] = defaultdict(int)
        sim[n][n] = 1
        sim_old[n] = defaultdict(int)
        sim_old[n][n] = 0

    # recursive calculation of SimRank
    
    for iter in range(max_iter):
    
        if _has_converged(sim, sim_old):
            break
        
        sim_old = copy.deepcopy(sim)
        
        for u in G.nodes():
            for v in G.nodes():
                
                if u == v:
                    continue
                
                sim_uv = 0.0
                
                for n_u in G.neighbors(u):
                    for n_v in G.neighbors(v):
                        if not weights_flag:
                            sim_uv += sim_old[n_u][n_v]
                        else:
                            sim_uv += weights[n_u] * weights[n_v] * sim_old[n_u][n_v]
                
                sim[u][v] = (C * sim_uv / (len(list(G.neighbors(u))) * len(list(G.neighbors(v)))))
    
    return sim


def _has_converged(s1, s2, eps=1e-4):
    
    for i in s1.keys():
        for j in s1[i].keys():
            if abs(s1[i][j] - s2[i][j]) >= eps:
                return False
    
    return True


def get_simrank(f_trusted='./trusted.txt',
                f_semi_trusted='./semi-trusted.txt',
                f_untrusted='./untrusted.txt',
                f_links='./links.txt',
                C=0.9,
                max_iter=100,
                weights_flag=False):
    
    G = nx.Graph()

    G.add_node("trusted")
    G.add_node("untrusted")
    G.add_node("semi-trusted")

    read_trusted_ases(G, f_trusted)
    read_semi_trusted_ases(G, f_semi_trusted)
    read_untrusted_ases(G, f_untrusted)

    read_links(G, f_links)
    
    sizes = np.array([i[1] for i in G.nodes(data="size")])
    sizes[pd.isna(sizes)] = int(sizes[pd.notna(sizes)].mean())
    
    weights = np.array([i[1] for i in G.nodes(data="weight")])
    weights[pd.isna(weights)] = int(weights[pd.notna(weights)].max())
    weights *= (1 / np.linalg.norm(weights, np.inf))
    weights = weights ** (1/2)

    for j, i in enumerate(G.nodes()):
        G.nodes[i]["size"]   = sizes[j]
        G.nodes[i]["weight"] = round(weights[j], 3)

    return G, simrank(G, C, max_iter, weights_flag=weights_flag)