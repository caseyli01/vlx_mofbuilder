import numpy as np
import networkx as nx



def sort_nodes_by_type_connectivity(G):
    Vnodes = [n for n in G.nodes() if G.nodes[n]['type']=='V']
    DVnodes = [n for n in G.nodes() if G.nodes[n]['type']=='DV']
    Vnodes = sorted(Vnodes,key=lambda x: G.degree(x),reverse=True)
    DVnodes = sorted(DVnodes,key=lambda x: G.degree(x),reverse=True)
    return Vnodes+DVnodes

def check_edge_inunitcell(G,e):
    if 'DV' in G.nodes[e[0]]['type'] or 'DV' in G.nodes[e[1]]['type']:
        return False
    return True

def find_and_sort_edges_bynodeconnectivity(graph, sorted_nodes):
    all_edges = list(graph.edges())

    sorted_edges = []
    #add unit_cell edge first

    ei = 0
    while ei < len(all_edges):
            e = all_edges[ei]
            if check_edge_inunitcell(graph,e):
                sorted_edges.append(e)
                all_edges.pop(ei)
            ei += 1
    #sort edge by sorted_nodes
    for n in sorted_nodes:
        ei = 0
        while ei < len(all_edges):
            e = all_edges[ei]
            if n in e:
                if n ==e[0]:
                    sorted_edges.append(e)
                else:
                    sorted_edges.append((e[1],e[0]))
                all_edges.pop(ei)
            else:
                ei += 1

    return sorted_edges   