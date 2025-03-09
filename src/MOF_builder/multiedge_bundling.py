import numpy as np
from scipy.optimize import linear_sum_assignment
from v2_functions import fractional_to_cartesian


def find_pair_x_edge(x_matrix, edge_matrix):
    dist_matrix = np.zeros((len(x_matrix), len(edge_matrix)))
    for i in range(len(x_matrix)):
        for j in range(len(edge_matrix)):
            dist_matrix[i, j] = np.linalg.norm(x_matrix[i] - edge_matrix[j])
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return row_ind, col_ind


def find_pair_x_edge_fc(x_matrix, edge_matrix, sc_unit_cell):
    dist_matrix = np.zeros((len(x_matrix), len(edge_matrix)))
    x_matrix = fractional_to_cartesian(x_matrix, sc_unit_cell)
    edge_matrix = fractional_to_cartesian(edge_matrix, sc_unit_cell)
    for i in range(len(x_matrix)):
        for j in range(len(edge_matrix)):
            dist_matrix[i, j] = np.linalg.norm(x_matrix[i] - edge_matrix[j])
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    ##print(row_ind, col_ind) # debug
    return row_ind, col_ind


def bundle_multiedge(sG):
    multiedge_bundlings = []
    for n in sG.nodes:
        if "CV" in n and sG.nodes[n]["type"] == "V":
            edges = []
            for con_n in list(sG.neighbors(n)):
                edges.append(sG.edges[n, con_n]["coords"])
                # edges = np.vstack(edges, dtype=float)
                ## extract the X atom in the CV node
                # cv_xatoms = np.asarray(
                #    sG.nodes[n]["x_coords"][:, 2:5], dtype=float
                # )  # modified for extra column of atom type
                # if len(cv_xatoms) < len(edges):
                #    # duplicate the cv_xatoms
                #    cv_xatoms = np.vstack([cv_xatoms] * len(edges))
                ##_, order = find_pair_x_edge(cv_xatoms, edges)
                ##con_n_order = [list(sG.neighbors(n))[i] for i in order]
            multiedge_bundlings.append((n, list(sG.neighbors(n))))
            # make dist matrix of each x to each edge and then use hugerian algorithm to find the pair of x and edge
    return multiedge_bundlings
