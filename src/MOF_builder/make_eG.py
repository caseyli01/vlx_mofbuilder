import networkx as nx
import numpy as np
from v2_functions import fetch_X_atoms_ind_array
from multiedge_bundling import find_pair_x_edge_fc
from makesuperG import pname, locate_min_idx
import re


def nn(s):
    return re.sub(r"\d+", "", s)


def nl(s):
    return re.sub(r"\D+", "", s)


def order_edge_array(row_ind, col_ind, edges_array):
    # according col_ind order  to reorder the connected edge points
    old_split = np.vsplit(edges_array, len(col_ind))
    old_order = []
    for i in range(len(col_ind)):
        old_order.append(
            (row_ind[i], col_ind[i], old_split[sorted(col_ind).index(col_ind[i])])
        )
    new_order = sorted(old_order, key=lambda x: x[0])
    ordered_arr = np.vstack([new_order[j][2] for j in range(len(new_order))])
    return ordered_arr


def make_paired_Xto_x(ec_arr, merged_arr, neighbor_number, sc_unit_cell):
    """
    ec_arr: the array of the linker center node
    merged_arr: the array of the merged edges
    neighbor_number: the number of the neighbor nodes of the linker center node
    """
    ec_indices, ec_fpoints = fetch_X_atoms_ind_array(ec_arr, 0, "X")
    if len(ec_indices) < neighbor_number:
        # duplicate the cv_xatoms
        ec_fpoints = np.vstack([ec_fpoints] * neighbor_number)
    nei_indices, nei_fcpoints = fetch_X_atoms_ind_array(
        merged_arr, 0, "X"
    )  # extract only the X atoms in the neighbor edges but not the cv nodes: [len(ec_arr) :]
    row_ind, col_ind = find_pair_x_edge_fc(
        ec_fpoints[:, 2:5].astype(float),
        nei_fcpoints[:, 2:5].astype(float),
        sc_unit_cell,
    )  # NOTE: modified to skip atom type

    # according col_ind order  to reorder the connected edge points
    # switch the X to x for nei_fcpoints
    for i in col_ind:
        if nn(merged_arr[nei_indices[i], 0]) == "X":
            merged_arr[nei_indices[i], 0] = "x" + nl(merged_arr[nei_indices[i], 0])
    for k in row_ind:
        if ec_indices[k] < len(ec_arr):
            if nn(ec_arr[ec_indices[k], 0]) == "X":
                ec_arr[ec_indices[k], 0] = "x" + nl(ec_arr[ec_indices[k], 0])

    ordered_edges_points_follow_ecXatoms = order_edge_array(
        row_ind, col_ind, merged_arr
    )
    # remove the duplicated cv_xatoms
    ec_merged_arr = np.vstack((ec_arr, ordered_edges_points_follow_ecXatoms))
    return ec_merged_arr


def superG_to_eG_multitopic(superG, sc_unit_cell):
    # CV + neighbor V -> E+index node
    eG = nx.Graph()
    edge_count = 0
    node_count = 0
    for n in superG.nodes():
        if superG.nodes[n]["note"] == "V":
            node_count += 1
            eG.add_node(
                n,
                f_points=superG.nodes[n]["f_points"],
                fcoords=superG.nodes[n]["fcoords"],
                type="V",
                name="NODE",
                note="V",
                index=node_count,
            )
            superG.nodes[n]["index"] = node_count
            # add virtual edge
            for e in superG.edges(n):
                if superG.edges[e]["type"] == "virtual":
                    eG.add_edge(e[0], e[1], type="virtual")

        elif superG.nodes[n]["note"] == "CV":
            edge_count -= 1
            neighbors = list(superG.neighbors(n))
            merged_edges = superG.nodes[n]["f_points"]
            # reorder neighbors according to the order of X atoms in CV node

            for ne in neighbors:
                merged_edges = np.vstack(
                    [superG.edges[n, ne]["f_points"] for ne in neighbors]
                )

                # follow the order of X atoms in CV node

            # for x atoms in merged_edges, use hungarian algorithm to find the nearest X-X atoms in the neighbor nodes and replace the X to x
            ec_merged_edges = make_paired_Xto_x(
                superG.nodes[n]["f_points"], merged_edges, len(neighbors), sc_unit_cell
            )
            eG.add_node(
                "EDGE_" + str(edge_count),
                f_points=ec_merged_edges,
                fcoords=superG.nodes[n]["fcoords"],
                type="Edge",
                name="EDGE",
                note="E",
                index=edge_count,
            )

            for ne in neighbors:
                eG.add_edge(
                    "EDGE_" + str(edge_count),
                    ne,
                    index="E_" + str(edge_count),
                    type="real",
                )

    return eG, superG


def superG_to_eG_ditopic(superG):
    # V + neighbor V -> E+index node
    eG = nx.Graph()
    edge_count = 0
    node_count = 0
    edge_record = []
    for n in superG.nodes():
        if superG.nodes[n]["note"] == "V":
            node_count += 1
            eG.add_node(
                n,
                f_points=superG.nodes[n]["f_points"],
                fcoords=superG.nodes[n]["fcoords"],
                type=superG.nodes[n]["type"],
                note=superG.nodes[n]["note"],
                name="NODE",
                index=node_count,
            )
            # print('add node',n,'type',superG.nodes[n]['type']) #debug
            superG.nodes[n]["index"] = node_count
            # add virtual edge
            for e in superG.edges(n):
                if superG.edges[e]["type"] == "virtual":
                    eG.add_edge(e[0], e[1], type="virtual")

            neighbors = list(superG.neighbors(n))
            for ne in neighbors:
                if sorted([n, ne]) in edge_record:
                    continue
                edge_record.append(sorted([n, ne]))
                edge_count -= 1
                eG.add_node(
                    "EDGE_" + str(edge_count),
                    f_points=superG.edges[n, ne]["f_points"],
                    fcoords=superG.edges[n, ne]["fcoords"],
                    type="Edge",
                    name="EDGE",
                    note="E",
                    index=edge_count,
                )
                eG.add_edge(n, ne, index="E_" + str(edge_count), type="real")
                eG.add_edge(
                    "EDGE_" + str(edge_count),
                    ne,
                    index="E_" + str(edge_count),
                    type="half",
                )
                eG.add_edge(
                    "EDGE_" + str(edge_count),
                    n,
                    index="E_" + str(edge_count),
                    type="half",
                )

    return eG, superG


def find_nearest_neighbor(i, n_n_distance_matrix):
    n_n_min_distance = np.min(n_n_distance_matrix[i : i + 1, :])
    _, n_j = locate_min_idx(n_n_distance_matrix[i : i + 1, :])
    # print('add virtual edge between',nodes_list[i],nodes_list[n_j])
    # n_n_distance_matrix[i, n_j] = 1000
    # set the column to 1000 to avoid the same atom being selected again
    n_n_distance_matrix[:, n_j] = 1000
    return n_j, n_n_min_distance, n_n_distance_matrix


def find_surrounding_points(ind, n_n_distance_matrix, max_number):
    stop = 0  # if while loop is too long, stop it
    nearest_neighbor = {}
    nearest_neighbor[ind] = []
    while len(nearest_neighbor[ind]) < max_number:
        stop += 1
        if stop > 100:
            break
        n_j, _, n_n_distance_matrix = find_nearest_neighbor(ind, n_n_distance_matrix)
        nearest_neighbor[ind].append(n_j)
    return nearest_neighbor


# Function to find 'XOO' pairs for a specific node
def xoo_pair_ind_node(single_node_fc, sc_unit_cell):
    # if the node x is not surrounded by two o atoms,
    # then modify the fetch_X_atoms_ind_array(single_node, 0, 'O') find_surrounding_points(k, xs_os_dist_matrix, 2)
    # this function is to find the XOO pairs in a specific node(by node_id),
    # this xoo pair is the indice of x and nearest two o atoms in the same node
    # return the indice of x and nearest two o atoms in the same node, which can be convert to a dict with x_index as key and o_indices as value
    # the distance is in cartesian coordinates
    # single_node_fc: coordinates of any node in the main fragment
    # sc_unit_cell: supercell unit cell matrix
    single_node = np.hstack(
        (
            single_node_fc[:, 0:1],
            np.dot(sc_unit_cell, single_node_fc[:, 2:5].astype(float).T).T,
        )
    )  # NOTE: modified to skip atom type
    xind, xs_coords = fetch_X_atoms_ind_array(single_node, 0, "X")
    oind, os_coords = fetch_X_atoms_ind_array(single_node, 0, "O")
    xs_os_dist_matrix = np.zeros((len(xs_coords), len(os_coords)))
    for i in range(len(xs_coords)):
        for j in range(len(os_coords)):
            xs_os_dist_matrix[i, j] = np.linalg.norm(
                xs_coords[i, 1:4].astype(float) - os_coords[j, 1:4].astype(float)
            )
    xoo_ind_list = []
    for k in range(len(xind)):
        nearest_dict = find_surrounding_points(k, xs_os_dist_matrix, 2)
        for key in nearest_dict.keys():
            xoo_ind_list.append(
                [xind[key], sorted([oind[m] for m in nearest_dict[key]])]
            )
    return xoo_ind_list


def get_xoo_dict_of_node(eG, sc_unit_cell):
    # quick check the order of xoo in every node are same, select n0 and n1, if xoo_ind_node0 == xoo_ind_node1, then xoo_dict is the same
    # return xoo dict of every node, key is x index, value is o index
    n0 = [i for i in eG.nodes() if pname(i) != "EDGE"][0]
    n1 = [i for i in eG.nodes() if pname(i) != "EDGE"][1]
    xoo_ind_node0 = xoo_pair_ind_node(
        eG.nodes[n0]["f_points"], sc_unit_cell
    )  # pick node one and get xoo_ind pair
    xoo_ind_node1 = xoo_pair_ind_node(
        eG.nodes[n1]["f_points"], sc_unit_cell
    )  # pick node two and get xoo_ind pair
    if xoo_ind_node0 == xoo_ind_node1:
        xoo_dict = {}
        for xoo in xoo_ind_node0:
            xoo_dict[xoo[0]] = xoo[1]
    else:
        print("the order of xoo in every node are not same, please check the input")
        print("xoo_ind_node0", xoo_ind_node0)
        print("xoo_ind_node1", xoo_ind_node1)
    return xoo_dict


def remove_node_by_index(eG, remove_node_list, remove_edge_list):
    for n in eG.nodes():
        if pname(n) != "EDGE":
            if eG.nodes[n]["index"] in remove_node_list:
                eG.remove_node(n)
        if pname(n) == "EDGE":
            if -1 * eG.nodes[n]["index"] in remove_edge_list:
                eG.remove_node(n)
    return eG


def addxoo2edge_multitopic(eG, sc_unit_cell):
    xoo_dict = get_xoo_dict_of_node(eG, sc_unit_cell)
    matched_vnode_X = []
    unsaturated_linker = []
    # for every X atom in the EDGE node, search for the paired(nearest) X atom in the connected V node
    # and then use the xoo_dict of the connected V node to extract the xoos of the connected V node
    # and then add the xoos to the EDGE node
    # all xoo_node for the V node is the same
    EDGE_nodes = [n for n in eG.nodes() if pname(n) == "EDGE"]
    for n in EDGE_nodes:
        Xs_edge_indices, Xs_edge_fpoints = fetch_X_atoms_ind_array(
            eG.nodes[n]["f_points"], 0, "X"
        )
        Xs_edge_ccpoints = np.hstack(
            (
                Xs_edge_fpoints[:, 0:2],
                np.dot(sc_unit_cell, Xs_edge_fpoints[:, 2:5].astype(float).T).T,
            )
        )  # NOTE: modified to skip atom type
        V_nodes = [i for i in eG.neighbors(n) if pname(i) != "EDGE"]
        if len(V_nodes) == 0:
            # unsaturated_linker.append(n)
            # print(
            #    "no V node connected to this edge node, this linker is a isolated linker, will be ignored",
            #    n,
            # ) # debug
            continue
        all_Xs_vnodes_ind = []
        all_Xs_vnodes_ccpoints = np.zeros((0, 5))
        for v in V_nodes:
            # find the connected V node and its X atoms
            Xs_vnode_indices, Xs_vnode_fpoints = fetch_X_atoms_ind_array(
                eG.nodes[v]["f_points"], 0, "X"
            )
            Xs_vnode_ccpoints = np.hstack(
                (
                    Xs_vnode_fpoints[:, 0:2],
                    np.dot(sc_unit_cell, Xs_vnode_fpoints[:, 2:5].astype(float).T).T,
                )
            )  # NOTE: modified to skip atom type
            for ind in Xs_vnode_indices:
                all_Xs_vnodes_ind.append([v, ind, n])
            all_Xs_vnodes_ccpoints = np.vstack(
                (all_Xs_vnodes_ccpoints, Xs_vnode_ccpoints)
            )
        edgeX_vnodeX_dist_matrix = np.zeros(
            (len(Xs_edge_ccpoints), len(all_Xs_vnodes_ccpoints))
        )
        for i in range(len(Xs_edge_ccpoints)):
            for j in range(len(all_Xs_vnodes_ccpoints)):
                edgeX_vnodeX_dist_matrix[i, j] = np.linalg.norm(
                    Xs_edge_ccpoints[i, 2:5].astype(float)
                    - all_Xs_vnodes_ccpoints[j, 2:5].astype(float)
                )

        for k in range(len(Xs_edge_fpoints)):
            n_j, min_dist, edgeX_vnodeX_dist_matrix = find_nearest_neighbor(
                k, edgeX_vnodeX_dist_matrix
            )

            if min_dist > 4:
                unsaturated_linker.append(n)
                # print(
                #    "no xoo for edge node, this linker is a dangling unsaturated linker",
                #    n,
                # ) # debug
                continue
            # add the xoo to the edge node

            nearest_vnode = all_Xs_vnodes_ind[n_j][0]
            nearest_X_ind_in_vnode = all_Xs_vnodes_ind[n_j][1]
            matched_vnode_X.append(all_Xs_vnodes_ind[n_j])
            corresponding_o_indices = xoo_dict[nearest_X_ind_in_vnode]
            xoo_ind_in_vnode = [[nearest_X_ind_in_vnode] + corresponding_o_indices]
            xoo_fpoints_in_vnode = [
                eG.nodes[nearest_vnode]["f_points"][i] for i in xoo_ind_in_vnode
            ]
            xoo_fpoints_in_vnode = np.vstack(xoo_fpoints_in_vnode)
            eG.nodes[n]["f_points"] = np.vstack(
                (eG.nodes[n]["f_points"], xoo_fpoints_in_vnode)
            )
            # print('add xoo to edge node',n) #debug
    return eG, unsaturated_linker, matched_vnode_X, xoo_dict


def addxoo2edge_ditopic(eG, sc_unit_cell):
    xoo_dict = get_xoo_dict_of_node(eG, sc_unit_cell)
    matched_vnode_X = []
    unsaturated_linker = []
    # for every X atom in the EDGE node, search for the paired(nearest) X atom in the connected V node
    # and then use the xoo_dict of the connected V node to extract the xoos of the connected V node
    # and then add the xoos to the EDGE node
    # all xoo_node for the V node is the same
    EDGE_nodes = [n for n in eG.nodes() if pname(n) == "EDGE"]
    for n in EDGE_nodes:
        Xs_edge_indices, Xs_edge_fpoints = fetch_X_atoms_ind_array(
            eG.nodes[n]["f_points"], 0, "X"
        )
        Xs_edge_ccpoints = np.hstack(
            (
                Xs_edge_fpoints[:, 0:2],
                np.dot(sc_unit_cell, Xs_edge_fpoints[:, 2:5].astype(float).T).T,
            )
        )  # NOTE: modified to skip atom type
        V_nodes = [i for i in eG.neighbors(n) if pname(i) != "EDGE"]
        if len(V_nodes) == 0:
            # unsaturated_linker.append(n)
            print(
                "no V node connected to this edge node, this linker is a isolated linker, will be ignored",
                n,
            )
            continue
        all_Xs_vnodes_ind = []
        all_Xs_vnodes_ccpoints = np.zeros((0, 5))
        for v in V_nodes:
            # find the connected V node
            Xs_vnode_indices, Xs_vnode_fpoints = fetch_X_atoms_ind_array(
                eG.nodes[v]["f_points"], 0, "X"
            )
            Xs_vnode_ccpoints = np.hstack(
                (
                    Xs_vnode_fpoints[:, 0:2],
                    np.dot(sc_unit_cell, Xs_vnode_fpoints[:, 2:5].astype(float).T).T,
                )
            )  # NOTE: modified to skip atom type
            for ind in Xs_vnode_indices:
                all_Xs_vnodes_ind.append([v, ind, n])
            all_Xs_vnodes_ccpoints = np.vstack(
                (all_Xs_vnodes_ccpoints, Xs_vnode_ccpoints)
            )
        edgeX_vnodeX_dist_matrix = np.zeros(
            (len(Xs_edge_ccpoints), len(all_Xs_vnodes_ccpoints))
        )
        for i in range(len(Xs_edge_ccpoints)):
            for j in range(len(all_Xs_vnodes_ccpoints)):
                edgeX_vnodeX_dist_matrix[i, j] = np.linalg.norm(
                    Xs_edge_ccpoints[i, 2:54].astype(float)
                    - all_Xs_vnodes_ccpoints[j, 2:5].astype(float)
                )
        for k in range(len(Xs_edge_fpoints)):
            n_j, min_dist, _ = find_nearest_neighbor(k, edgeX_vnodeX_dist_matrix)
            if min_dist > 4:
                unsaturated_linker.append(n)
                # print(
                #   min_dist,
                #   "no xoo for edge node, this linker is a dangling unsaturated linker",
                #   n,
                # ) # debug
                continue
            # add the xoo to the edge node
            nearest_vnode = all_Xs_vnodes_ind[n_j][0]
            nearest_X_ind_in_vnode = all_Xs_vnodes_ind[n_j][1]
            matched_vnode_X.append(all_Xs_vnodes_ind[n_j])
            corresponding_o_indices = xoo_dict[nearest_X_ind_in_vnode]
            xoo_ind_in_vnode = [[nearest_X_ind_in_vnode] + corresponding_o_indices]
            xoo_fpoints_in_vnode = [
                eG.nodes[nearest_vnode]["f_points"][i] for i in xoo_ind_in_vnode
            ]
            xoo_fpoints_in_vnode = np.vstack(xoo_fpoints_in_vnode)
            eG.nodes[n]["f_points"] = np.vstack(
                (eG.nodes[n]["f_points"], xoo_fpoints_in_vnode)
            )
            # print('add xoo to edge node',n) #debug
    return eG, unsaturated_linker, matched_vnode_X, xoo_dict


def find_unsaturated_node(eG, node_topics):
    # find unsaturated node V in eG
    unsaturated_node = []
    for n in eG.nodes():
        if pname(n) != "EDGE":
            real_neighbor = []
            for cn in eG.neighbors(n):
                if eG.edges[(n, cn)]["type"] == "real":
                    real_neighbor.append(cn)
            if len(real_neighbor) < node_topics:
                unsaturated_node.append(n)
    return unsaturated_node


def find_unsaturated_linker(eG, linker_topics):
    # find unsaturated linker in eG
    unsaturated_linker = []
    for n in eG.nodes():
        if pname(n) == "EDGE" and len(list(eG.neighbors(n))) < linker_topics:
            unsaturated_linker.append(n)
    return unsaturated_linker
