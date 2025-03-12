import numpy as np
import re
import os
import networkx as nx
from _place_node_edge import fractional_to_cartesian, cartesian_to_fractional
from _superimpose import superimpose, superimpose_rotateonly
from _readcif_pdb import process_node_pdb
from add_dummy2node import nn
from makesuperG import pname


def check_inside_unit_cell(point):
    return all([i >= -0.01 and i <= 1.01 for i in point])


def find_pair_v_e_c(
    vvnode333, ecnode333, eenode333, unit_cell
):  # exist center of linker  in mof
    G = nx.Graph()
    pair_ve = []
    for e in eenode333:
        # print(e, "check")
        # dist_v_e = np.linalg.norm(vvnode333 - e, axis=1)
        dist_v_e = np.linalg.norm(np.dot(unit_cell, (vvnode333 - e).T).T, axis=1)
        # find two v which are nearest to e, and at least one v is in [0,1] unit cell
        v1 = vvnode333[np.argmin(dist_v_e)]
        v1_idx = np.argmin(dist_v_e)
        dist_c_e = np.linalg.norm(np.dot(unit_cell, (ecnode333 - e).T).T, axis=1)
        # find two v which are nearest to e, and at least one v is in [0,1] unit cell
        v2 = ecnode333[np.argmin(dist_c_e)]
        v2_idx = np.argmin(dist_c_e)
        # print(v1, v2, "v1,v2")

        # find the center of the pair of v
        center = (v1 + v2) / 2
        # check if there is a v in [0,1] unit cell
        if check_inside_unit_cell(v1) or check_inside_unit_cell(v2):
            # check if the center of the pair of v is around e
            # if abs(np.linalg.norm(v1 - e)+np.linalg.norm(v2 - e) - np.linalg.norm(v1 - v2))< 1e-2: #v1,v2,e are colinear
            if np.linalg.norm(center - e) < 0.1:
                # print(e,v1,v2,'check')
                G.add_node("V" + str(v1_idx), fcoords=v1, note="V")
                G.add_node("CV" + str(v2_idx), fcoords=v2, note="CV")
                (
                    G.add_edge(
                        "V" + str(v1_idx),
                        "CV" + str(v2_idx),
                        fcoords=(v1, v2),
                        fc_center=e,
                    ),
                )
                pair_ve.append(("V" + str(v1_idx), "CV" + str(v2_idx), e))

    return pair_ve, len(pair_ve), G


def sort_nodes_by_type_connectivity(G):
    CV_nodes = [n for n in G.nodes() if G.nodes[n]["note"] == "CV"]
    if len(CV_nodes) == 0:  # ditopic linker MOF
        Vnodes = [n for n in G.nodes() if G.nodes[n]["type"] == "V"]
        DVnodes = [n for n in G.nodes() if G.nodes[n]["type"] == "DV"]
        Vnodes = sorted(Vnodes, key=lambda x: G.degree(x), reverse=True)
        DVnodes = sorted(DVnodes, key=lambda x: G.degree(x), reverse=True)
        return Vnodes + DVnodes
    else:
        # CV+V
        # get CV_Vnode
        CV_Vnodes = [
            n
            for n in G.nodes()
            if G.nodes[n]["type"] == "V" and G.nodes[n]["note"] == "CV"
        ]
        CV_DVnodes = [
            n
            for n in G.nodes()
            if G.nodes[n]["type"] == "DV" and G.nodes[n]["note"] == "CV"
        ]
        V_Vnodes = [
            n
            for n in G.nodes()
            if G.nodes[n]["type"] == "V" and G.nodes[n]["note"] == "V"
        ]
        V_DVnodes = [
            n
            for n in G.nodes()
            if G.nodes[n]["type"] == "DV" and G.nodes[n]["note"] == "V"
        ]
        CV_Vnodes = sorted(CV_Vnodes, key=lambda x: G.degree(x), reverse=True)
        CV_DVnodes = sorted(CV_DVnodes, key=lambda x: G.degree(x), reverse=True)
        V_Vnodes = sorted(V_Vnodes, key=lambda x: G.degree(x), reverse=True)
        V_DVnodes = sorted(V_DVnodes, key=lambda x: G.degree(x), reverse=True)

        return V_Vnodes + CV_Vnodes + V_DVnodes + CV_DVnodes


def check_edge_inunitcell(G, e):
    if "DV" in G.nodes[e[0]]["type"] or "DV" in G.nodes[e[1]]["type"]:
        return False
    return True


def arr_dimension(arr):
    if arr.ndim > 1:
        return 2
    else:
        return 1


def check_supercell_box_range(point, supercell, buffer_plus, buffer_minus):
    # to cleave eG to supercell box

    supercell_x = supercell[0] + buffer_plus
    supercell_y = supercell[1] + buffer_plus
    supercell_z = supercell[2] + buffer_plus
    if (
        point[0] >= 0 + buffer_minus
        and point[0] <= supercell_x
        and point[1] >= 0 + buffer_minus
        and point[1] <= supercell_y
        and point[2] >= 0 + buffer_minus
        and point[2] <= supercell_z
    ):
        return True
    else:
        # print(point, 'out of supercell box range:  [',supercell_x,supercell_y,supercell_z, '],   will be excluded') #debug
        return False


def put_V_ahead_of_CV(e):
    if "CV" in e[0] and "V" in e[1]:
        return (e[1], e[0])
    else:
        return e


def find_and_sort_edges_bynodeconnectivity(graph, sorted_nodes):
    all_edges = list(graph.edges())

    sorted_edges = []
    # add unit_cell edge first

    ei = 0
    while ei < len(all_edges):
        e = all_edges[ei]
        if check_edge_inunitcell(graph, e):
            sorted_edges.append(put_V_ahead_of_CV(e))
            all_edges.pop(ei)
        ei += 1
    # sort edge by sorted_nodes
    for n in sorted_nodes:
        ei = 0
        while ei < len(all_edges):
            e = all_edges[ei]
            if n in e:
                if n == e[0]:
                    sorted_edges.append(put_V_ahead_of_CV(e))
                else:
                    sorted_edges.append(put_V_ahead_of_CV((e[1], e[0])))
                all_edges.pop(ei)
            else:
                ei += 1

    return sorted_edges


def is_list_A_in_B(A, B):
    return all([np.allclose(a, b, atol=1e-8) for a, b in zip(A, B)])


# Function to fetch indices and coordinates of atoms with a specific label
def fetch_X_atoms_ind_array(array, column, X):
    # array: input array
    # column: column index to check for label
    # X: label to search for

    ind = [k for k in range(len(array)) if re.sub(r"\d", "", array[k, column]) == X]
    x_array = array[ind]
    return ind, x_array


def make_unsaturated_vnode_xoo_dict(
    unsaturated_node, xoo_dict, matched_vnode_xind, eG, sc_unit_cell
):
    """
    make a dictionary of the unsaturated node and the exposed X connected atom index and the corresponding O connected atoms
    """

    # process matched_vnode_xind make it to a dictionary
    matched_vnode_xind_dict = {}
    for [k, v, e] in matched_vnode_xind:
        if k in matched_vnode_xind_dict.keys():
            matched_vnode_xind_dict[k].append(v)
        else:
            matched_vnode_xind_dict[k] = [v]

    unsaturated_vnode_xind_dict = {}
    xoo_keys = list(xoo_dict.keys())
    # for each unsaturated node, get the upmatched x index and xoo atoms
    for unsat_v in unsaturated_node:
        if unsat_v in matched_vnode_xind_dict.keys():
            unsaturated_vnode_xind_dict[unsat_v] = [
                i for i in xoo_keys if i not in matched_vnode_xind_dict[unsat_v]
            ]
            # print(unsaturated_vnode_xind_dict[unsat_v],'unsaturated_vnode_xind_dict[unsat_v]') #DEBUG
        else:
            unsaturated_vnode_xind_dict[unsat_v] = xoo_keys

    # based on the unsaturated_vnode_xind_dict, add termination to the unsaturated node xoo
    # loop over unsaturated nodes, and find all exposed X atoms and use paied xoo atoms to form a termination
    unsaturated_vnode_xoo_dict = {}
    for vnode, exposed_x_indices in unsaturated_vnode_xind_dict.items():
        for xind in exposed_x_indices:
            x_fpoints = eG.nodes[vnode]["f_points"][xind]
            x_cpoints = np.hstack(
                (x_fpoints[0:2], fractional_to_cartesian(x_fpoints[2:5], sc_unit_cell))
            )  # NOTE: modified add the atom type and atom name
            oo_ind_in_vnode = xoo_dict[xind]
            oo_fpoints_in_vnode = [
                eG.nodes[vnode]["f_points"][i] for i in oo_ind_in_vnode
            ]
            oo_fpoints_in_vnode = np.vstack(oo_fpoints_in_vnode)
            oo_cpoints = np.hstack(
                (
                    oo_fpoints_in_vnode[:, 0:2],
                    fractional_to_cartesian(oo_fpoints_in_vnode[:, 2:5], sc_unit_cell),
                )
            )  # NOTE: modified add the atom type and atom name

            unsaturated_vnode_xoo_dict[(vnode, xind)] = {
                "xind": xind,
                "oo_ind": oo_ind_in_vnode,
                "x_fpoints": x_fpoints,
                "x_cpoints": x_cpoints,
                "oo_fpoints": oo_fpoints_in_vnode,
                "oo_cpoints": oo_cpoints,
            }

    return (
        unsaturated_vnode_xind_dict,
        unsaturated_vnode_xoo_dict,
        matched_vnode_xind_dict,
    )


def update_matched_nodes(removed_nodes, removed_edges, matched_vnode_xind):
    # if linked edge is removed and the connected node is not removed, then remove this line from matched_vnode_xind
    # add remove the middle xind of the node to matched_vnode_xind_dict[node] list
    to_remove_row = []

    for i in range(len(matched_vnode_xind)):
        node, xind, edge = matched_vnode_xind[i]
        if edge in removed_edges and node not in removed_nodes:
            to_remove_row.append(i)
        elif node in removed_nodes:
            to_remove_row.append(i)
    # remove the rows
    matched_vnode_xind = [
        i for j, i in enumerate(matched_vnode_xind) if j not in to_remove_row
    ]
    return matched_vnode_xind


# functions for write
# write gro file
def extract_node_edge_term(tG, sc_unit_cell):
    nodes_tG = []
    terms_tG = []
    edges_tG = []
    node_res_num = 0
    term_res_num = 0
    edge_res_num = 0
    nodes_check_set = set()
    nodes_name_set = set()
    edges_check_set = set()
    edges_name_set = set()
    terms_check_set = set()
    terms_name_set = set()
    for n in tG.nodes():
        if pname(n) != "EDGE":
            postions = tG.nodes[n]["noxoo_f_points"]
            name = tG.nodes[n]["name"]
            nodes_check_set.add(len(postions))
            nodes_name_set.add(name)
            if len(nodes_check_set) > len(nodes_name_set):
                raise ValueError("node index is not continuous")
            node_res_num += 1
            nodes_tG.append(
                np.hstack(
                    (
                        np.tile(
                            np.array([node_res_num, name]), (len(postions), 1)
                        ),  # residue number and residue name
                        postions[:, 1:2],  # atom type (element)
                        fractional_to_cartesian(
                            postions[:, 2:5], sc_unit_cell
                        ),  # Cartesian coordinates
                        postions[:, 0:1],  # atom name
                        np.tile(np.array([n]), (len(postions), 1)),
                    )
                )
            )  # node name in eG is added to the last column
            if "term_c_points" in tG.nodes[n]:
                for term_ind_key, c_positions in tG.nodes[n]["term_c_points"].items():
                    terms_check_set.add(len(c_positions))
                    name = "T" + tG.nodes[n]["name"]
                    terms_name_set.add(name)
                    if len(terms_check_set) > len(terms_name_set):
                        raise ValueError("term index is not continuous")

                    term_res_num += 1
                    terms_tG.append(
                        np.hstack(
                            (
                                np.tile(
                                    np.array([term_res_num, name]),
                                    (len(c_positions), 1),
                                ),  # residue number and residue name
                                c_positions[:, 1:2],  # atom type (element)
                                c_positions[:, 2:5],  # Cartesian coordinates
                                c_positions[:, 0:1],  # atom name
                                np.tile(
                                    np.array([term_ind_key]), (len(c_positions), 1)
                                ),
                            )
                        )
                    )  # term name in eG is added to the last column

        elif pname(n) == "EDGE":
            postions = tG.nodes[n]["f_points"]
            name = tG.nodes[n]["name"]
            edges_check_set.add(len(postions))
            edges_name_set.add(name)
            if len(edges_check_set) > len(edges_name_set):
                print(edges_check_set)
                # raise ValueError('edge atom number is not continuous')
                print(
                    "edge atom number is not continuous,ERROR edge name:",
                    len(edges_check_set),
                    len(edges_name_set),
                )
            edge_res_num += 1
            edges_tG.append(
                np.hstack(
                    (
                        np.tile(
                            np.array([edge_res_num, name]), (len(postions), 1)
                        ),  # residue number and residue name
                        postions[:, 1:2],  # atom type (element)
                        fractional_to_cartesian(
                            postions[:, 2:5], sc_unit_cell
                        ),  # Cartesian coordinates
                        postions[:, 0:1],  # atom name
                        np.tile(np.array([n]), (len(postions), 1)),
                    )
                )
            )  # edge name in eG is added to the last column

    # nodes_tG = np.vstack(nodes_tG)
    # terms_tG = np.vstack(terms_tG)
    # edges_tG = np.vstack(edges_tG)
    return nodes_tG, edges_tG, terms_tG, node_res_num, edge_res_num, term_res_num


def convert_node_array_to_gro_lines(array, line_num_start, res_num_start):
    formatted_gro_lines = []

    for i in range(len(array)):
        line = array[i]
        ind_inres = i + 1
        name = line[1]
        value_atom_number_in_gro = int(ind_inres + line_num_start)  # atom_number
        value_label = re.sub("\d", "", line[2]) + str(ind_inres)  # atom_label
        value_resname = str(name)[0:3]  # +str(eG.nodes[n]['index'])  # residue_name
        value_resnumber = int(res_num_start + int(line[0]))  # residue number
        value_x = 0.1 * float(line[3])  # x
        value_y = 0.1 * float(line[4])  # y
        value_z = 0.1 * float(line[5])  # z
        formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
            value_resnumber,
            value_resname,
            value_label,
            value_atom_number_in_gro,
            value_x,
            value_y,
            value_z,
        )
        formatted_gro_lines.append(formatted_line + "\n")
    return formatted_gro_lines, value_atom_number_in_gro


def merge_node_edge_term(nodes_tG, edges_tG, terms_tG, node_res_num, edge_res_num):
    merged_node_edge_term = []
    line_num = 0
    for node in nodes_tG:
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            node, line_num, 0
        )
        merged_node_edge_term += formatted_gro_lines
    for edge in edges_tG:
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            edge, line_num, node_res_num
        )
        merged_node_edge_term += formatted_gro_lines
    for term in terms_tG:
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            term, line_num, node_res_num + edge_res_num
        )
        merged_node_edge_term += formatted_gro_lines
    return merged_node_edge_term


def save_node_edge_term_gro(merged_node_edge_term, gro_name):
    os.makedirs("output_gros", exist_ok=True)
    gro_name = os.path.join("output_gros", gro_name)
    with open(str(gro_name) + ".gro", "w") as f:
        head = []
        head.append("eG_NET\n")
        head.append(str(len(merged_node_edge_term)) + "\n")
        f.writelines(head)
        f.writelines(merged_node_edge_term)
        tail = ["10 10 10 \n"]
        f.writelines(tail)


# debug write gro function
def convert_positions_to_gro_lines(line, line_num, res_num, name):
    line_num += 1
    value_atom_number = int(line_num)  # atom_number
    value_label = re.sub("\d", "", line[0])  # atom_label
    value_resname = str(name)[0:3]  # +str(eG.nodes[n]['index'])  # residue_name
    value_resnumber = int(res_num)  # residue number
    value_x = 0.1 * float(line[1])  # x
    value_y = 0.1 * float(line[2])  # y
    value_z = 0.1 * float(line[3])  # z
    formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
        value_resnumber,
        value_resname,
        value_label,
        value_atom_number,
        value_x,
        value_y,
        value_z,
    )

    return formatted_line, line_num, res_num


def temp_save_eGterm_gro(eG, sc_unit_cell):
    with open("eG_TERM.gro", "w") as f:
        newgro = []
        res_num = 0
        line_num = 0
        for n in eG.nodes():
            if pname(n) != "EDGE":
                postions = eG.nodes[n]["f_points"]
                res_num += 1
                fc = np.asarray(postions[:, 2:5], dtype=float)
                cc = np.dot(sc_unit_cell, fc.T).T
                positionss = np.hstack((postions[:, 1:2], cc))
                for line in positionss:
                    formatted_line, line_num, res_num = convert_positions_to_gro_lines(
                        line, line_num, res_num, n
                    )
                    newgro.append(formatted_line + "\n")
                for term_ind_key, c_positions in eG.nodes[n]["term_c_points"].items():
                    for line in c_positions:
                        formatted_line, line_num, res_num = (
                            convert_positions_to_gro_lines(
                                line, line_num, res_num, term_ind_key
                            )
                        )
                        newgro.append(formatted_line + "\n")

            elif pname(n) == "EDGE":
                postions = eG.nodes[n]["f_points"]
                res_num += 1
                fc = np.asarray(postions[:, 2:5], dtype=float)
                cc = np.dot(sc_unit_cell, fc.T).T
                positionss = np.hstack((postions[:, 1:2], cc))
                for line in positionss:
                    formatted_line, line_num, res_num = convert_positions_to_gro_lines(
                        line, line_num, res_num, n
                    )
                    newgro.append(formatted_line + "\n")
        head = []
        head.append("eG_TERM\n")
        head.append(str(line_num) + "\n")
        tail = ["10 10 10 \n"]
        f.writelines(head)
        f.writelines(newgro)
        f.writelines(tail)


def replace_edges_by_callname(
    edge_n_list, eG, sc_unit_cell_inv, new_linker_pdb, prefix="R"
):
    new_linker_atoms, new_linker_ccoords, new_linker_x_ccoords = process_node_pdb(
        new_linker_pdb, "X"
    )
    for edge_n in edge_n_list:
        edge_n = edge_n
        edge_f_points = eG.nodes[edge_n]["f_points"]
        x_indices = [
            i for i in range(len(edge_f_points)) if nn(edge_f_points[i][0]) == "X"
        ]
        edge_x_points = edge_f_points[x_indices]
        edge_com = np.mean(edge_x_points[:, 2:5].astype(float), axis=0)
        edge_x_fcoords = edge_x_points[:, 2:5].astype(float) - edge_com

        new_linker_x_fcoords = cartesian_to_fractional(
            new_linker_x_ccoords, sc_unit_cell_inv
        )
        new_linker_fcoords = cartesian_to_fractional(
            new_linker_ccoords, sc_unit_cell_inv
        )

        _, rot, trans = superimpose(new_linker_x_fcoords, edge_x_fcoords)
        replaced_linker_fcoords = np.dot(new_linker_fcoords, rot) + edge_com
        replaced_linker_f_points = np.hstack(
            (new_linker_atoms, replaced_linker_fcoords)
        )

        eG.nodes[edge_n]["f_points"] = replaced_linker_f_points
        eG.nodes[edge_n]["name"] = prefix + eG.nodes[edge_n]["name"]

    return eG


# the following functions are used for the split node to metal, hho,ho,o and update name and residue number


def extract_node_name_from_gro_resindex(res_index, node_array_list):
    node_array = np.vstack(node_array_list)
    nodes_name = set()
    for node_ind in res_index:
        node_name = node_array[node_array[:, 0] == str(node_ind)][:, -1]
        name_set = set(node_name)
        nodes_name = nodes_name.union(name_set)
    return nodes_name


def make_dummy_split_node_dict(dummy_node_name):
    node_split_dict = {}
    dict_path = dummy_node_name.split(".")[0] + "_dict"
    with open(dict_path, "r") as f:
        lines = f.readlines()
    node_res_counts = 0
    for li in lines:
        li = li.strip("\n")
        key = li[:20].strip(" ")
        value = li[-4:].strip(" ")
        node_split_dict[key] = int(value)
    return node_split_dict


def chunk_array(chunk_list, array, chunk_num, chunksize):
    chunk_list.extend(
        array[i * chunksize : (i + 1) * chunksize] for i in range(chunk_num)
    )
    return chunk_list


def rename_node_arr(node_split_dict, node_arr):
    metal_count = node_split_dict["METAL_count"]
    dummy_len = int(node_split_dict["dummy_res_len"])
    metal_num = metal_count * dummy_len
    hho_num = node_split_dict["HHO_count"] * 3
    ho_num = node_split_dict["HO_count"] * 2
    o_num = node_split_dict["O_count"] * 1
    metal_range = metal_num
    hho_range = metal_range + hho_num
    ho_range = hho_range + ho_num
    o_range = ho_range + o_num
    # print(metal_range,hho_range,ho_range,o_range) #debug

    metals_list = []
    hhos_list = []
    hos_list = []
    os_list = []
    for idx in set(node_arr[:, 0]):
        idx_arr = node_arr[node_arr[:, 0] == idx]
        if metal_num > 0:
            metal = idx_arr[0:metal_range].copy()
            metal[:, 1] = "METAL"
            metals_list = chunk_array(
                metals_list, metal, node_split_dict["METAL_count"], dummy_len
            )
        if hho_num > 0:
            hho = idx_arr[metal_range:hho_range].copy()
            hho[:, 1] = "HHO"
            hhos_list = chunk_array(hhos_list, hho, node_split_dict["HHO_count"], 3)
        if ho_num > 0:
            ho = idx_arr[hho_range:ho_range].copy()
            ho[:, 1] = "HO"
            hos_list = chunk_array(hos_list, ho, node_split_dict["HO_count"], 2)
        if o_num > 0:
            o = idx_arr[ho_range:o_range].copy()
            o[:, 1] = "O"
            os_list = chunk_array(os_list, o, node_split_dict["O_count"], 1)

    return metals_list, hhos_list, hos_list, os_list


def merge_metal_list_to_node_array(
    merged_node_edge_term, metals_list, line_num, res_count
):
    if any([len(metal) == 0 for metal in metals_list]):
        return merged_node_edge_term, line_num, res_count
    for i in range(len(metals_list)):
        metal = metals_list[i]
        metal[:, 0] = i + 1
        formatted_gro_lines, line_num = convert_node_array_to_gro_lines(
            metal, line_num, res_count
        )
        merged_node_edge_term += formatted_gro_lines
    res_count += len(metals_list)
    return merged_node_edge_term, line_num, res_count


def reorder_pairs_for_CV_edgeorder(pairs):
    # sort the pairs by the ind of CV nodes if there are any
    # make the dictionary of CV nodes and their optimized paired_X index in CV node and paired_V_node, and X ind inside the V node
    optimized_CV_pairs = {}
    for pair, paired_values in pairs.items():
        if "CV" in pair[0]:
            if pair[0] in optimized_CV_pairs:
                optimized_CV_pairs[pair[0]].append(
                    (paired_values[0], (pair[1], paired_values[1]))
                )
            else:
                optimized_CV_pairs[pair[0]] = [
                    (paired_values[0], (pair[1], paired_values[1]))
                ]
        elif "CV" in pair[1]:
            if pair[1] in optimized_CV_pairs:
                optimized_CV_pairs[pair[1]].append(
                    (paired_values[1], (pair[0], paired_values[0]))
                )
            else:
                optimized_CV_pairs[pair[1]] = [
                    (paired_values[1], (pair[0], paired_values[0]))
                ]

    # sort the values of the CV nodes
    # muiltitopiclinker_topics = max lengths of the values in the optimized_CV_pairs
    muiltitopiclinker_topics = max(
        [len(values) for values in optimized_CV_pairs.values()]
    )

    cvnode_to_remove = []
    for CV_node, values in optimized_CV_pairs.items():
        if len(values) < muiltitopiclinker_topics:
            # mark the CV node for removal
            cvnode_to_remove.append(CV_node)
        else:
            # sort the values
            values.sort(key=lambda x: x[0])

    new_pairs = {}
    for cvnode, pair_info in optimized_CV_pairs.items():
        for i in range(len(pair_info)):
            if i == 0:
                new_pairs[(cvnode, pair_info[i][1][0])] = (
                    pair_info[i][0],
                    pair_info[i][1][1],
                )
            else:
                new_pairs[(cvnode, pair_info[i][1][0])] = (
                    pair_info[i][0],
                    pair_info[i][1][1],
                )

    return new_pairs, cvnode_to_remove


def recenter_and_norm_vectors(vectors, extra_mass_center=None):
    vectors = np.array(vectors)
    if extra_mass_center is not None:
        mass_center = extra_mass_center
    else:
        mass_center = np.mean(vectors, axis=0)
    vectors = vectors - mass_center
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return vectors, mass_center


def get_connected_nodes_vectors(node, G):
    # use adjacent nodes to get vectors
    vectors = []
    for i in list(G.neighbors(node)):
        vectors.append(G.nodes[i]["ccoords"])
    return vectors, G.nodes[node]["ccoords"]


def get_rot_trans_matrix(node, G, sorted_nodes, Xatoms_positions_dict):
    node_id = sorted_nodes.index(node)
    node_xvecs = Xatoms_positions_dict[node_id][:, 1:]
    vecsA, _ = recenter_and_norm_vectors(node_xvecs, extra_mass_center=None)
    v2, node_center = get_connected_nodes_vectors(node, G)
    vecsB, _ = recenter_and_norm_vectors(v2, extra_mass_center=node_center)
    _, rot, tran = superimpose_rotateonly(vecsA, vecsB)
    return rot, tran
