from frag_recognizer import create_lG, plot2dedge
import networkx as nx
import veloxchem as vlx
import os
import numpy as np


def create_graph_from_matrix(matrix):
    """Create a graph from a given adjacency matrix."""
    G = nx.Graph()
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] > 0:
                G.add_edge(i, j, weight=matrix[i][j])
    return G


def find_isomorphism_and_mapping(matrix1, matrix2):
    """Check if two matrices are isomorphic and return the mapping."""
    G1 = create_graph_from_matrix(matrix1)
    G2 = create_graph_from_matrix(matrix2)

    gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2)
    if gm.is_isomorphic():
        return True, gm.mapping
    else:
        print("The graphs are not isomorphic.", len(G1.nodes()), len(G2.nodes()))
        print("G1.edges()", G1.edges())
        print("G2.edges()", G2.edges())
        return False, None


def get_mapping_between_nometal_linker_xyz(
    linker_topic,
    center_frag_nodes_num,
    center_Xs,
    single_frag_nodes_num,
    frag_Xs,
    template_xyz,
    new_xyz="Residues/EDG.xyz",
):
    molecule = vlx.Molecule.read_xyz_file(template_xyz)
    _, mol_metals, _ = create_lG(molecule)

    m2 = vlx.Molecule.read_xyz_file(new_xyz)
    coords = m2.get_coordinates_in_angstrom()
    G, metals, mass_center_angstrom = create_lG(m2)
    if linker_topic == 2:
        matrix_mof = m2.get_connectivity_matrix()
        matrix_user = molecule.get_connectivity_matrix()
        # G_user = create_graph_from_matrix(matrix_user)
        isomorphic, mapping = find_isomorphism_and_mapping(matrix_mof, matrix_user)
        if isomorphic:
            print("The graphs are isomorphic.")
            print("Node mapping:", mapping)
            # permuted_matrix = permute_matrix(matrix1, mapping)
            # print("Permuted Matrix:\n", permuted_matrix)
        else:
            print("The graphs are not isomorphic.")
            plot2dedge(G, coords, G.edges())
        return mapping, metals, mol_metals
    else:
        if len(metals) == 1:
            coords = m2.get_coordinates_in_angstrom()
            labels = m2.get_labels()
            center_nums = center_frag_nodes_num + 1
            frag_nums = single_frag_nodes_num
            # raise ValueError('cannot process molecule including metals,try the function with metal linker')

        else:
            coords = m2.get_coordinates_in_angstrom()
            labels = m2.get_labels()
            center_nums = center_frag_nodes_num
            frag_nums = single_frag_nodes_num

        center_range = range(0, center_nums)
        frag1_range = range(center_nums, center_nums + frag_nums)
        frag2_range = range(center_nums + frag_nums, center_nums + frag_nums * 2)
        frag3_range = range(center_nums + frag_nums * 2, center_nums + frag_nums * 3)
        frag4_range = range(center_nums + frag_nums * 3, center_nums + frag_nums * 4)

        frag1_Xs = [
            i + center_nums for i in frag_Xs
        ]  # frag_Xs is Xs indices in a single outer_edge_frag
        frag2_Xs = [j + frag_nums for j in frag1_Xs]
        frag3_Xs = [k + frag_nums for k in frag2_Xs]

        if linker_topic == 3:
            cn_bond = []
            frag4_Xs = []
            for x_i in center_Xs:
                x_center = coords[x_i]
                for x_j in frag1_Xs + frag2_Xs + frag3_Xs:
                    x_frag = coords[x_j]
                    if np.linalg.norm(x_center - x_frag) < 3.5:
                        cn_bond.append((x_i, x_j))
            left_xs = list(
                set(frag1_Xs + frag2_Xs + frag3_Xs) - set(i[1] for i in cn_bond)
            )
            print(left_xs, "left_xs")
            if linker_topic == 3:
                for m in range(len(labels) - 3 * 3, len(labels)):
                    if labels[m] == "C":
                        x_ooc = coords[m]
                        for n in left_xs:
                            x_frag = coords[n]
                            if np.linalg.norm(x_frag - x_ooc) < 5:
                                cn_bond.append((n, m))

        elif linker_topic == 4:
            # xX bond should follow centernodeX order
            frag4_Xs = [m + frag_nums for m in frag3_Xs]
            cn_bond = []
            center_Xs.sort()
            frag_Xs = frag1_Xs + frag2_Xs + frag3_Xs + frag4_Xs
            frag_Xs.sort()

            for x_i in center_Xs:
                x_center = coords[x_i]
                for x_j in frag1_Xs + frag2_Xs + frag3_Xs + frag4_Xs:
                    x_frag = coords[x_j]
                    if np.linalg.norm(x_center - x_frag) < 4:
                        cn_bond.append((x_i, x_j))
            left_xs = list(
                set(frag1_Xs + frag2_Xs + frag3_Xs + frag4_Xs)
                - set(i[1] for i in cn_bond)
            )
            print(left_xs, "left_xs")
            for m in range(len(labels) - 4 * 3, len(labels)):
                if labels[m] == "C":
                    x_ooc = coords[m]
                    for n in left_xs:
                        x_frag = coords[n]

                        if np.linalg.norm(x_frag - x_ooc) < 5:
                            cn_bond.append((n, m))
                            print(
                                "np.linalg.norm(x_frag-x_ooc)",
                                n,
                                m,
                                np.linalg.norm(x_frag - x_ooc),
                            )

        # seperate all frags and clean bonds between frags

        for edge in G.edges():
            if edge[0] in center_range:
                if edge[1] not in center_range:
                    G.remove_edge(edge[0], edge[1])
                    print(edge)
            elif edge[0] in frag1_range:
                if edge[1] not in frag1_range:
                    G.remove_edge(edge[0], edge[1])
                    print(edge)
            elif edge[0] in frag2_range:
                if edge[1] not in frag2_range:
                    G.remove_edge(edge[0], edge[1])
                    print(edge)
            elif edge[0] in frag3_range:
                if edge[1] not in frag3_range:
                    G.remove_edge(edge[0], edge[1])
                    print(edge)
            elif edge[0] in frag4_range:
                if edge[1] not in frag4_range:
                    G.remove_edge(edge[0], edge[1])
                    print(edge)

        if len(metals) > 0:
            # remove all edges from metal

            edges_with_Metal = list(G.edges(int(metals[0])))
            G.remove_edges_from(edges_with_Metal)

        if len(metals) == 0:
            if len(sorted(nx.connected_components(G))) == 2 * linker_topic + 1:
                print(len(sorted(nx.connected_components(G))), " parts are seperated")
            else:
                print(len(sorted(nx.connected_components(G))), " parts are seperated")
                # plot2dedge(G,coords,G.edges())
                raise ValueError("nx.connected_components is not linker_topic+1")
        else:
            if len(sorted(nx.connected_components(G))) == 2 * linker_topic + 2:
                print(len(sorted(nx.connected_components(G))), " parts are seperated")
            else:
                print(len(sorted(nx.connected_components(G))), " parts are seperated")
                # plot2dedge(G,coords,G.edges())
                raise ValueError("nx.connected_components is not linker_topic+2")

        # rebuild bond between fragXs and get_connectivity matrix of EDGE
        cn_matrix = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
        # edges in seperated frags
        for edge in G.edges():
            cn_matrix[edge[0], edge[1]] = 1
            cn_matrix[edge[1], edge[0]] = 1
        # +xx #XX bonds
        for xx_cn in cn_bond:
            cn_matrix[xx_cn[0], xx_cn[1]] = 1
            cn_matrix[xx_cn[1], xx_cn[0]] = 1
            G.add_edge(int(xx_cn[0]), int(xx_cn[1]))
        if len(metals) == 0:
            if len(sorted(nx.connected_components(G))) == 1:
                print("reconnect succeed")
            else:
                print(len(sorted(nx.connected_components(G))), cn_bond)
                plot2dedge(G, coords, G.edges())
                raise ValueError("reconnect failed")
        elif len(metals) == 1:
            if len(sorted(nx.connected_components(G))) == 2:
                print("reconnect succeed")
            else:
                print(len(sorted(nx.connected_components(G))), cn_bond)
                plot2dedge(G, coords, G.edges())
                raise ValueError("reconnect failed")

        matrix_mof = cn_matrix
        matrix_user = molecule.get_connectivity_matrix()
        if len(mol_metals) > 0:
            matrix_user[:, mol_metals[0]] = 0
            matrix_user[mol_metals[0], :] = 0

        # G_user = create_graph_from_matrix(matrix_user)

        isomorphic, mapping = find_isomorphism_and_mapping(matrix_mof, matrix_user)
        if isomorphic:
            print("The graphs are isomorphic.")
            print("Node mapping:", mapping)
            # permuted_matrix = permute_matrix(matrix1, mapping)
            # print("Permuted Matrix:\n", permuted_matrix)
        else:
            print("The graphs are not isomorphic.")
            plot2dedge(G, coords, G.edges())
    return mapping, metals, mol_metals


def write_mapping_file(path, mapping, metals, mol_metals, map_name="linker_ff_mapping"):
    map_path = os.path.join(path, map_name)
    if os.path.exists(path):
        with open(map_path, "w") as f:
            for i in list(mapping):
                f.write(str(mapping[i] + 1))
                f.write("\t")
                f.write(str(i + 1))
                f.write("\n")
            if len(mol_metals) > 0:
                f.write(str(mol_metals[0] + 1))
                f.write("\t")
            if len(metals) > 0:
                f.write(str(metals[0] + 1))
            f.write("\n")

        return map_path
    else:
        raise ValueError("cannot find /Residues/parsedfile/ folder")
