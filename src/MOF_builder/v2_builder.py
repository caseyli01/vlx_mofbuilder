import re
import time
import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R

from _readcif_pdb import (
    process_node,
    read_cif,
    read_pdb,
    process_node_pdb,
    extract_type_atoms_fcoords_in_primitive_cell,
)
from _node_rotation_matrix_optimizer import (
    optimize_rotations_pre,
    optimize_rotations_after,
    apply_rotations_to_atom_positions,
    apply_rotations_to_Xatoms_positions,
)
from _node_rotation_matrix_optimizer import update_ccoords_by_optimized_cell_params
from _scale_cif_optimizer import optimize_cell_parameters
from _place_node_edge import (
    addidx,
    get_edge_lengths,
    update_node_ccoords,
    unit_cell_to_cartesian_matrix,
    fractional_to_cartesian,
    cartesian_to_fractional,
)
from _superimpose import superimpose, superimpose_rotation_only
from v2_functions import (
    arr_dimension,
    fetch_X_atoms_ind_array,
    sort_nodes_by_type_connectivity,
    find_and_sort_edges_bynodeconnectivity,
    is_list_A_in_B,
    get_rot_trans_matrix,
    update_matched_nodes_xind,
)
from make_eG import (
    superG_to_eG_ditopic,
    superG_to_eG_multitopic,
    remove_node_by_index,
    addxoo2edge_ditopic,
    addxoo2edge_multitopic,
    find_unsaturated_node,
    find_unsaturated_linker,
)
from multiedge_bundling import bundle_multiedge
from makesuperG import (
    pname,
    replace_DV_with_corresponding_V,
    replace_bundle_dvnode_with_vnode,
    make_super_multiedge_bundlings,
)
from makesuperG import (
    check_multiedge_bundlings_insuperG,
    add_virtual_edge,
    update_supercell_bundle,
    update_supercell_edge_fpoints,
    update_supercell_node_fpoints_loose,
)
from _terminations import termpdb, Xpdb
from v2_functions import (
    make_unsaturated_vnode_xoo_dict,
    check_supercell_box_range,
    save_node_edge_term_gro,
    extract_node_edge_term,
    merge_node_edge_term,
    reorder_pairs_for_CV_edgeorder,
)
from _learn_template import (
    make_supercell_3x3x3,
    find_pair_v_e,
    find_pair_v_e_c,
    extract_unit_cell,
)
from _learn_template import add_ccoords, set_DV_V, set_DE_E


class net_optimizer:
    """
    net_optimizer is a class to optimize the node and edge structure of the MOF, add terminations to nodes.

    :param template_cif (str):
        cif file of the template, including only V and E *(EC)nodes info in primitive cell
    Instance variables:
        - node_cif (str):cif file of the node
        - node_target_type (str):metal atom type of the node
        - node_unit_cell (array):unit cell of the node
        - node_atom (array):2 columns, atom_name, atom_type of the node
        - node_x_fcoords (array):fractional coordinates of the X connected atoms of node
        - node_fcoords (array):fractional coordinates of the whole node
        - node_x_ccoords (array):cartesian coordinates of the X connected atoms of node
        - node_coords (array):cartesian coordinates of the whole node
        - linker_cif (str):cif file of the ditopic linker or branch of multitopic linker
        - linker_unit_cell (array):unit cell of the ditopic linker or branch of multitopic linker
        - linker_atom (array):2 columns, atom_name, atom_type of the ditopic linker or branch of multitopic linker
        - linker_x_fcoords (array):fractional coordinates of the X connected atoms of ditopic linker or branch of multitopic linker
        - linker_fcoords (array):fractional coordinates of the whole ditopic linker or branch of multitopic linker
        - linker_x_ccoords (array):cartesian coordinates of the X connected atoms of ditopic linker or branch of multitopic linker
        - linker_length (float):distance between two X-X connected atoms of the ditopic linker or branch of multitopic linker
        - linker_ccoords (array):cartesian coordinates of the whole ditopic linker or branch of multitopic linker
        - linker_center_cif (str):cif file of the center of the multitopic linker
        - ec_unit_cell (array):unit cell of the center of the multitopic linker
        - ec_atom (array):2 columns, atom_name, atom_type of the center of the multitopic linker
        - ec_x_vecs (array):fractional coordinates of the X connected atoms of the center of the multitopic linker
        - ec_fcoords (array):fractional coordinates of the whole center of the multitopic linker
        - ec_xcoords (array):cartesian coordinates of the X connected atoms of the center of the multitopic linker
        - eccoords (array):cartesian coordinates of the whole center of the multitopic linker
        - constant_length (float):constant length to add to the linker length, normally 1.54 for single bond of C-C,
        because C is always used as the connecting atom in the builder
        - maxfun (int):maximum number of function evaluations for the node rotation optimization
        - opt_method (str):optimization method for the node rotation optimization
        - G (networkx graph):graph of the template
        - node_max_degree (int):maximum degree of the node in the template, should be the same as the node topic
        - sorted_nodes (list):sorted nodes in the template by connectivity
        - sorted_edges (list):sorted edges in the template by connectivity
        - sorted_edges_of_sortednodeidx (list):sorted edges in the template by connectivity with the index of the sorted nodes
        - optimized_rotations (dict):optimized rotations for the nodes in the template
        - optimized_params (array):optimized cell parameters for the template topology to fit the target MOF cell
        - new_edge_length (float):new edge length of the ditopic linker or branch of multitopic linker, 2*constant_length+linker_length
        - optimized_pair (dict): pair of connected nodes in the template with the index of the X connected atoms, used for the edge placement
        - scaled_rotated_node_positions (dict):scaled and rotated node positions in the target MOF cell
        - scaled_rotated_Xatoms_positions (dict):scaled and rotated X connected atom positions of nodes in the target MOF cell
        - sc_unit_cell (array):(scaled) unit cell of the target MOF cell
        - sc_unit_cell_inv (array):inverse of the (scaled) unit cell of the target MOF cell
        - sG_node (networkx graph):graph of the target MOF cell
        - nodes_atom (dict):atom name and atom type of the nodes
        - rotated_node_positions (dict):rotated node positions
        - supercell (array): supercell set by user, along x,y,z direction
        - multiedge_bundlings (dict):multiedge bundlings of center and branches of the multitopic linker, used for the supercell
        construction and merging of center and branches to form one EDGE
        - prim_multiedge_bundlings (dict):multiedge bundlings in primitive cell, used for the supercell construction
        - super_multiedge_bundlings (dict):multiedge bundlings in the supercell, used for the supercell construction
        - dv_v_pairs (dict):DV and V pairs in the template, used for the supercell construction
        - super_multiedge_bundlings (dict):multiedge bundlings in the supercell, used for the supercell construction
        - superG (networkx graph):graph of the supercell
        - add_virtual_edge (bool): add virtual edge to the target MOF cell
        - vir_edge_range (float): range to search the virtual edge between two Vnodes directly, should <= 0.5,
        used for the virtual edge addition of bridge type nodes: nodes and nodes can connect directly without linker
        - vir_edge_max_neighbor (int): maximum number of neighbors of the node with virtual edge, used for the virtual edge addition of bridge type nodes
        - remove_node_list (list):list of nodes to remove in the target MOF cell
        - remove_edge_list (list):list of edges to remove in the target MOF cell
        - eG (networkx graph):graph of the target MOF cell with only EDGE and V nodes
        - node_topic (int):maximum degree of the node in the template, should be the same as the node_max_degree
        - unsaturated_node (list):unsaturated nodes in the target MOF cell
        - term_info (array):information of the node terminations
        - term_coords (array):coordinates of the node terminations
        - term_xoovecs (array):X and O vectors (usually carboxylate group) of the node terminations
        - unsaturated_vnode_xind_dict (dict):unsaturated node and the exposed X connected atom index
        - unsaturated_vnode_xoo_dict (dict):unsaturated node and the exposed X connected atom index and the corresponding O connected atoms
    """

    def __init__(self):
        pass

    def analyze_template_multitopic(self, template_cif):
        """
        analyze the template topology of the multitopic linker

        :param vvnode333 (array):
            supercell of V nodes in template topology
        :param ecnode333 (array):
            supercell of EC nodes (Center of multitopic linker) in template topology
        :param eenode333 (array):
            supercell of E nodes(ditopic linker or branch of multitopic linker) in template
        :param unit_cell (array):
            unit cell of the template
        :param cell_info (array):
            cell information of the template
        """
        template_cell_info, _, vvnode = extract_type_atoms_fcoords_in_primitive_cell(
            template_cif, "V"
        )
        _, _, eenode = extract_type_atoms_fcoords_in_primitive_cell(template_cif, "E")
        _, _, ecnode = extract_type_atoms_fcoords_in_primitive_cell(template_cif, "EC")
        unit_cell = extract_unit_cell(template_cell_info)

        vvnode = np.unique(np.array(vvnode, dtype=float), axis=0)
        eenode = np.unique(np.array(eenode, dtype=float), axis=0)
        ecnode = np.unique(np.array(ecnode, dtype=float), axis=0)
        ##loop over super333xxnode and super333yynode to find the pair of x node in unit cell which pass through the yynode
        vvnode333 = make_supercell_3x3x3(vvnode)
        eenode333 = make_supercell_3x3x3(eenode)
        ecnode333 = make_supercell_3x3x3(ecnode)

        _, _, G = find_pair_v_e_c(vvnode333, ecnode333, eenode333, unit_cell)
        G = add_ccoords(G, unit_cell)
        G, self.node_max_degree = set_DV_V(G)
        self.G = set_DE_E(G)
        self.cell_info = template_cell_info
        self.vvnode333 = vvnode333
        self.eenode333 = eenode333
        self.ecnode333 = ecnode333

    def analyze_template_ditopic(self, template_cif, pair_v_e_range=[]):
        """
        analyze the template topology of the ditopic linker, only V and E nodes in the template

        :param template_cif (str):
            cif file of the template, including only V and E nodes info in primitive cell
        """
        template_cell_info, _, vvnode = extract_type_atoms_fcoords_in_primitive_cell(
            template_cif, "V"
        )
        _, _, eenode = extract_type_atoms_fcoords_in_primitive_cell(template_cif, "E")
        unit_cell = extract_unit_cell(template_cell_info)

        vvnode = np.unique(np.array(vvnode, dtype=float), axis=0)
        eenode = np.unique(np.array(eenode, dtype=float), axis=0)
        ##loop over super333xxnode and super333yynode to find the pair of x node in unicell which pass through the yynode
        vvnode333 = make_supercell_3x3x3(vvnode)
        eenode333 = make_supercell_3x3x3(eenode)
        _, _, G = find_pair_v_e(
            vvnode333, eenode333, unit_cell, distance_range=pair_v_e_range
        )
        G = add_ccoords(G, unit_cell)
        G, self.node_max_degree = set_DV_V(G)
        self.G = set_DE_E(G)
        self.cell_info = template_cell_info
        self.vvnode333 = vvnode333
        self.eenode333 = eenode333

    def _node_info(self, node_cif, node_target_type):
        """
        get the node information

        :param node_cif (str):
            cif file of the node
        :param node_target_type (str):
            metal atom type of the node
        """
        self.node_cif = node_cif
        self.node_target_type = node_target_type
        self.node_unit_cell, self.node_atom, self.node_x_fcoords, self.node_fcoords = (
            process_node(node_cif, node_target_type)
        )
        self.node_x_ccoords = np.dot(self.node_unit_cell, self.node_x_fcoords.T).T
        self.node_coords = np.dot(self.node_unit_cell, self.node_fcoords.T).T

    def _linker_info(self, linker_cif):
        """
        get the linker information

        :param linker_cif (str):
            cif file of the ditopic linker or branch of multitopic linker
        """
        self.linker_cif = linker_cif
        (
            self.linker_unit_cell,
            self.linker_atom,
            self.linker_x_fcoords,
            self.linker_fcoords,
        ) = process_node(linker_cif, "X")
        self.linker_x_ccoords = np.dot(self.linker_unit_cell, self.linker_x_fcoords.T).T
        self.linker_length = np.linalg.norm(
            self.linker_x_ccoords[0] - self.linker_x_ccoords[1]
        )
        linker_ccoords = np.dot(self.linker_unit_cell, self.linker_fcoords.T).T
        self.linker_ccoords = linker_ccoords - np.mean(linker_ccoords, axis=0)

    def _linker_center(self, linker_center_cif):
        """
        get the center of the multitopic linker information

        :param linker_center_cif (str):
            cif file of the center of the multitopic linker
        """
        self.linker_center_cif = linker_center_cif
        self.ec_unit_cell, self.ec_atom, self.ec_x_vecs, self.ec_fcoords = process_node(
            self.linker_center_cif, "X"
        )
        self.ec_xcoords = np.dot(self.ec_unit_cell, self.ec_x_vecs.T).T
        self.eccoords = np.dot(self.ec_unit_cell, self.ec_fcoords.T).T

    def node_info(self, node_pdb, com_target_type="X"):
        """
        get the node information

        :param node_cif (str):
            cif file of the node
        :param node_target_type (str):
            metal atom type of the node
        """
        self.node_pdb = node_pdb
        self.node_atom, self.node_ccoords, self.node_x_ccoords = process_node_pdb(
            node_pdb,
            com_target_type,  # TODO: change to the target type X
        )  # com type could be metal in bridge nodes

    def linker_info(self, linker_pdb):
        """
        get the linker information

        :param linker_cif (str):
            cif file of the ditopic linker or branch of multitopic linker
        """
        self.linker_pdb = linker_pdb
        self.linker_atom, self.linker_ccoords, self.linker_x_ccoords = process_node_pdb(
            linker_pdb, "X"
        )
        self.linker_length = np.linalg.norm(
            self.linker_x_ccoords[0] - self.linker_x_ccoords[1]
        )

    def linker_center_info(self, linker_center_pdb):
        """
        get the linker information

        :param linker_cif (str):
            cif file of the ditopic linker or branch of multitopic linker
        """
        self.linker_center_pdb = linker_center_pdb
        self.ec_atom, self.ec_ccoords, self.ec_x_ccoords = process_node_pdb(
            linker_center_pdb, "X"
        )

    def _linker_center(self, linker_center_pdb):
        """
        get the center of the multitopic linker information

        :param linker_center_cif (str):
            cif file of the center of the multitopic linker
        """

        self.linker_center_pdb = linker_center_pdb
        self.ec_atom, self.ec_ccoords, self.ec_x_ccoords = process_node_pdb(
            linker_center_pdb, "X"
        )  # com type could be another

    def set_constant_length(self, constant_length):
        """
        set the constant length to add to the linker length, normally 1.54 (default setting)for single bond of C-C, because C is always used as the connecting atom in the builder
        """
        self.constant_length = constant_length

    def set_rotation_optimizer_maxfun(self, maxfun):
        """
        set the maximum number of function evaluations for the node rotation optimization
        """
        self.maxfun = maxfun

    def set_rotation_optimizer_maxiter(self, maxiter):
        """
        set the maximum number of iterations for the node rotation optimization
        """
        self.maxiter = maxiter

    def set_rotation_optimizer_method(self, opt_method):
        """
        set the optimization method for the node rotation optimization
        """
        self.opt_method = opt_method

    def set_rotation_optimizer_display(self, display):
        """
        set the display of the optimization process
        """
        self.display = display

    def set_rotation_optimizer_eps(self, eps):
        """
        set the eps of the optimization
        """
        self.eps = eps

    def set_rotation_optimizer_iprint(self, iprint):
        """
        set the iprint of the optimization
        """
        self.iprint = iprint

    def check_node_template_match(self):
        """
        precheck, check if the number of nodes in the template matches the maximum degree of the node in the template
        """
        return len(self.node_x_ccoords) == self.node_max_degree

    def load_saved_optimized_rotations(self, optimized_rotations):
        """
        use the saved optimized rotations from the previous optimization
        """
        self.saved_optimized_rotations = optimized_rotations
        print("load the saved optimized_rotations from the previous optimization")

    def to_save_optimized_rotations(self, filename):
        """
        save the optimized rotations to the file
        """
        self.to_save_optimized_rotations_filename = filename

    def use_saved_rotations_as_initial_guess(
        self, use_saved_rotations_as_initial_guess
    ):
        """
        use the saved optimized rotations as initial guess
        """
        self.use_saved_rotations_as_initial_guess = use_saved_rotations_as_initial_guess

    def optimize(self):  # TODO: modified for mil53
        """
        two optimization steps:
        1. optimize the node rotation
        2. optimize the cell parameters to fit the target MOF cell
        """
        if not self.check_node_template_match():
            print(
                "The number of nodes in the template does not match the maximum degree of the node in the template"
            )
            raise ValueError(
                "The number of nodes in the template does not match the maximum degree of the node in the template"
            )

        if hasattr(self, "ec_x_ccoords"):
            ec_x_ccoords = self.ec_x_ccoords
            ecoords = self.ec_ccoords

        if not hasattr(self, "opt_method"):
            self.opt_method = "L-BFGS-B"

        if not hasattr(self, "constant_length"):
            self.constant_length = 1.54

        if not hasattr(self, "maxfun"):
            self.maxfun = 15000

        if not hasattr(self, "maxiter"):
            self.maxiter = 15000

        if not hasattr(self, "display"):
            self.display = True

        if not hasattr(self, "eps"):
            self.eps = 1e-8

        if not hasattr(self, "iprint"):
            self.iprint = -1

        G = self.G
        node_xcoords = self.node_x_ccoords
        node_coords = self.node_ccoords
        linker_length = self.linker_length
        opt_method = self.opt_method

        constant_length = self.constant_length

        x_com_length = np.mean([np.linalg.norm(i) for i in node_xcoords])
        sorted_nodes = sort_nodes_by_type_connectivity(G)

        # firstly, check if all V nodes have highest connectivity
        # secondly, sort all DV nodes by connectivity

        sorted_edges = find_and_sort_edges_bynodeconnectivity(G, sorted_nodes)

        nodes_atoms = {}
        for n in sorted_nodes:
            if "CV" in n:
                nodes_atoms[n] = self.ec_atom

            else:
                nodes_atoms[n] = self.node_atom

        Xatoms_positions_dict = {}
        node_positions_dict = {}
        # reindex the nodes in the Xatoms_positions with the index in the sorted_nodes, like G has 16 nodes[2,5,7], but the new dictionary should be [0,1,2]
        for n in sorted_nodes:
            if "CV" in n:
                Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    G.nodes[n]["ccoords"] + ec_x_ccoords
                )
            else:
                Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    G.nodes[n]["ccoords"] + node_xcoords
                )

        for n in sorted_nodes:
            if "CV" in n:
                node_positions_dict[sorted_nodes.index(n)] = (
                    G.nodes[n]["ccoords"] + ecoords
                )
            else:
                node_positions_dict[sorted_nodes.index(n)] = (
                    G.nodes[n]["ccoords"] + node_coords
                )

        # reindex the edges in the G with the index in the sorted_nodes
        sorted_edges_of_sortednodeidx = [
            (sorted_nodes.index(e[0]), sorted_nodes.index(e[1])) for e in sorted_edges
        ]

        # Optimize rotations
        num_nodes = G.number_of_nodes()
        pname_list = [pname(n) for n in sorted_nodes]
        pname_set = set(pname_list)
        pname_set_dict = {}
        for node_pname in pname_set:
            pname_set_dict[node_pname] = {
                "ind_ofsortednodes": [],
            }
        for i, node in enumerate(sorted_nodes):
            pname_set_dict[pname(node)]["ind_ofsortednodes"].append(i)
            if len(pname_set_dict[pname(node)]["ind_ofsortednodes"]) == 1:  # first node
                pname_set_dict[pname(node)]["rot_trans"] = get_rot_trans_matrix(
                    node, G, sorted_nodes, Xatoms_positions_dict
                )  # initial guess
        self.pname_set_dict = pname_set_dict

        for p_name in pname_set_dict:
            rot, trans = pname_set_dict[p_name]["rot_trans"]
            for k in pname_set_dict[p_name]["ind_ofsortednodes"]:
                node = sorted_nodes[k]

                Xatoms_positions_dict[k][:, 1:] = (
                    np.dot(
                        Xatoms_positions_dict[k][:, 1:] - G.nodes[node]["ccoords"],
                        rot,
                    )
                    + trans
                    + G.nodes[node]["ccoords"]
                )
                node_positions_dict[k] = (
                    np.dot(node_positions_dict[k] - G.nodes[node]["ccoords"], rot)
                    + trans
                    + G.nodes[node]["ccoords"]
                )
        ###3D free rotation
        if not hasattr(self, "saved_optimized_rotations"):
            print("-" * 80)
            print(" " * 20, "start to optimize the rotations", " " * 20)
            print("-" * 80)

            ##initial_rots = []
            ##
            ##for node in sorted_nodes:
            ##    rot = get_rotation_matrix(node, G, sorted_nodes, Xatoms_positions_dict)
            ##    # print(rot)
            ##    initial_rots.append(rot)
            ##initial_guess_rotations = np.array(
            ##    initial_rots
            ##).flatten()  # Initial guess for rotation matrices

            initial_guess_set_rotations = (
                np.eye(3, 3).reshape(1, 3, 3).repeat(len(pname_set), axis=0)
            )

            ####TODO: modified for mil53
            (
                optimized_rotations_pre,
                _,
            ) = optimize_rotations_pre(
                num_nodes,
                G,
                sorted_nodes,
                sorted_edges_of_sortednodeidx,
                Xatoms_positions_dict,
                initial_guess_set_rotations,
                pname_set_dict,
                opt_method=self.opt_method,
                maxfun=self.maxfun,
                maxiter=self.maxiter,
                disp=self.display,
                eps=self.eps,
                iprint=self.iprint,
            )

            (
                optimized_set_rotations,
                _,
            ) = optimize_rotations_after(
                num_nodes,
                G,
                sorted_nodes,
                sorted_edges_of_sortednodeidx,
                Xatoms_positions_dict,
                # initial_guess_set_rotations,  # TODO: modified for mil53
                optimized_rotations_pre,
                pname_set_dict,
                opt_method=self.opt_method,
                maxfun=self.maxfun,
                maxiter=self.maxiter,
                disp=self.display,
                eps=self.eps,
                iprint=self.iprint,
            )
            print("-" * 80)
            print(" " * 20, "rotations optimization completed", " " * 20)
            print("-" * 80)
            # to save the optimized rotations as npy
            if hasattr(self, "to_save_optimized_rotations_filename"):
                np.save(
                    self.to_save_optimized_rotations_filename + ".npy",
                    optimized_set_rotations,
                )
                print(
                    "optimized rotations are saved to: ",
                    self.to_save_optimized_rotations_filename + ".npy",
                )

        else:
            if hasattr(self, "use_saved_rotations_as_initial_guess"):
                if self.use_saved_rotations_as_initial_guess:
                    print("use the saved optimized_rotations as initial guess")
                    print("-" * 80)
                    print(" " * 20, "start to optimize the rotations", " " * 20)
                    print("-" * 80)

                    saved_set_rotations = self.saved_optimized_rotations.reshape(
                        -1, 3, 3
                    )
                    # (
                    #    optimized_rotations_pre,
                    #    _,
                    # ) = optimize_rotations_pre(
                    #    num_nodes,
                    #    G,
                    #    sorted_nodes,
                    #    sorted_edges_of_sortednodeidx,
                    #    Xatoms_positions_dict,
                    #    saved_set_rotations,
                    #    pname_set_dict,
                    #    opt_method=self.opt_method,
                    #    maxfun=self.maxfun,
                    #    maxiter=self.maxiter,
                    #    disp=self.display,
                    #    eps=self.eps,
                    #    iprint=self.iprint,
                    # )

                    (
                        optimized_set_rotations,
                        _,
                    ) = optimize_rotations_after(
                        num_nodes,
                        G,
                        sorted_nodes,
                        sorted_edges_of_sortednodeidx,
                        Xatoms_positions_dict,
                        saved_set_rotations,
                        pname_set_dict,
                        opt_method=self.opt_method,
                        maxfun=self.maxfun,
                        maxiter=self.maxiter,
                        disp=self.display,
                        eps=self.eps,
                        iprint=self.iprint,
                    )
                    print("-" * 80)
                    print(" " * 20, "rotations optimization completed", " " * 20)
                    print("-" * 80)
                    # to save the optimized rotations as npy
                    if hasattr(self, "to_save_optimized_rotations_filename"):
                        np.save(
                            self.to_save_optimized_rotations_filename + ".npy",
                            optimized_set_rotations,
                        )
                        print(
                            "optimized rotations are saved to: ",
                            self.to_save_optimized_rotations_filename + ".npy",
                        )

                else:
                    optimized_set_rotations = self.saved_optimized_rotations.reshape(
                        -1, 3, 3
                    )

            else:
                print(
                    "use the loaded optimized_rotations from the previous optimization"
                )
                optimized_set_rotations = self.saved_optimized_rotations.reshape(
                    -1, 3, 3
                )
        from _node_rotation_matrix_optimizer import expand_setrots

        optimized_rotations = expand_setrots(
            pname_set_dict, optimized_set_rotations, sorted_nodes
        )
        # Apply rotations
        rotated_node_positions = apply_rotations_to_atom_positions(
            optimized_rotations, G, sorted_nodes, node_positions_dict
        )

        # Save results to XYZ
        # save_xyz("optimized_nodesstructure.xyz", rotated_node_positions) #DEBUG

        rotated_Xatoms_positions_dict, optimized_pair = (
            apply_rotations_to_Xatoms_positions(
                optimized_rotations,
                G,
                sorted_nodes,
                sorted_edges_of_sortednodeidx,
                Xatoms_positions_dict,
            )
        )

        start_node = sorted_edges[0][0]  # find_nearest_node_to_beginning_point(G)
        # loop all of the edges in G and get the lengths of the edges, length is the distance between the two nodes ccoords
        edge_lengths, lengths = get_edge_lengths(G)

        x_com_length = np.mean([np.linalg.norm(i) for i in node_xcoords])
        new_edge_length = linker_length + 2 * constant_length + 2 * x_com_length
        # update the node ccoords in G by loop edge, start from the start_node, and then update the connected node ccoords by the edge length, and update the next node ccords from the updated node

        updated_ccoords, original_ccoords = update_node_ccoords(
            G, edge_lengths, start_node, new_edge_length
        )
        # exclude the start_node in updated_ccoords and original_ccoords
        updated_ccoords = {k: v for k, v in updated_ccoords.items() if k != start_node}
        original_ccoords = {
            k: v for k, v in original_ccoords.items() if k != start_node
        }

        # use optimized_params to update all of nodes ccoords in G, according to the fccoords
        if not hasattr(self, "optimized_params"):
            print("-" * 80)
            print(" " * 20, "start to optimize the cell parameters", " " * 20)
            print("-" * 80)
            optimized_params = optimize_cell_parameters(
                self.cell_info, original_ccoords, updated_ccoords
            )
            print("-" * 80)
            print(" " * 20, "cell parameters optimization completed", " " * 20)
            print("-" * 80)
        else:
            print("use the optimized_params from the previous optimization")
            optimized_params = self.optimized_params

        sc_unit_cell = unit_cell_to_cartesian_matrix(
            optimized_params[0],
            optimized_params[1],
            optimized_params[2],
            optimized_params[3],
            optimized_params[4],
            optimized_params[5],
        )
        sc_unit_cell_inv = np.linalg.inv(sc_unit_cell)
        sG, scaled_ccoords = update_ccoords_by_optimized_cell_params(
            self.G, optimized_params
        )
        scaled_node_positions_dict = {}
        scaled_Xatoms_positions_dict = {}

        for n in sorted_nodes:
            if "CV" in n:
                scaled_Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    sG.nodes[n]["ccoords"] + ec_x_ccoords
                )
            else:
                scaled_Xatoms_positions_dict[sorted_nodes.index(n)] = addidx(
                    sG.nodes[n]["ccoords"] + node_xcoords
                )

        for n in sorted_nodes:
            if "CV" in n:
                scaled_node_positions_dict[sorted_nodes.index(n)] = (
                    sG.nodes[n]["ccoords"] + ecoords
                )
            else:
                scaled_node_positions_dict[sorted_nodes.index(n)] = (
                    sG.nodes[n]["ccoords"] + node_coords
                )

        # Apply rotations
        for p_name in pname_set_dict:
            rot, trans = pname_set_dict[p_name]["rot_trans"]
            for k in pname_set_dict[p_name]["ind_ofsortednodes"]:
                node = sorted_nodes[k]
                scaled_Xatoms_positions_dict[k][:, 1:] = (
                    np.dot(
                        scaled_Xatoms_positions_dict[k][:, 1:]
                        - sG.nodes[node]["ccoords"],
                        rot,
                    )
                    + trans
                    + sG.nodes[node]["ccoords"]
                )

                scaled_node_positions_dict[k] = (
                    np.dot(
                        scaled_node_positions_dict[k] - sG.nodes[node]["ccoords"], rot
                    )
                    + trans
                    + sG.nodes[node]["ccoords"]
                )

        scaled_rotated_node_positions = apply_rotations_to_atom_positions(
            optimized_rotations, sG, sorted_nodes, scaled_node_positions_dict
        )
        scaled_rotated_Xatoms_positions, optimized_pair = (
            apply_rotations_to_Xatoms_positions(
                optimized_rotations,
                sG,
                sorted_nodes,
                sorted_edges_of_sortednodeidx,
                scaled_Xatoms_positions_dict,
            )
        )
        # Save results to XYZ

        self.sorted_nodes = sorted_nodes
        self.sorted_edges = sorted_edges
        self.sorted_edges_of_sortednodeidx = sorted_edges_of_sortednodeidx
        self.optimized_rotations = optimized_rotations
        self.optimized_params = optimized_params
        self.new_edge_length = new_edge_length
        self.optimized_pair = optimized_pair
        self.scaled_rotated_node_positions = scaled_rotated_node_positions
        self.scaled_rotated_Xatoms_positions = scaled_rotated_Xatoms_positions
        self.sc_unit_cell = sc_unit_cell
        self.sc_unit_cell_inv = sc_unit_cell_inv
        self.sG_node = sG
        self.nodes_atom = nodes_atoms
        self.rotated_node_positions = rotated_node_positions
        self.Xatoms_positions_dict = Xatoms_positions_dict
        self.node_positions_dict = node_positions_dict
        # save_xyz("scale_optimized_nodesstructure.xyz", scaled_rotated_node_positions)

    def place_edge_in_net(self):
        """
        based on the optimized rotations and cell parameters, use optimized pair to find connected X-X pair in optimized cell,
        and place the edge in the target MOF cell

        return:
            sG (networkx graph):graph of the target MOF cell, with scaled and rotated node and edge positions
        """
        # linker_middle_point = np.mean(linker_x_vecs,axis=0)
        linker_xx_vec = self.linker_x_ccoords
        linker_length = self.linker_length
        optimized_pair = self.optimized_pair
        scaled_rotated_Xatoms_positions = self.scaled_rotated_Xatoms_positions
        scaled_rotated_node_positions = self.scaled_rotated_node_positions
        sorted_nodes = self.sorted_nodes
        sG_node = self.sG_node
        sc_unit_cell_inv = self.sc_unit_cell_inv
        nodes_atom = self.nodes_atom

        sG = sG_node.copy()
        scalar = (linker_length + 2 * self.constant_length) / linker_length
        extended_linker_xx_vec = [i * scalar for i in linker_xx_vec]
        norm_xx_vector_record = []
        rot_record = []

        # edges = {}
        for (i, j), pair in optimized_pair.items():
            x_idx_i, x_idx_j = pair
            reindex_i = sorted_nodes.index(i)
            reindex_j = sorted_nodes.index(j)
            x_i = scaled_rotated_Xatoms_positions[reindex_i][x_idx_i][1:]
            x_j = scaled_rotated_Xatoms_positions[reindex_j][x_idx_j][1:]
            x_i_x_j_middle_point = np.mean([x_i, x_j], axis=0)
            xx_vector = np.vstack(
                [x_i - x_i_x_j_middle_point, x_j - x_i_x_j_middle_point]
            )
            norm_xx_vector = xx_vector / np.linalg.norm(xx_vector)

            # print(i, j, reindex_i, reindex_j, x_idx_i, x_idx_j)
            # use superimpose to get the rotation matrix
            # use record to record the rotation matrix for get rid of the repeat calculation
            indices = [
                index
                for index, value in enumerate(norm_xx_vector_record)
                if is_list_A_in_B(norm_xx_vector, value)
            ]
            if len(indices) == 1:
                rot = rot_record[indices[0]]
                # rot = reorthogonalize_matrix(rot)
            else:
                _, rot, _ = superimpose_rotation_only(extended_linker_xx_vec, xx_vector)
                # rot = reorthogonalize_matrix(rot)
                norm_xx_vector_record.append(norm_xx_vector)
                # the rot may be opposite, so we need to check the angle between the two vectors
                # if the angle is larger than 90 degree, we need to reverse the rot
                roted_xx = np.dot(extended_linker_xx_vec, rot)

                if np.dot(roted_xx[1] - roted_xx[0], xx_vector[1] - xx_vector[0]) < 0:
                    ##rotate 180 around the axis of the cross product of the two vectors
                    axis = np.cross(
                        roted_xx[1] - roted_xx[0], xx_vector[1] - xx_vector[0]
                    )
                    # if 001 not linear to the two vectors
                    if np.linalg.norm(axis) == 0:
                        check_z_axis = np.cross(roted_xx[1] - roted_xx[0], [0, 0, 1])
                        if np.linalg.norm(check_z_axis) == 0:
                            axis = np.array([1, 0, 0])
                        else:
                            axis = np.array([0, 0, 1])

                    axis = axis / np.linalg.norm(axis)
                    flip_matrix = R.from_rotvec(np.pi * np.array(axis)).as_matrix()

                    rot = np.dot(rot, flip_matrix)
                # Flip the last column of the rotation matrix if the determinant is negative

                rot_record.append(rot)

            # use the rotation matrix to rotate the linker x coords
            # rotated_xx = np.dot(extended_linker_xx_vec, rot)
            # print(rotated_xx,'rotated_xx',xx_vector) #DEBUG
            placed_edge_ccoords = (
                np.dot(self.linker_ccoords, rot) + x_i_x_j_middle_point
            )

            placed_edge = np.hstack((np.asarray(self.linker_atom), placed_edge_ccoords))
            sG.edges[(i, j)]["coords"] = x_i_x_j_middle_point
            sG.edges[(i, j)]["c_points"] = placed_edge

            sG.edges[(i, j)]["f_points"] = np.hstack(
                (
                    placed_edge[:, 0:2],
                    cartesian_to_fractional(placed_edge[:, 2:5], sc_unit_cell_inv),
                )
            )  # NOTE: modified add the atom type and atom name

            _, sG.edges[(i, j)]["x_coords"] = fetch_X_atoms_ind_array(
                placed_edge, 0, "X"
            )
            # edges[(i,j)]=placed_edge
        # placed_node = {}
        for k, v in scaled_rotated_node_positions.items():
            # print(k,v)
            # placed_node[k] = np.hstack((nodes_atom[k],v))
            sG.nodes[k]["c_points"] = np.hstack((nodes_atom[k], v))
            sG.nodes[k]["f_points"] = np.hstack(
                (nodes_atom[k], cartesian_to_fractional(v, sc_unit_cell_inv))
            )
            # find the atoms starts with "x" and extract the coordinates
            _, sG.nodes[k]["x_coords"] = fetch_X_atoms_ind_array(
                sG.nodes[k]["c_points"], 0, "X"
            )
        self.sG = sG
        return sG

    def set_supercell(self, supercell):
        """
        set the supercell of the target MOF model
        """
        self.supercell = supercell

    def make_supercell_multitopic(self):
        """
        make the supercell of the multitopic linker MOF
        """
        sG = self.sG
        self.multiedge_bundlings = bundle_multiedge(sG)
        # self.dv_v_pairs, sG = replace_DV_with_corresponding_V(sG) #debug
        superG = update_supercell_node_fpoints_loose(sG, self.supercell)
        superG = update_supercell_edge_fpoints(sG, superG, self.supercell)
        # self.prim_multiedge_bundlings = replace_bundle_dvnode_with_vnode(  #debug
        #    self.dv_v_pairs, self.multiedge_bundlings
        # )
        self.prim_multiedge_bundlings = self.multiedge_bundlings
        self.super_multiedge_bundlings = make_super_multiedge_bundlings(
            self.prim_multiedge_bundlings, self.supercell
        )
        superG = update_supercell_bundle(superG, self.super_multiedge_bundlings)
        superG = check_multiedge_bundlings_insuperG(
            self.super_multiedge_bundlings, superG
        )
        self.superG = superG
        return superG

    def make_supercell_ditopic(self):
        """
        make the supercell of the ditopic linker MOF
        """

        sG = self.sG
        # self.dv_v_pairs, sG = replace_DV_with_corresponding_V(sG)
        superG = update_supercell_node_fpoints_loose(sG, self.supercell)
        superG = update_supercell_edge_fpoints(sG, superG, self.supercell)
        self.superG = superG
        return superG

    def set_virtual_edge(self, bool_x=False, range=0.5, max_neighbor=2):
        """
        set the virtual edge addition for the bridge type nodes,
        range is the range to search the virtual edge between two Vnodes directly, should <= 0.5,
        max_neighbor is the maximum number of neighbors of the node with virtual edge
        """

        self.add_virtual_edge = bool(bool_x)
        self.vir_edge_range = range
        self.vir_edge_max_neighbor = max_neighbor

    def add_virtual_edge_for_bridge_node(self, superG):
        """
        after setting the virtual edge search, add the virtual edge to the target supercell superG MOF
        """
        if self.add_virtual_edge:
            add_superG = add_virtual_edge(
                self.sc_unit_cell,
                superG,
                self.vir_edge_range,
                self.vir_edge_max_neighbor,
            )
            print("add virtual edge")
            return add_superG
        else:
            return superG

    def set_remove_node_list(self, remove_node_list):
        """
        make defect in the target MOF model by removing nodes
        """
        self.remove_node_list = remove_node_list

    def set_remove_edge_list(self, remove_edge_list):
        """
        make defect in the target MOF model by removing edges
        """
        self.remove_edge_list = remove_edge_list

    def make_eG_from_supereG_multitopic(self):
        """
        make the target MOF cell graph with only EDGE and V, link the XOO atoms to the EDGE
        always need to execute with make_supercell_multitopic
        """

        eG, _ = superG_to_eG_multitopic(self.superG, self.sc_unit_cell)
        self.eG = eG
        return eG

    def add_xoo_to_edge_multitopic(self):
        eG = self.eG
        eG, unsaturated_linker, matched_vnode_xind, xoo_dict = addxoo2edge_multitopic(
            eG, self.sc_unit_cell
        )
        self.unsaturated_linker = unsaturated_linker
        self.matched_vnode_xind = matched_vnode_xind
        self.xoo_dict = xoo_dict
        self.eG = eG
        return eG

    def make_eG_from_supereG_ditopic(self):
        """
        make the target MOF cell graph with only EDGE and V, link the XOO atoms to the EDGE
        always execute with make_supercell_ditopic
        """

        eG, _ = superG_to_eG_ditopic(self.superG)
        self.eG = eG
        return eG

    def add_xoo_to_edge_ditopic(self):
        """
        analyze eG and link the XOO atoms to the EDGE, update eG, for ditopic linker MOF
        """
        eG = self.eG
        eG, unsaturated_linker, matched_vnode_xind, xoo_dict = addxoo2edge_ditopic(
            eG, self.sc_unit_cell
        )
        self.unsaturated_linker = unsaturated_linker
        self.matched_vnode_xind = matched_vnode_xind
        self.xoo_dict = xoo_dict
        self.eG = eG
        return eG

    def main_frag_eG(self):
        """
        only keep the main fragment of the target MOF cell, remove the other fragments, to avoid the disconnected fragments
        """
        eG = self.eG
        self.eG = [eG.subgraph(c).copy() for c in nx.connected_components(eG)][0]
        print("main fragment of the MOF cell is kept")  # ,len(self.eG.nodes()),'nodes')
        # print('fragment size list:',[len(c) for c in nx.connected_components(eG)]) #debug
        return self.eG

    def make_supercell_range_cleaved_eG(self, buffer_plus=0, buffer_minus=0):
        supercell = self.supercell
        new_eG = self.eG.copy()
        eG = self.eG
        removed_edges = []
        removed_nodes = []
        for n in eG.nodes():
            if pname(n) != "EDGE":
                if check_supercell_box_range(
                    eG.nodes[n]["fcoords"], supercell, buffer_plus, buffer_minus
                ):
                    pass
                else:
                    new_eG.remove_node(n)
                    removed_nodes.append(n)
            elif pname(n) == "EDGE":
                if (
                    arr_dimension(eG.nodes[n]["fcoords"]) == 2
                ):  # ditopic linker have two points in the fcoords
                    edge_coords = np.mean(eG.nodes[n]["fcoords"], axis=0)
                elif (
                    arr_dimension(eG.nodes[n]["fcoords"]) == 1
                ):  # multitopic linker have one point in the fcoords from EC
                    edge_coords = eG.nodes[n]["fcoords"]

                if check_supercell_box_range(
                    edge_coords, supercell, buffer_plus, buffer_minus
                ):
                    pass
                else:
                    new_eG.remove_node(n)
                    removed_edges.append(n)

        matched_vnode_xind = self.matched_vnode_xind
        self.matched_vnode_xind = update_matched_nodes_xind(
            removed_nodes,
            removed_edges,
            matched_vnode_xind,
        )

        self.eG = new_eG
        return new_eG, removed_edges, removed_nodes

    def set_node_topic(self, node_topic):
        """
        manually set the node topic, normally should be the same as the maximum degree of the node in the template
        """
        self.node_topic = node_topic

    def find_unsaturated_node_eG(self):
        """
        use the eG to find the unsaturated nodes, whose degree is less than the node topic
        """
        eG = self.eG
        if hasattr(self, "node_topic"):
            node_topic = self.node_topic
        else:
            node_topic = self.node_max_degree
        unsaturated_node = find_unsaturated_node(eG, node_topic)
        self.unsaturated_node = unsaturated_node
        return unsaturated_node

    def find_unsaturated_linker_eG(eG, linker_topics):
        """
        use the eG to find the unsaturated linkers, whose degree is less than linker topic
        """
        new_unsaturated_linker = find_unsaturated_linker(eG, linker_topics)
        return new_unsaturated_linker

    def set_node_terminamtion(self, term_file):
        """
        pdb file, set the node termination file, which contains the information of the node terminations, should have X of connected atom (normally C),
        Y of two connected O atoms (if in carboxylate group) to assist the placement of the node terminations
        """

        term_data = termpdb(term_file)
        term_info = term_data[:, :-3]
        term_coords = term_data[:, -3:]
        xterm, _ = Xpdb(term_data, "X")
        oterm, _ = Xpdb(term_data, "Y")
        term_xvecs = xterm[:, -3:]
        term_ovecs = oterm[:, -3:]
        term_coords = term_coords.astype("float")
        term_xvecs = term_xvecs.astype("float")
        term_ovecs = term_ovecs.astype("float")

        term_ovecs_c = np.mean(np.asarray(term_ovecs), axis=0)
        term_coords = term_coords - term_ovecs_c
        term_xoovecs = np.vstack((term_xvecs, term_ovecs))
        term_xoovecs = term_xoovecs - term_ovecs_c
        self.node_termination = term_file
        self.term_info = term_info
        self.term_coords = term_coords
        self.term_xoovecs = term_xoovecs

    # Function to add node terminations
    def add_terminations_to_unsaturated_node(self):
        """
        use the node terminations to add terminations to the unsaturated nodes

        """
        unsaturated_node = [n for n in self.unsaturated_node if n in self.eG.nodes()]
        xoo_dict = self.xoo_dict
        matched_vnode_xind = self.matched_vnode_xind
        eG = self.eG
        sc_unit_cell = self.sc_unit_cell
        (
            unsaturated_vnode_xind_dict,
            unsaturated_vnode_xoo_dict,
            self.matched_vnode_xind_dict,
        ) = make_unsaturated_vnode_xoo_dict(
            unsaturated_node, xoo_dict, matched_vnode_xind, eG, sc_unit_cell
        )
        # term_file: path to the termination file
        # ex_node_cxo_cc: exposed node coordinates

        node_oovecs_record = []
        for n in eG.nodes():
            eG.nodes[n]["term_c_points"] = {}
        for exvnode_xind_key in unsaturated_vnode_xoo_dict.keys():
            exvnode_x_ccoords = unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "x_cpoints"
            ]
            exvnode_oo_ccoords = unsaturated_vnode_xoo_dict[exvnode_xind_key][
                "oo_cpoints"
            ]
            node_xoo_ccoords = np.vstack([exvnode_x_ccoords, exvnode_oo_ccoords])
            # make the beginning point of the termination to the center of the oo atoms
            node_oo_center_cvec = np.mean(
                exvnode_oo_ccoords[:, 2:5].astype(float), axis=0
            )  # NOTE: modified add the atom type and atom name
            node_xoo_cvecs = (
                node_xoo_ccoords[:, 2:5].astype(float) - node_oo_center_cvec
            )  # NOTE: modified add the atom type and atom name
            node_xoo_cvecs = node_xoo_cvecs.astype("float")
            # use record to record the rotation matrix for get rid of the repeat calculation

            indices = [
                index
                for index, value in enumerate(node_oovecs_record)
                if is_list_A_in_B(node_xoo_cvecs, value[0])
            ]
            if len(indices) == 1:
                rot = node_oovecs_record[indices[0]][1]
            else:
                _, rot, _ = superimpose(self.term_xoovecs, node_xoo_cvecs)
                node_oovecs_record.append((node_xoo_cvecs, rot))
            adjusted_term_vecs = np.dot(self.term_coords, rot) + node_oo_center_cvec
            adjusted_term = np.hstack(
                (
                    np.asarray(self.term_info[:, 0:1]),
                    np.asarray(self.term_info[:, 2:3]),
                    adjusted_term_vecs,
                )
            )
            # add the adjusted term to the terms, add index, add the node name
            unsaturated_vnode_xoo_dict[exvnode_xind_key]["node_term_c_points"] = (
                adjusted_term
            )
            eG.nodes[exvnode_xind_key[0]]["term_c_points"][exvnode_xind_key[1]] = (
                adjusted_term
            )

        self.unsaturated_vnode_xoo_dict = unsaturated_vnode_xoo_dict
        self.eG = eG
        return eG

    def remove_xoo_from_node(self):
        """
        remove the XOO atoms from the node after adding the terminations, add ['noxoo_f_points'] to the node in eG
        """
        eG = self.eG
        xoo_dict = self.xoo_dict

        all_xoo_indices = []
        for x_ind, oo_ind in xoo_dict.items():
            all_xoo_indices.append(x_ind)
            all_xoo_indices.extend(oo_ind)

        for n in eG.nodes():
            if pname(n) != "EDGE":
                all_f_points = eG.nodes[n]["f_points"]
                noxoo_f_points = np.delete(all_f_points, all_xoo_indices, axis=0)
                eG.nodes[n]["noxoo_f_points"] = noxoo_f_points
        self.eG = eG

        return eG

    def write_node_edge_node_gro(self, gro_name):
        """
        write the node, edge, node to the gro file
        """

        nodes_eG, edges_eG, terms_eG, node_res_num, edge_res_num, term_res_num = (
            extract_node_edge_term(self.eG, self.sc_unit_cell)
        )
        merged_node_edge_term = merge_node_edge_term(
            nodes_eG, edges_eG, terms_eG, node_res_num, edge_res_num
        )
        save_node_edge_term_gro(merged_node_edge_term, gro_name)
        print(str(gro_name) + ".gro is saved")
        print("node_res_num: ", node_res_num)
        print("edge_res_num: ", edge_res_num)
        print("term_res_num: ", term_res_num)

        self.nodes_eG = nodes_eG
        self.edges_eG = edges_eG
        self.terms_eG = terms_eG
        self.node_res_num = node_res_num
        self.edge_res_num = edge_res_num
        self.term_res_num = term_res_num
        self.merged_node_edge_term = merged_node_edge_term


if __name__ == "__main__":
    start_time = time.time()
    linker_pdb = "edges/diedge.pdb"
    # in database by calling MOF family name and node metal type, dummy node True or False
    template_cif = "fcu.cif"
    node_pdb = "data/nodes_database/12c_Zr.pdb"
    node_target_type = "Zr"

    fcu = net_optimizer()
    fcu.analyze_template_ditopic(template_cif)
    fcu.node_info(node_pdb)
    fcu.linker_info(linker_pdb)
    fcu.optimize()
    fcu.set_supercell([1, 1, 1])
    print("--- %s seconds ---" % (time.time() - start_time))
    fcu.place_edge_in_net()
    fcu.make_supercell_ditopic()
    fcu.make_eG_from_supereG_ditopic()
    fcu.main_frag_eG()
    fcu.make_supercell_range_cleaved_eG()
    fcu.add_xoo_to_edge_ditopic()
    fcu.find_unsaturated_node_eG()
    fcu.set_node_terminamtion("methyl.pdb")
    fcu.add_terminations_to_unsaturated_node()
    fcu.remove_xoo_from_node()
    fcu.write_node_edge_node_gro("fcu_ditopic")
    print("--- %s seconds ---" % (time.time() - start_time))
    # temp_save_eGterm_gro(fcu.eG,fcu.sc_unit_cell) #debug
