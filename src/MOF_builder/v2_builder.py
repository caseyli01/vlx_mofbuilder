import re
import time
import numpy as np
import networkx as nx
from _readcif import process_node,read_cif,extract_type_atoms_fcoords_in_primitive_cell
from _node_rotation_matrix_optimizer import optimize_rotations,apply_rotations_to_atom_positions,apply_rotations_to_xxxx_positions,update_ccoords_by_optimized_cell_params
from _scale_cif_optimizer import optimize_cell_parameters
from _place_node_edge import addidx,get_edge_lengths,update_node_ccoords,unit_cell_to_cartesian_matrix,fractional_to_cartesian,cartesian_to_fractional
from _superimpose import superimpose
from v2_functions import temp_save_eG_TERM_gro,fetch_X_atoms_ind_array,find_pair_v_e_c,sort_nodes_by_type_connectivity,find_and_sort_edges_bynodeconnectivity,is_list_A_in_B
from make_eG import superG_to_eG_ditopic,superG_to_eG_multitopic,remove_node_by_index,addxoo2edge_ditopic,addxoo2edge_multitopic,find_unsaturated_node
from multiedge_bundling import bundle_multiedge
from makesuperG import pname,replace_DV_with_corresponding_V,replace_bundle_dvnode_with_vnode,make_super_multiedge_bundlings,check_multiedge_bundlings_insuperG,add_virtual_edge,update_supercell_bundle,update_supercell_edge_fpoints,update_supercell_node_fpoints_loose
from _terminations import termpdb,Xpdb
from v2_functions import make_unsaturated_vnode_xoo_dict
#the following are for test need to be removed

from _learn_template import make_supercell_3x3x3,find_pair_v_e,extract_unit_cell
from _learn_template import add_ccoords,set_DV_V,set_DE_E




class net_optimizer():
    """
    net_optimizer is a class to optimize the node and edge structure of the MOF, add terminations to nodes.

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
        - constant_length (float):constant length to add to the linker length, normally 1.54 for single bond of C-C, because C is always used as the connecting atom in the builder
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
        - scaled_rotated_xxxx_positions (dict):scaled and rotated X connected atom positions of nodes in the target MOF cell
        - sc_unit_cell (array):(scaled) unit cell of the target MOF cell
        - sc_unit_cell_inv (array):inverse of the (scaled) unit cell of the target MOF cell
        - sG_node (networkx graph):graph of the target MOF cell
        - nodes_atom (dict):atom name and atom type of the nodes 
        - rotated_node_positions (dict):rotated node positions
        - supercell (array): supercell set by user, along x,y,z direction
        - multiedge_bundlings (dict):multiedge bundlings of center and branches of the multitopic linker, used for the supercell construction and merging of center and branches to form one EDGE
        - prim_multiedge_bundlings (dict):multiedge bundlings in primitive cell, used for the supercell construction
        - super_multiedge_bundlings (dict):multiedge bundlings in the supercell, used for the supercell construction
        - dv_v_pairs (dict):DV and V pairs in the template, used for the supercell construction
        - super_multiedge_bundlings (dict):multiedge bundlings in the supercell, used for the supercell construction
        - superG (networkx graph):graph of the supercell
        - add_virtual_edge (bool): add virtual edge to the target MOF cell
        - vir_edge_range (float): range to search the virtual edge between two Vnodes directly, should <= 0.5, used for the virtual edge addition of bridge type nodes: nodes and nodes can connect directly without linker
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

    
    def analyze_template_multitopic(self,vvnode333,ecnode333,eenode333,unit_cell,cell_info):
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
        self.vvnode333 = vvnode333
        self.ecnode333 = ecnode333
        self.eenode333 = eenode333
        _,_,G = find_pair_v_e_c(vvnode333,ecnode333,eenode333)
        G = add_ccoords(G,unit_cell)
        G = set_DV_V(G)
        self.G = set_DE_E(G)
        self.cell_info = cell_info
    
    def analyze_template_ditopic(self,vvnode333,eenode333,unit_cell,cell_info):
        """
        analyze the template topology of the ditopic linker
        
        :param vvnode333 (array):
            supercell of V nodes in template topology
        :param eenode333 (array):
            supercell of E nodes(ditopic linker or branch of multitopic linker) in template
        :param unit_cell (array):
            unit cell of the template
        :param cell_info (array):
            cell information of the template
        """
        self.vvnode333 = vvnode333
        self.eenode333 = eenode333
        _,_,G = find_pair_v_e(vvnode333,eenode333)
        G = add_ccoords(G,unit_cell)
        G,self.node_max_degree = set_DV_V(G)
        self.G = set_DE_E(G)
        self.cell_info = cell_info


    def node_info(self,node_cif,node_target_type):
        """
        get the node information

        :param node_cif (str):
            cif file of the node
        :param node_target_type (str):
            metal atom type of the node
        """
        self.node_cif = node_cif
        self.node_target_type = node_target_type
        self.node_unit_cell,self.node_atom, self.node_x_fcoords, self.node_fcoords = process_node(node_cif, node_target_type)
        self.node_x_ccoords = np.dot(self.node_unit_cell,self.node_x_fcoords.T).T
        self.node_coords = np.dot(self.node_unit_cell,self.node_fcoords.T).T

    def linker_info(self,linker_cif):
        """
        get the linker information
        
        :param linker_cif (str):
            cif file of the ditopic linker or branch of multitopic linker    
        """
        self.linker_cif = linker_cif
        self.linker_unit_cell,self.linker_atom, self.linker_x_fcoords, self.linker_fcoords = process_node(linker_cif, 'X')
        self.linker_x_ccoords = np.dot(self.linker_unit_cell,self.linker_x_fcoords.T).T
        self.linker_length = np.linalg.norm(self.linker_x_ccoords[0]-self.linker_x_ccoords[1])
        linker_ccoords = np.dot(self.linker_unit_cell,self.linker_fcoords.T).T
        self.linker_ccoords = linker_ccoords - np.mean(linker_ccoords,axis=0)

    
    def linker_center(self,linker_center_cif):
        """
        get the center of the multitopic linker information

        :param linker_center_cif (str):
            cif file of the center of the multitopic linker
        """
        self.linker_center_cif = linker_center_cif
        self.ec_unit_cell,self.ec_atom, self.ec_x_vecs, self.ec_fcoords = process_node(self.linker_center_cif, 'X')
        self.ec_xcoords = np.dot(self.ec_unit_cell,self.ec_x_vecs.T).T
        self.eccoords = np.dot(self.ec_unit_cell,self.ec_fcoords.T).T

    def set_constant_length(self,constant_length):
        """
        set the constant length to add to the linker length, normally 1.54 (default setting)for single bond of C-C, because C is always used as the connecting atom in the builder
        """
        self.constant_length = constant_length
    
    def set_maxfun(self,maxfun):
        """
        set the maximum number of function evaluations for the node rotation optimization
        """
        self.maxfun = maxfun

    def set_opt_method(self,opt_method):
        """
        set the optimization method for the node rotation optimization
        """
        self.opt_method = opt_method

    def check_node_template_match(self):
        """
        precheck, check if the number of nodes in the template matches the maximum degree of the node in the template
        """
        return len(self.node_x_ccoords)== self.node_max_degree


    def optimize(self):
        """
        two optimization steps:
        1. optimize the node rotation
        2. optimize the cell parameters to fit the target MOF cell    
        """
        if not self.check_node_template_match():
            print('The number of nodes in the template does not match the maximum degree of the node in the template')
            raise ValueError('The number of nodes in the template does not match the maximum degree of the node in the template')

        if hasattr(self,'ec_xcoords'):
            ec_xcoords = self.ec_xcoords
            ecoords = self.eccoords
            
        if not hasattr(self,'opt_method'):
            self.opt_method = 'L-BFGS-B'

        if not hasattr(self,'constant_length'):
            self.constant_length = 1.54

        if not hasattr(self,'maxfun'):
            self.maxfun = 10000
  

        G = self.G
        node_xcoords = self.node_x_ccoords
        node_coords = self.node_coords
        linker_length = self.linker_length
        opt_method = self.opt_method
        maxfun = self.maxfun
        constant_length = self.constant_length
        sorted_nodes = sort_nodes_by_type_connectivity(G)

        #firstly, check if all V nodes have highest connectivity
        #secondly, sort all DV nodes by connectivity

        sorted_edges = find_and_sort_edges_bynodeconnectivity(G,sorted_nodes)

        nodes_atoms = {}
        for n in sorted_nodes:
            if 'CV' in n:
                nodes_atoms[n]= self.ec_atom
            else:
                nodes_atoms[n]= self.node_atom

        xxxx_positions_dict = {}
        node_positions_dict = {}
        #reindex the nodes in the xxxx_positions with the index in the sorted_nodes, like G has 16 nodes[2,5,7], but the new dictionary should be [0,1,2]
        for n in sorted_nodes:
            if 'CV' in n:
                xxxx_positions_dict[sorted_nodes.index(n)]=addidx(G.nodes[n]['ccoords']+ec_xcoords) 
            else:
                xxxx_positions_dict[sorted_nodes.index(n)]=addidx(G.nodes[n]['ccoords']+node_xcoords) 

        for n in sorted_nodes:
            if 'CV' in n:
                node_positions_dict[sorted_nodes.index(n)]=G.nodes[n]['ccoords']+ecoords
            else:
                node_positions_dict[sorted_nodes.index(n)]=G.nodes[n]['ccoords']+node_coords

        #reindex the edges in the G with the index in the sorted_nodes
        sorted_edges_of_sortednodeidx = [(sorted_nodes.index(e[0]),sorted_nodes.index(e[1])) for e in sorted_edges]


        # Optimize rotations
        num_nodes = G.number_of_nodes()

        ###3D free rotation
        optimized_rotations,static_xxxx_positions = optimize_rotations(num_nodes,G,sorted_nodes,
                                                                            sorted_edges_of_sortednodeidx, 
                                                                            xxxx_positions_dict,opt_method,maxfun)
        
        # Apply rotations
        rotated_node_positions = apply_rotations_to_atom_positions(optimized_rotations, G, sorted_nodes,node_positions_dict)

        # Save results to XYZ
        #save_xyz("optimized_nodesstructure.xyz", rotated_node_positions) #DEBUG

        rotated_xxxx_positions_dict,optimized_pair=apply_rotations_to_xxxx_positions(optimized_rotations,
                                                                                     G, 
                                                                                     sorted_nodes, 
                                                                                     sorted_edges_of_sortednodeidx,
                                                                                     xxxx_positions_dict)

        start_node = sorted_edges[0][0]#find_nearest_node_to_beginning_point(G)
        #loop all of the edges in G and get the lengths of the edges, length is the distance between the two nodes ccoords
        edge_lengths,lengths = get_edge_lengths(G)

        x_com_length = np.mean([np.linalg.norm(i) for i in node_xcoords])
        new_edge_length = linker_length+2*constant_length+2*x_com_length
        #update the node ccoords in G by loop edge, start from the start_node, and then update the connected node ccoords by the edge length, and update the next node ccords from the updated node

        updated_ccoords,original_ccoords = update_node_ccoords(G,edge_lengths,start_node,new_edge_length)
        #exclude the start_node in updated_ccoords and original_ccoords
        updated_ccoords = {k:v for k,v in updated_ccoords.items() if k!=start_node}
        original_ccoords = {k:v for k,v in original_ccoords.items() if k!=start_node}


        #use optimized_params to update all of nodes ccoords in G, according to the fccoords

        optimized_params = optimize_cell_parameters(self.cell_info,original_ccoords,updated_ccoords)
        sc_unit_cell = unit_cell_to_cartesian_matrix(optimized_params[0],optimized_params[1],optimized_params[2],optimized_params[3],optimized_params[4],optimized_params[5])
        sc_unit_cell_inv = np.linalg.inv(sc_unit_cell)
        sG,scaled_ccoords = update_ccoords_by_optimized_cell_params(self.G,optimized_params)
        scaled_node_positions_dict = {}
        scaled_xxxx_positions_dict = {}

        for n in sorted_nodes:
            if 'CV' in n:
                scaled_xxxx_positions_dict[sorted_nodes.index(n)]=addidx(sG.nodes[n]['ccoords']+ec_xcoords) 
            else:
                scaled_xxxx_positions_dict[sorted_nodes.index(n)]=addidx(sG.nodes[n]['ccoords']+node_xcoords) 

        for n in sorted_nodes:
            if 'CV' in n:
                scaled_node_positions_dict[sorted_nodes.index(n)]=sG.nodes[n]['ccoords']+ecoords
            else:
                scaled_node_positions_dict[sorted_nodes.index(n)]=sG.nodes[n]['ccoords']+node_coords


        # Apply rotations
        scaled_rotated_node_positions  = apply_rotations_to_atom_positions(optimized_rotations, sG, sorted_nodes,scaled_node_positions_dict)
        scaled_rotated_xxxx_positions,optimized_pair = apply_rotations_to_xxxx_positions(optimized_rotations, sG,sorted_nodes, sorted_edges_of_sortednodeidx, scaled_xxxx_positions_dict)
        # Save results to XYZ

        self.sorted_nodes = sorted_nodes
        self.sorted_edges = sorted_edges
        self.sorted_edges_of_sortednodeidx = sorted_edges_of_sortednodeidx
        self.optimized_rotations = optimized_rotations
        self.optimized_params = optimized_params
        self.new_edge_length = new_edge_length
        self.optimized_pair = optimized_pair
        self.scaled_rotated_node_positions = scaled_rotated_node_positions
        self.scaled_rotated_xxxx_positions = scaled_rotated_xxxx_positions
        self.sc_unit_cell = sc_unit_cell
        self.sc_unit_cell_inv = sc_unit_cell_inv
        self.sG_node = sG
        self.nodes_atom = nodes_atoms
        self.rotated_node_positions = rotated_node_positions
        #save_xyz("scale_optimized_nodesstructure.xyz", scaled_rotated_node_positions)

    def place_edge_in_net(self):
        """
        based on the optimized rotations and cell parameters, use optimized pair to find connected X-X pair in optimized cell, 
        and place the edge in the target MOF cell

        return:
            sG (networkx graph):graph of the target MOF cell, with scaled and rotated node and edge positions
        """
        #linker_middle_point = np.mean(linker_x_vecs,axis=0)
        linker_xx_vec = self.linker_x_ccoords
        linker_length = self.linker_length
        optimized_pair = self.optimized_pair
        scaled_rotated_xxxx_positions = self.scaled_rotated_xxxx_positions
        scaled_rotated_node_positions = self.scaled_rotated_node_positions
        sorted_nodes = self.sorted_nodes
        sG_node = self.sG_node
        sc_unit_cell_inv = self.sc_unit_cell_inv
        nodes_atom = self.nodes_atom

        sG = sG_node.copy()
        scalar = (linker_length+2*self.constant_length)/linker_length
        extended_linker_xx_vec = [i*scalar for i in linker_xx_vec]
        norm_xx_vector_record = []
        rot_record = []
        #edges = {}
        for (i,j),pair in optimized_pair.items():
            x_idx_i,x_idx_j = pair
            reindex_i = sorted_nodes.index(i)
            reindex_j = sorted_nodes.index(j)
            x_i = scaled_rotated_xxxx_positions[reindex_i][x_idx_i][1:]
            x_j = scaled_rotated_xxxx_positions[reindex_j][x_idx_j][1:]
            x_i_x_j_middle_point = np.mean([x_i,x_j],axis=0)
            xx_vector = np.vstack([x_i-x_i_x_j_middle_point,x_j-x_i_x_j_middle_point])
            norm_xx_vector = xx_vector/np.linalg.norm(xx_vector)
            
            #print(i,j,reindex_i,reindex_j,x_idx_i,x_idx_j)
            #use superimpose to get the rotation matrix
            #use record to record the rotation matrix for get rid of the repeat calculation
            indices = [index for index, value in enumerate(norm_xx_vector_record) if is_list_A_in_B(norm_xx_vector, value)]
            if len(indices) == 1: 
                rot = rot_record[indices[0]]
                #rot = reorthogonalize_matrix(rot)
            else:
                _, rot, _ = superimpose(extended_linker_xx_vec,xx_vector)
                #rot = reorthogonalize_matrix(rot)
                norm_xx_vector_record.append(norm_xx_vector)
                rot_record.append(rot)
                
            #use the rotation matrix to rotate the linker x coords
            #rotated_xx = np.dot(extended_linker_xx_vec, rot)
            #print(rotated_xx,'rotated_xx',xx_vector) #DEBUG
            placed_edge_ccoords = np.dot(self.linker_ccoords, rot) + x_i_x_j_middle_point

            placed_edge = np.hstack((np.asarray(self.linker_atom), placed_edge_ccoords))
            sG.edges[(i,j)]['coords'] = x_i_x_j_middle_point
            sG.edges[(i,j)]['c_points']=placed_edge
            sG.edges[(i,j)]['f_points'] = np.hstack((placed_edge[:,0:2],cartesian_to_fractional(placed_edge[:,2:5],sc_unit_cell_inv))) #NOTE: modified add the atom type and atom name
            _,sG.edges[(i,j)]['x_coords'] = fetch_X_atoms_ind_array(placed_edge,0,'X')
            #edges[(i,j)]=placed_edge
        #placed_node = {}
        for k,v in scaled_rotated_node_positions.items():
            #print(k,v)
            #placed_node[k] = np.hstack((nodes_atom[k],v))
            sG.nodes[k]['c_points'] = np.hstack((nodes_atom[k],v))
            sG.nodes[k]['f_points'] = np.hstack((nodes_atom[k],cartesian_to_fractional(v,sc_unit_cell_inv)))
            #find the atoms starts with "x" and extract the coordinates
            _,sG.nodes[k]['x_coords'] = fetch_X_atoms_ind_array(sG.nodes[k]['c_points'],0,'X')
        self.sG = sG
        return sG  

    def set_supercell(self,supercell):
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
        self.dv_v_pairs,sG = replace_DV_with_corresponding_V(sG)
        superG= update_supercell_node_fpoints_loose(sG,self.supercell)
        superG = update_supercell_edge_fpoints(sG,superG,self.supercell)
        self.prim_multiedge_bundlings = replace_bundle_dvnode_with_vnode(self.dv_v_pairs,self.multiedge_bundlings)
        self.super_multiedge_bundlings = make_super_multiedge_bundlings(self.prim_multiedge_bundlings,self.supercell)
        superG = update_supercell_bundle(superG,self.super_multiedge_bundlings)
        self.superG = check_multiedge_bundlings_insuperG(self.super_multiedge_bundlings,superG)
        return superG
    
    def make_supercell_ditopic(self):
        """
        make the supercell of the ditopic linker MOF
        """

        sG = self.sG
        self.dv_v_pairs,sG = replace_DV_with_corresponding_V(sG)
        superG= update_supercell_node_fpoints_loose(sG,self.supercell)
        superG = update_supercell_edge_fpoints(sG,superG,self.supercell)
        self.superG = superG
        return superG
    
    
    
    def set_virtual_edge(self,bool_x=False,range=0.5, max_neighbor=2):
        """
        set the virtual edge addition for the bridge type nodes, 
        range is the range to search the virtual edge between two Vnodes directly, should <= 0.5, 
        max_neighbor is the maximum number of neighbors of the node with virtual edge
        """

        self.add_virtual_edge = bool(bool_x)
        self.vir_edge_range = range
        self.vir_edge_max_neighbor = max_neighbor

    
    def add_virtual_edge_for_bridge_node(self):
        """
        after setting the virtual edge search, add the virtual edge to the target supercell superG MOF
        """
        if self.add_virtual_edge:
            superG = add_virtual_edge(self.superG,self.vir_edge_range,self.vir_edge_max_neighbor)
            print('add virtual edge')
            self.superG = superG
        else:
            pass
    
    def set_remove_node_list(self,remove_node_list):
        """
        make defect in the target MOF model by removing nodes
        """
        self.remove_node_list = remove_node_list
    
    def set_remove_edge_list(self,remove_edge_list):
        """
        make defect in the target MOF model by removing edges
        """
        self.remove_edge_list = remove_edge_list

    def make_eG_from_supereG_multitopic(self):
        """
        make the target MOF cell graph with only EDGE and V, link the XOO atoms to the EDGE
        always need to execute with make_supercell_multitopic
        """
        if hasattr(self,'remove_node_list'):
            remove_node_list = self.remove_node_list
        else:
            remove_node_list = []
        if hasattr(self,'remove_edge_list'):
            remove_edge_list = self.remove_edge_list
        else:
            remove_edge_list = []

        eG,_ = superG_to_eG_multitopic(self.superG)
        to_remove_node_name = []
        to_remove_edge_name = []
        for n in eG.nodes():
            if pname(n)!='EDGE':
                if eG.nodes[n]['index'] in remove_node_list:
                    to_remove_node_name .append(n)
            if pname(n)=='EDGE':
                if -1*eG.nodes[n]['index'] in remove_edge_list:
                    to_remove_edge_name.append(n)
        for n in to_remove_node_name+to_remove_edge_name:
            eG.remove_node(n)
       

        eG = remove_node_by_index(eG,remove_node_list,remove_edge_list)
        self.eG = eG
        return eG
        

    def add_xoo_to_edge_multitopic(self): 
        eG = self.eG
        eG,unsaturated_linker,matched_vnode_xind,xoo_dict = addxoo2edge_multitopic(eG,self.sc_unit_cell)
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
        if hasattr(self,'remove_node_list'):
            remove_node_list = self.remove_node_list
        else:
            remove_node_list = []
        if hasattr(self,'remove_edge_list'):
            remove_edge_list = self.remove_edge_list
        else:
            remove_edge_list = []

        eG,_ = superG_to_eG_ditopic(self.superG)
        to_remove_node_name = []
        to_remove_edge_name = []
        for n in eG.nodes():
            if pname(n)!='EDGE':
                if eG.nodes[n]['index'] in remove_node_list:
                    to_remove_node_name.append(n)
            if pname(n)=='EDGE':
                if -1*eG.nodes[n]['index'] in remove_edge_list:
                    to_remove_edge_name.append(n)
        for n in to_remove_node_name+to_remove_edge_name:
            eG.remove_node(n)
       
        eG = remove_node_by_index(eG,remove_node_list,remove_edge_list)
        self.eG = eG
        return eG

    def add_xoo_to_edge_ditopic(self):
        """
        analyze eG and link the XOO atoms to the EDGE, update eG, for ditopic linker MOF
        """
        eG = self.eG
        eG,unsaturated_linker,matched_vnode_xind,xoo_dict = addxoo2edge_ditopic(eG,self.sc_unit_cell)
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
        return self.eG


    def set_node_topic(self,node_topic):
        """
        manually set the node topic, normally should be the same as the maximum degree of the node in the template
        """
        self.node_topic = node_topic
    
    def find_unsaturated_node_eG(self):
        """
        use the eG to find the unsaturated nodes, whose degree is less than the node topic
        """
        eG = self.eG
        if hasattr(self,'node_topic'):
            node_topic = self.node_topic
        else:
            node_topic = self.node_max_degree
        unsaturated_node = find_unsaturated_node(eG,node_topic)
        self.unsaturated_node = unsaturated_node
        return unsaturated_node
    
    def _make_unsaturated_vnode_xoo_dict(self):
        """
        make a dictionary of the unsaturated node and the exposed X connected atom index and the corresponding O connected atoms
        """
        unsaturated_node = self.unsaturated_node
        xoo_dict = self.xoo_dict
        matched_vnode_xind = self.matched_vnode_xind
        #process matched_vnode_xind make it to a dictionary
        matched_vnode_xind_dict = {}
        for [k,v] in matched_vnode_xind:
            if k in xoo_dict.keys():
                matched_vnode_xind_dict[k].extend(v)
            else:
                matched_vnode_xind_dict[k] = [v]

        unsaturated_vnode_xind_dict ={}
        xoo_keys = list(xoo_dict.keys())
        #for each unsaturated node, get the upmatched x index and xoo atoms
        for unsat_v in unsaturated_node:
            if unsat_v in matched_vnode_xind_dict.keys():
                unsaturated_vnode_xind_dict[unsat_v] = [i for i in xoo_keys if i not in matched_vnode_xind_dict[unsat_v]]
            else:
                unsaturated_vnode_xind_dict[unsat_v] = xoo_keys
        
        #based on the unsaturated_vnode_xind_dict, add termination to the unsaturated node xoo
        #loop over unsaturated nodes, and find all exposed X atoms and use paied xoo atoms to form a termination
        unsaturated_vnode_xoo_dict = {}
        for vnode,exposed_x_indices in unsaturated_vnode_xind_dict.items():
            for xind in exposed_x_indices:  
                x_fpoints = self.eG.nodes[vnode]['f_points'][xind]
                x_cpoints = np.hstack((x_fpoints[0:2],fractional_to_cartesian(x_fpoints[2:5],self.sc_unit_cell))) #NOTE: modified add the atom type and atom name
                oo_ind_in_vnode =  self.xoo_dict[xind]
                oo_fpoints_in_vnode = [self.eG.nodes[vnode]['f_points'][i] for i in oo_ind_in_vnode]
                oo_fpoints_in_vnode = np.vstack(oo_fpoints_in_vnode)
                oo_cpoints = np.hstack((oo_fpoints_in_vnode[:,0:2],fractional_to_cartesian(oo_fpoints_in_vnode[:,2:5],self.sc_unit_cell)))#NOTE: modified add the atom type and atom name

                unsaturated_vnode_xoo_dict[(vnode,xind)] = {'xind':xind,'oo_ind':oo_ind_in_vnode,
                                                    'x_fpoints':x_fpoints,
                                                    'x_cpoints':x_cpoints,
                                                    'oo_fpoints':oo_fpoints_in_vnode,
                                                    'oo_cpoints':oo_cpoints}
                
        self.unsaturated_vnode_xind_dict = unsaturated_vnode_xind_dict
        self.unsaturated_vnode_xoo_dict = unsaturated_vnode_xoo_dict

    def set_node_terminamtion(self,term_file):
        """
        pdb file, set the node termination file, which contains the information of the node terminations, should have X of connected atom (normally C),
        Y of two connected O atoms (if in carboxylate group) to assist the placement of the node terminations
        """

        term_data = termpdb(term_file)
        term_info = term_data[:, :-3]
        term_coords = term_data[:, -3:]
        xterm, _ = Xpdb(term_data, 'X')
        oterm, _ = Xpdb(term_data, 'Y')
        term_xvecs = xterm[:, -3:]
        term_ovecs = oterm[:, -3:]
        term_coords = term_coords.astype('float')
        term_xvecs = term_xvecs.astype('float')
        term_ovecs = term_ovecs.astype('float')

        term_ovecs_c = np.mean(np.asarray(term_ovecs), axis=0)
        term_coords = term_coords - term_ovecs_c
        term_xoovecs = np.vstack((term_xvecs, term_ovecs))
        term_xoovecs = term_xoovecs - term_ovecs_c

        self.term_info = term_info
        self.term_coords = term_coords
        self.term_xoovecs = term_xoovecs

    # Function to add node terminations
    def add_terminations_to_unsaturated_node(self):
        """
        use the node terminations to add terminations to the unsaturated nodes
        
        """
        unsaturated_node = self.unsaturated_node
        xoo_dict = self.xoo_dict
        matched_vnode_xind = self.matched_vnode_xind
        eG = self.eG
        sc_unit_cell = self.sc_unit_cell
        unsaturated_vnode_xind_dict,unsaturated_vnode_xoo_dict = make_unsaturated_vnode_xoo_dict(unsaturated_node,xoo_dict,matched_vnode_xind,eG,sc_unit_cell)
        # term_file: path to the termination file
        # ex_node_cxo_cc: exposed node coordinates
    
        node_oovecs_record = []
        for n in eG.nodes():
            eG.nodes[n]['term_c_points'] = {}
        for exvnode_xind_key in unsaturated_vnode_xoo_dict.keys():
            exvnode_x_ccoords =unsaturated_vnode_xoo_dict[exvnode_xind_key]['x_cpoints']
            exvnode_oo_ccoords = unsaturated_vnode_xoo_dict[exvnode_xind_key]['oo_cpoints']
            node_xoo_ccoords = np.vstack([exvnode_x_ccoords,exvnode_oo_ccoords])
            #make the beginning point of the termination to the center of the oo atoms
            node_oo_center_cvec = np.mean(exvnode_oo_ccoords[:,2:5],axis=0) #NOTE: modified add the atom type and atom name
            node_xoo_cvecs = node_xoo_ccoords[:,2:5] - node_oo_center_cvec #NOTE: modified add the atom type and atom name
            node_xoo_cvecs = node_xoo_cvecs.astype('float')
        #use record to record the rotation matrix for get rid of the repeat calculation

            indices = [index for index, value in enumerate(node_oovecs_record) if is_list_A_in_B(node_xoo_cvecs, value[0])]
            if len(indices) == 1: 
                rot = node_oovecs_record[indices[0]][1]
            else:
                _, rot, _ = superimpose(self.term_xoovecs, node_xoo_cvecs)
                node_oovecs_record.append((node_xoo_cvecs, rot))
            adjusted_term_vecs = np.dot(self.term_coords, rot) + node_oo_center_cvec
            adjusted_term = np.hstack((np.asarray(self.term_info[:,0:1]), adjusted_term_vecs))
            #add the adjusted term to the terms, add index, add the node name
            unsaturated_vnode_xoo_dict[exvnode_xind_key]['node_term_c_points'] = adjusted_term
            eG.nodes[exvnode_xind_key[0]]['term_c_points'][exvnode_xind_key[1]] = adjusted_term
    
        self.unsaturated_vnode_xoo_dict = unsaturated_vnode_xoo_dict
        self.eG = eG
        return eG

if __name__ == '__main__':
    start_time = time.time()
    node_cif = 'node1.cif'
    linker_cif = 'diedge.cif'
    node_target_type = 'Zr'
    #template cif 
    template_cif = 'fcu.cif'
    template_cell_info,_,vvnode = extract_type_atoms_fcoords_in_primitive_cell(template_cif, 'V')
    _,_,eenode = extract_type_atoms_fcoords_in_primitive_cell(template_cif, 'E')
    unit_cell = extract_unit_cell(template_cell_info)

    vvnode = np.unique(np.array(vvnode,dtype=float),axis=0)
    eenode = np.unique(np.array(eenode,dtype=float),axis=0)
    ##loop over super333xxnode and super333yynode to find the pair of x node in unicell which pass through the yynode
    vvnode333 = make_supercell_3x3x3(vvnode)
    eenode333 = make_supercell_3x3x3(eenode)
    fcu = net_optimizer()
    fcu.analyze_template_ditopic(vvnode333,eenode333,unit_cell,template_cell_info)
    fcu.node_info(node_cif,node_target_type)
    fcu.linker_info(linker_cif)
    fcu.set_maxfun(10000)
    fcu.set_opt_method('L-BFGS-B')
    fcu.optimize()
    print("--- %s seconds ---" % (time.time() - start_time))
    fcu.place_edge_in_net()
    supercell = [0,0,0]
    fcu.set_supercell(supercell)
    fcu.make_supercell_ditopic()
    fcu.set_virtual_edge(False)
    all_eG = fcu.make_eG_from_supereG_ditopic()
    eG = fcu.main_frag_eG()
    fcu.add_xoo_to_edge_ditopic()
    fcu.set_node_topic(12)
    unsaturated_node = fcu.find_unsaturated_node_eG()
    sc_unit_cell = fcu.sc_unit_cell
    fcu.set_node_terminamtion('methyl.pdb')
    fcu.add_terminations_to_unsaturated_node()
    print("--- %s seconds ---" % (time.time() - start_time))
    unsaturated_node_xoo_dict = fcu.unsaturated_vnode_xoo_dict
    temp_save_eG_TERM_gro(eG,sc_unit_cell,unsaturated_node_xoo_dict)
    
