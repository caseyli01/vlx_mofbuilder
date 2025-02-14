import re
import numpy as np
import networkx as nx
from _readcif import process_node
from _node_rotation_matrix_optimizer import optimize_rotations,apply_rotations_to_atom_positions,apply_rotations_to_xxxx_positions,update_ccoords_by_optimized_cell_params
from _scale_cif_optimizer import optimize_cell_parameters
from _place_node_edge import addidx,get_edge_lengths,update_node_ccoords,unit_cell_to_cartesian_matrix,cartesian_to_fractional
from _superimpose import superimpose
#the following are for test need to be removed

from _learn_template import make_supercell_3x3x3,find_pair_v_e,extract_unit_cell
from _learn_template import add_ccoords,set_DV_V,set_DE_E
from _readcif import extract_type_atoms_fcoords_in_primitive_cell
from _place_node_edge import place_edgeinnodeframe
from _output import write_cif_nobond
from only_for_temp_test import MOF_ditopic


def save_xyz(filename, rotated_positions_dict):
    """
    Save the rotated positions to an XYZ file for visualization.
    """
    with open(filename, "w") as file:
        num_atoms = sum(len(positions) for positions in rotated_positions_dict.values())
        file.write(f"{num_atoms}\n")
        file.write("Optimized structure\n")

        for node, positions in rotated_positions_dict.items():
            for pos in positions:
                file.write(f"X{node}   {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")
                
def is_list_A_in_B(A,B):
        return all([np.allclose(a,b,atol=0.05) for a,b in zip(A,B)])


# Function to fetch indices and coordinates of atoms with a specific label
def fetch_X_atoms_ind_array(array, column, X):
	# array: input array
	# column: column index to check for label
	# X: label to search for

	ind = [k for k in range(len(array)) if re.sub(r'\d', '', array[k, column]) == X]
	x_array = array[ind]
	return ind, x_array

def check_inside_unit_cell(point):
    return all([i>=-0.01 and i<=1.01 for i in point])

def find_pair_v_e_c(vvnode333, ecnode333,eenode333): #exist center of linker  in mof 
    G = nx.Graph()
    pair_ve = []
    for e in eenode333:
        #print(e,'check')
        dist_v_e = np.linalg.norm(vvnode333 - e, axis=1)
        # find two v which are nearest to e, and at least one v is in [0,1] unit cell
        v1 = vvnode333[np.argmin(dist_v_e)]
        v1_idx = np.argmin(dist_v_e)
        dist_c_e = np.linalg.norm(ecnode333 - e, axis=1)
        # find two v which are nearest to e, and at least one v is in [0,1] unit cell
        v2 = ecnode333[np.argmin(dist_c_e)]
        v2_idx = np.argmin(dist_c_e)
        #print(v1,v2,'v1,v2')

        # find the center of the pair of v
        center = (v1 + v2) / 2
        # check if there is a v in [0,1] unit cell
        if check_inside_unit_cell(v1) or check_inside_unit_cell(v2):
            # check if the center of the pair of v is around e
            #if abs(np.linalg.norm(v1 - e)+np.linalg.norm(v2 - e) - np.linalg.norm(v1 - v2))< 1e-2: #v1,v2,e are colinear
            if np.linalg.norm(center - e) < 0.1:
                #print(e,v1,v2,'check')
                G.add_node('V'+str(v1_idx), fcoords=v1,note='V')
                G.add_node('CV'+str(v2_idx), fcoords=v2,note ='CV')
                G.add_edge('V'+str(v1_idx), 'CV'+str(v2_idx), fcoords=(v1, v2),fc_center=e),
                pair_ve.append(('V'+str(v1_idx), 'CV'+str(v2_idx), e))
    return pair_ve, len(pair_ve), G

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

class net_optimizer():
    def __init__(self,G):
        self.G = G
    
    def set_template_cell_info(self,cell_info):
        self.template_cell_info = cell_info
    
    def set_constant_length(self,constant_length):
        self.constant_length = constant_length

    def node_info(self,node_cif,node_target_type):
        self.node_cif = node_cif
        self.node_target_type = node_target_type
        self.node_unit_cell,self.node_atom, self.node_x_fcoords, self.node_fcoords = process_node(node_cif, node_target_type)
        self.node_x_ccoords = np.dot(self.node_unit_cell,self.node_x_fcoords.T).T
        self.node_coords = np.dot(self.node_unit_cell,self.node_fcoords.T).T

    def linker_info(self,linker_cif):
        self.linker_cif = linker_cif
        self.linker_unit_cell,self.linker_atom, self.linker_x_fcoords, self.linker_fcoords = process_node(linker_cif, 'X')
        self.linker_x_ccoords = np.dot(self.linker_unit_cell,self.linker_x_fcoords.T).T
        self.linker_length = np.linalg.norm(self.linker_x_ccoords[0]-self.linker_x_ccoords[1])
        linker_ccoords = np.dot(self.linker_unit_cell,self.linker_fcoords.T).T
        self.linker_ccoords = linker_ccoords - np.mean(linker_ccoords,axis=0)

    

    def linker_center(self,linker_center_cif):
        self.linker_center_cif = linker_center_cif
        self.ec_unit_cell,self.ec_atom, self.ec_x_fcoords, self.ec_fcoords = process_node(self.linker_center_cif, 'X')
        self.ec_ccoords = np.dot(self.ec_unit_cell,self.ec_x_fcoords.T).T
        self.eccoords = np.dot(self.ec_unit_cell,self.ec_fcoords.T).T

    def set_maxfun(self,maxfun):
        self.maxfun = maxfun

    def set_opt_method(self,opt_method):
        self.opt_method = opt_method

    def optimize(self):

        node_xcoords = self.node_x_ccoords
        node_coords = self.node_coords
        G = self.G
        linker_length = self.linker_length
        opt_method = self.opt_method
        cell_info = self.template_cell_info

        if not hasattr(self,'constant_length'):
            self.constant_length = 1.54
            constant_length = 1.54
            print('constant_length is not set, use default C-C value 1.54')
        else:
            constant_length = self.constant_length

        if not hasattr(self,'ec_atom'):
            self.ec_atom = None
        if not hasattr(self,'ec_x_ccoords'):
            self.ec_x_ccoords = None
        if not hasattr(self,'ec_fcoords'):
            self.ec_fcoords = None
        if not hasattr(self,'ec_unit_cell'):
            self.ec_unit_cell = None
        if not hasattr(self,'ec_xcoords'):
            self.ec_xcoords = None
            ec_xcoords = None
        if not hasattr(self,'eccoords'):
            self.eccoords = None
            eccoords = None

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
                node_positions_dict[sorted_nodes.index(n)]=G.nodes[n]['ccoords']+eccoords
            else:
                node_positions_dict[sorted_nodes.index(n)]=G.nodes[n]['ccoords']+node_coords
                

        #reindex the edges in the G with the index in the sorted_nodes
        sorted_edges_of_sortednodeidx = [(sorted_nodes.index(e[0]),sorted_nodes.index(e[1])) for e in sorted_edges]


        # Optimize rotations
        num_nodes = G.number_of_nodes()

        #
        ##3D free rotation #TODO:DEBUG
        #
        optimized_rotations,static_xxxx_positions = optimize_rotations(num_nodes,G,sorted_nodes,
                                                                            sorted_edges_of_sortednodeidx, 
                                                                            xxxx_positions_dict,opt_method,self.maxfun)
#
        #DEBUG

        ##optimized_quaternions,static_xxxx_positions = optimize_q_rotations(num_nodes,G,sorted_nodes,sorted_edges_of_sortednodeidx,xxxx_positions_dict,opt_method,self.maxfun)
        ##        # make optimized quaternion to rotation matrix
        ##optimized_rotations = []
        ##for q in optimized_quaternions:
        ##    r = Rotation.from_quat(q)
        ##    #make rotation matrix to be orthogonal
        ##    r = r.as_matrix()
        ##    optimized_rotations.append(r)

        # Apply rotations
        rotated_node_positions = apply_rotations_to_atom_positions(optimized_rotations, G, sorted_nodes,node_positions_dict)

        # Save results to XYZ
        save_xyz("optimized_nodesstructure.xyz", rotated_node_positions)

        rotated_xxxx_positions_dict,optimized_pair=apply_rotations_to_xxxx_positions(optimized_rotations,
                                                                                     G, 
                                                                                     sorted_nodes, 
                                                                                     sorted_edges_of_sortednodeidx,
                                                                                     xxxx_positions_dict)

        start_node_a = sorted_edges[0][0]#find_nearest_node_to_beginning_point(G)
        start_node_b = sorted_edges[0][1]#find_nearest_node_to_beginning_point(G)
        start_nodes = [start_node_a,start_node_b]
        #loop all of the edges in G and get the lengths of the edges, length is the distance between the two nodes ccoords
        edge_lengths,lengths = get_edge_lengths(G)

        x_com_length = np.mean([np.linalg.norm(i) for i in node_xcoords])
        new_edge_length = linker_length+2*constant_length+2*x_com_length
        #update the node ccoords in G by loop edge, start from the start_node, and then update the connected node ccoords by the edge length, and update the next node ccords from the updated node

        updated_ccoords,original_ccoords = update_node_ccoords(G,edge_lengths,start_node_a,new_edge_length) #TODO: include more than one start node to avoid only 2 components connected
        updated_ccoords_b,original_ccoord_b = update_node_ccoords(G,edge_lengths,start_node_b,new_edge_length)
        #updated_ccoords = {**updated_ccoords_a,**updated_ccoords_b}
        #original_ccoords= {**original_ccoord_a,**original_ccoord_b}

        #exclude the start_node in updated_ccoords and original_ccoords
        updated_ccoords = {k:v for k,v in updated_ccoords.items() if k not in start_nodes}
        original_ccoords = {k:v for k,v in original_ccoords.items() if k not in start_nodes}


        #use optimized_params to update all of nodes ccoords in G, according to the fccoords

        optimized_params = optimize_cell_parameters(cell_info,original_ccoords,updated_ccoords)
        sc_unit_cell = unit_cell_to_cartesian_matrix(optimized_params[0],optimized_params[1],optimized_params[2],optimized_params[3],optimized_params[4],optimized_params[5])
        sc_unit_cell_inv = np.linalg.inv(sc_unit_cell)
        sG,scaled_ccoords = update_ccoords_by_optimized_cell_params(G,optimized_params)
        scaled_node_positions_dict = {}
        scaled_xxxx_positions_dict = {}

        for n in sorted_nodes:
            if 'CV' in n:
                scaled_xxxx_positions_dict[sorted_nodes.index(n)]=addidx(sG.nodes[n]['ccoords']+ec_xcoords) 
            else:
                scaled_xxxx_positions_dict[sorted_nodes.index(n)]=addidx(sG.nodes[n]['ccoords']+node_xcoords) 

        for n in sorted_nodes:
            if 'CV' in n:
                scaled_node_positions_dict[sorted_nodes.index(n)]=sG.nodes[n]['ccoords']+eccoords
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
        save_xyz("scale_optimized_nodesstructure.xyz", scaled_rotated_node_positions)

    def place_edge_in_net(self):
        #linker_middle_point = np.mean(linker_x_ccoords,axis=0)
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

            placed_edge = np.hstack((np.asarray(self.linker_atom[:,0:1]), placed_edge_ccoords))
            sG.edges[(i,j)]['coords'] = x_i_x_j_middle_point
            sG.edges[(i,j)]['c_points']=placed_edge
            sG.edges[(i,j)]['f_points'] = np.hstack((placed_edge[:,0:1],cartesian_to_fractional(placed_edge[:,1:4],sc_unit_cell_inv)))
            _,sG.edges[(i,j)]['x_coords'] = fetch_X_atoms_ind_array(placed_edge,0,'X')
            #edges[(i,j)]=placed_edge
        #placed_node = {}
        for k,v in scaled_rotated_node_positions.items():
            #print(k,v)
            #placed_node[k] = np.hstack((nodes_atom[k],v))
            sG.nodes[k]['c_points'] = np.hstack((nodes_atom[k],v))
            sG.nodes[k]['f_points'] = np.hstack((nodes_atom[k][:,0:1],cartesian_to_fractional(v,sc_unit_cell_inv)))
            #find the atoms starts with "x" and extract the coordinates
            _,sG.nodes[k]['x_coords'] = fetch_X_atoms_ind_array(sG.nodes[k]['c_points'],0,'X')
        self.sG = sG
        return sG  

#test if __main__
if __name__ == '__main__':
    node_cif = 'node2.cif'
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

    pair_vv_e,_,G=find_pair_v_e(vvnode333,eenode333)
    G = add_ccoords(G,unit_cell)



    G,_ = set_DV_V(G)
    G = set_DE_E(G)

    fcu = net_optimizer(G)
    fcu.set_template_cell_info(template_cell_info)
    fcu.set_constant_length(1.0)
    fcu.node_info(node_cif,node_target_type)
    fcu.linker_info(linker_cif)
    fcu.set_maxfun(10000)
    fcu.set_opt_method('l-bfgs-b')
    fcu.optimize()
    fcu.place_edge_in_net()
    sorted_nodes = fcu.sorted_nodes
    optimized_pair = fcu.optimized_pair
    node_atom = fcu.node_atom
    linker_atom = fcu.linker_atom
    linker_x_ccoords = fcu.linker_x_ccoords
    linker_ccoords = fcu.linker_ccoords
    scaled_rotated_xxxx_positions = fcu.scaled_rotated_xxxx_positions
    scaled_rotated_chain_node_positions = fcu.scaled_rotated_node_positions
    sc_unit_cell = fcu.sc_unit_cell
    optimized_params = fcu.optimized_params
    sG = fcu.sG

    placed_node,placed_edge = place_edgeinnodeframe(sorted_nodes,optimized_pair,node_atom,linker_atom,linker_x_ccoords,linker_ccoords,scaled_rotated_xxxx_positions,scaled_rotated_chain_node_positions)



    # Save results to XYZ
    placed_all = []
    with open("placed_structure.xyz", "w") as file:
        #node_num_atoms = sum(len(positions) for positions in scaled_rotated_chain_node_positions.values())
        node_num_atoms = sum(len(positions) for positions in placed_node.values())
        edge_num_atoms = sum(len(positions) for positions in placed_edge.values())
        num_atoms = node_num_atoms + edge_num_atoms
        file.write(f"{num_atoms}\n")
        file.write("Optimized structure\n")
        node_idx = 1
        edge_idx = -2
        for node, positions in placed_node.items():
            if 'DV' in G.nodes[node]['type']:
                continue
            for pos in positions:
                file.write(f"{pos[0]}   {pos[1]:.8f} {pos[2]:.8f} {pos[3]:.8f}\n")
                #if pos[0]=='X':
                #    pos[0] = 'C'
                line = np.array([pos[0],pos[1],pos[2],pos[3],0,pos[0],node_idx,'NODE'])
                placed_all.append(line)
            node_idx += 1

        for edge,positions in placed_edge.items():
            
            if 'DE' in G.edges[edge]['type']:
            # pass
                continue
            for pos in positions:
                file.write(f"{pos[0]}   {pos[1]:.8f} {pos[2]:.8f} {pos[3]:.8f}\n")
                #if pos[0]=='X':
                # pos[0] = 'C'
                line = np.array([pos[0],pos[1],pos[2],pos[3],0,pos[0],edge_idx,'EDGE'])
                placed_all.append(line)
            edge_idx -= 1
                #placed_all.append(pos)





    placed_all = np.array(placed_all)

    write_cif_nobond(placed_all, optimized_params, 'placed_structure.cif',sc_unit_cell)

    placed_N = placed_all[placed_all[:,7]=='NODE']
    placed_E = placed_all[placed_all[:,7]=='EDGE']

    n_term_file = 'methyl.pdb'
    e_termfile = 'CCO2.pdb'

    supercell = [0,0,0]

    mof = MOF_ditopic()
    mof.node_topics = 2
    mof.load(sG,placed_all,sc_unit_cell,placed_N,placed_E)
    mof.basic_supercell(np.asarray(supercell),term_file =n_term_file,boundary_cut_buffer = 0.2,edge_center_check_buffer = 0.3,cutxyz=[True,True,True])
    mof.write_basic_supercell('20test.gro','20test.xyz')
    mof.defect_missing()
    mof.term_defective_model(n_term_file=n_term_file,e_termfile=e_termfile)
    mof.write_tntemof('30.gro')    
    #  
    #'''cleave all unsaturated linkers'''
    u_edge_idx=[]
    for i in mof.unsaturated_main_frag_edges:
        u_edge_idx.append(int(i[0][1:]))
    print(u_edge_idx)

    mof.defect_missing([],u_edge_idx)#+[26,20,18,30,8,34,33,4,28,29])
    mof.term_defective_model(n_term_file=n_term_file,e_termfile=e_termfile)
    mof.write_tntemof('33.gro') 

