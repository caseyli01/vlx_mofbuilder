import numpy as np
node_cif = 'node.cif'
linker_cif = 'linker.xyz'
template = 'uio66.cif'

#linker
linker_x_vecs = selected_type_vecs(linker_cif,'.','X',False)
#_,_, linker_x_vecs=extract_type_atoms_ccoords_in_primitive_cell(linker_cif, 'X')
#ditopic linker only has two x vectors
#linker_length = calc_edge_len(linker_cif,'.') #length of the edge should be x-x length in linker cif file, unit angstrom
linker_length = np.linalg.norm(linker_x_vecs[0]-linker_x_vecs[1])
linker_cell_info,_, linker_atom_site_sector = read_cif(linker_cif)

ll,linker_atom, linker_ccoords = extract_atoms_ccoords_from_lines(linker_cell_info,linker_atom_site_sector)
print(linker_length,'linker_length')


#chainnode

node_target_type = 'Al'
node_unit_cell,node_atom,node_pillar_fvec, node_x_vecs, chain_node_fcoords = process_chain_node(chain_node_cif, node_target_type)

#template cif 

template_cif_file ='MIL53templatecif.cif'
cluster_distance_threshhold = 0.1
vvnode,cell_info,unit_cell = extract_cluster_center_from_templatecif(template_cif_file, 'YY',1,1) # node com in template cif file, use fcluster to find cluster and the center of the cluster
eenode,_,_ = extract_cluster_center_from_templatecif(template_cif_file, 'XX',1,1) # edge com in template cif file, use fcluster to find the cluster and center of the cluster

#loop over super333xxnode and super333yynode to find the pair of x node in unicell which pass through the yynode
vvnode333 = make_supercell_3x3x3(vvnode)
eenode333 = make_supercell_3x3x3(eenode)
pair_vv_e,_,G=find_pair_v_e(vvnode333,eenode333)
G = add_ccoords(G,unit_cell)
G = set_DV_V(G)
G = set_DE_E(G)
#debug
#check_connectivity(G)
#check_edge(G)
#firstly, check if all V nodes have highest connectivity
#secondly, sort all DV nodes by connectivity

sorted_nodes = sort_nodes_by_type_connectivity(G)
#fix one direction of the box, which should be parallel to the pilalar direction
#rotate the node to pillar direction and put all nodes into the cartesian coordinate 
sorted_edges = find_and_sort_edges_bynodeconnectivity(G,sorted_nodes)


PILLAR,pillar_vec = check_if_pillarstack(G) 
pillar_oriented_node_xcoords,pillar_oriented_node_coords = rotate_node_for_pillar(G,node_unit_cell,node_pillar_fvec,pillar_vec,node_x_vecs,chain_node_fcoords)

nodexxxx = []
xxxx_positions_dict = {}
chain_node_positions_dict = {}
chain_node_positions = []
#reindex the nodes in the xxxx_positions with the index in the sorted_nodes, like G has 16 nodes[2,5,7], but the new dictionary should be [0,1,2]
xxxx_positions_dict = {sorted_nodes.index(n):addidx(G.nodes[n]['ccoords']+pillar_oriented_node_xcoords) for n in sorted_nodes}
chain_node_positions_dict = {sorted_nodes.index(n):G.nodes[n]['ccoords']+pillar_oriented_node_coords for n in sorted_nodes}
#reindex the edges in the G with the index in the sorted_nodes
sorted_edges_of_sortednodeidx = [(sorted_nodes.index(e[0]),sorted_nodes.index(e[1])) for e in sorted_edges]

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

# Optimize rotations
num_nodes = G.number_of_nodes()

###3D free rotation
#optimized_rotations,static_xxxx_positions = optimize_rotations(num_nodes,sorted_edges, xxxx_positions_dict)
###2D axis rotation
axis = pillar_vec  # Rotate around x-axis
optimized_rotations = axis_optimize_rotations(axis, num_nodes, G,sorted_nodes,sorted_edges_of_sortednodeidx, xxxx_positions_dict)

# Apply rotations
rotated_node_positions = apply_rotations_to_atom_positions(optimized_rotations, G, sorted_nodes,chain_node_positions_dict)

# Save results to XYZ
save_xyz("optimized_nodesstructure.xyz", rotated_node_positions)

rotated_xxxx_positions_dict,optimized_pair=apply_rotations_to_xxxx_positions(optimized_rotations, G, sorted_nodes, sorted_edges_of_sortednodeidx,xxxx_positions_dict)


start_node = sorted_edges[0][0]#find_nearest_node_to_beginning_point(G)
#loop all of the edges in G and get the lengths of the edges, length is the distance between the two nodes ccoords
edge_lengths,lengths = get_edge_lengths(G)

constant_length = 1.6
x_com_length = np.mean([np.linalg.norm(i) for i in pillar_oriented_node_xcoords])
new_edge_length = linker_length+2*constant_length+2*x_com_length
#update the node ccoords in G by loop edge, start from the start_node, and then update the connected node ccoords by the edge length, and update the next node ccords from the updated node


updated_ccoords,original_ccoords = update_node_ccoords(G,edge_lengths,start_node,new_edge_length)
updated_ccoords,original_ccoords
#exclude the start_node in updated_ccoords and original_ccoords
updated_ccoords = {k:v for k,v in updated_ccoords.items() if k!=start_node}
original_ccoords = {k:v for k,v in original_ccoords.items() if k!=start_node}


#use optimized_params to update all of nodes ccoords in G, according to the fccoords

optimized_params = optimize_cell_parameters(cell_info,original_ccoords,updated_ccoords)
sG,scaled_ccoords = update_ccoords_by_optimized_cell_params(G,optimized_params)
scaled_chain_node_positions_dict = {sorted_nodes.index(n):sG.nodes[n]['ccoords']+pillar_oriented_node_coords for n in sorted_nodes}
scaled_xxxx_positions_dict = {sorted_nodes.index(n):addidx(sG.nodes[n]['ccoords']+pillar_oriented_node_xcoords) for n in sorted_nodes}

# Apply rotations
scaled_rotated_chain_node_positions = apply_rotations_to_atom_positions(optimized_rotations, sG, sorted_nodes,scaled_chain_node_positions_dict)
scaled_rotated_xxxx_positions,optimized_pair = apply_rotations_to_xxxx_positions(optimized_rotations, sG,sorted_nodes, sorted_edges_of_sortednodeidx, scaled_xxxx_positions_dict)
# Save results to XYZ
save_xyz("scale_optimized_nodesstructure.xyz", scaled_rotated_chain_node_positions)

placed_node,placed_edge = place_edgeinnodeframe(sorted_nodes,optimized_pair,node_atom,linker_atom,linker_x_vecs,linker_ccoords,scaled_rotated_xxxx_positions,scaled_rotated_chain_node_positions)