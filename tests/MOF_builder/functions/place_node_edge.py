import numpy as np
from place_bbs import superimpose

def unit_cell_to_cartesian_matrix(aL,bL,cL,alpha,beta,gamma):
    pi = np.pi
    """Convert unit cell parameters to a Cartesian transformation matrix."""
    aL,bL,cL,alpha,beta,gamma = list(map(float, (aL,bL,cL,alpha,beta,gamma)))
    ax = aL
    ay = 0.0
    az = 0.0
    bx = bL * np.cos(gamma * pi / 180.0)
    by = bL * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = cL * np.cos(beta * pi / 180.0)
    cy = (cL * bL * np.cos(alpha * pi /180.0) - bx * cx) / by
    cz = (cL ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5
    unit_cell = np.asarray([[ax,ay,az],[bx,by,bz],[cx,cy,cz]]).T
    return unit_cell

def fractional_to_cartesian(fractional_coords, T):
    """Convert fractional coordinates to Cartesian using the transformation matrix."""
    return np.dot(T,fractional_coords.T).T

def cartesian_to_fractional(cartesian_coords, unit_cell_inv):
    """Convert Cartesian coordinates to fractional coordinates using the inverse transformation matrix."""
 
    return np.dot(unit_cell_inv,cartesian_coords.T).T



# Add row indices as the first column
def addidx(array):
    row_indices = np.arange(array.shape[0]).reshape(-1, 1).astype(int)
    new_array = np.hstack((row_indices, array))
    return new_array

def get_edge_lengths(G):
    edge_lengths = {}
    lengths = []
    for e in G.edges():
        i,j = e
        length = np.linalg.norm(G.nodes[i]['ccoords']-G.nodes[j]['ccoords'])
        length = np.round(length,3)
        edge_lengths[(i,j)] = length
        lengths.append(length)
    if len(set(lengths)) != 1:
        print('more than one type of edge length')
        #if the length are close, which can be shown by std 
        if np.std(lengths) < 0.1:
            print('the edge lengths are close')
        else:
            print('the edge lengths are not close')
        print(set(lengths))
    return edge_lengths,set(lengths)

def update_node_ccoords(G,edge_lengths,start_node,new_edge_length):
    updated_ccoords = {}
    original_ccoords = {}
    updated_ccoords[start_node] = G.nodes[start_node]['ccoords']
    original_ccoords[start_node] = G.nodes[start_node]['ccoords']
    updated_node = [start_node]
    for i in range(len(G.nodes())-1):
        for n in updated_node:
            for nn in G.neighbors(n):
                if nn in updated_node:
                    continue
                edge = (n,nn)
                edge_length = edge_lengths[edge]
                updated_ccoords[nn] =  updated_ccoords[n] + (G.nodes[nn]['ccoords']-G.nodes[n]['ccoords'])*new_edge_length/edge_length
                original_ccoords[nn] = G.nodes[nn]['ccoords'] 
                updated_node.append(nn)  

    return updated_ccoords,original_ccoords


#according to the optimized_pair of x in each node, we extract all the linkers x-x vector and then superimpose linkers in 
def is_list_A_in_B(A,B):
        return all([np.allclose(a,b,atol=0.05) for a,b in zip(A,B)])

def place_edgeinnodeframe(sorted_nodes,optimized_pair,node_atom,linker_atom,linker_x_vecs,linker_ccoords,scaled_rotated_xxxx_positions,scaled_rotated_chain_node_positions):
    
    linker_middle_point = np.mean(linker_x_vecs,axis=0)
    linker_xx_vec = linker_x_vecs - linker_middle_point
    norm_linker_xx_vec = linker_xx_vec/np.linalg.norm(linker_xx_vec)
    translated_linker_coords = linker_ccoords - linker_middle_point
    norm_xx_vector_record = []
    rot_record = []
    edges = {}
    for (i,j),pair in optimized_pair.items():
        x_idx_i,x_idx_j = pair
        reindex_i = sorted_nodes.index(i)
        reindex_j = sorted_nodes.index(j)
        x_i = scaled_rotated_xxxx_positions[reindex_i][x_idx_i][1:]
        x_j = scaled_rotated_xxxx_positions[reindex_j][x_idx_j][1:]
        x_i_x_j_middle_point = np.mean([x_i,x_j],axis=0)
        xx_vector = np.vstack([x_i-x_i_x_j_middle_point,x_j-x_i_x_j_middle_point])
        norm_xx_vector = xx_vector/np.linalg.norm(xx_vector)
        norm_xx_vector = np.round(xx_vector,6)
        #print(i,j,reindex_i,reindex_j,x_idx_i,x_idx_j) #DEBUG
        #use superimpose to get the rotation matrix
        #use record to record the rotation matrix for get rid of the repeat calculation
        indices = [index for index, value in enumerate(norm_xx_vector_record) if is_list_A_in_B(norm_xx_vector, value)]
        if len(indices) == 1: 
            rot = rot_record[indices[0]]
            #rot = reorthogonalize_matrix(rot)
        else:
            _, rot, _ = superimpose(norm_linker_xx_vec,norm_xx_vector)
            #rot = reorthogonalize_matrix(rot)
            norm_xx_vector_record.append(norm_xx_vector)
            rot_record.append(rot)
            
        #use the rotation matrix to rotate the linker x coords
        placed_edge_ccoords = np.dot(translated_linker_coords, rot) + x_i_x_j_middle_point
        placed_edge = np.hstack((np.asarray(linker_atom), placed_edge_ccoords))
        edges[(i,j)]=placed_edge
    placed_node = {}
    for k,v in scaled_rotated_chain_node_positions.items():
        placed_node[k] = np.hstack((node_atom,v))
    return placed_node,edges