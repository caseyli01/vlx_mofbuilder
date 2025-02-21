import numpy as np
from scipy.optimize import minimize,differential_evolution
from _place_node_edge import unit_cell_to_cartesian_matrix, fractional_to_cartesian
from scipy.spatial.transform import Rotation as R

def locate_min_idx(a_array):
    #print(a_array,np.min(a_array))
    idx = np.argmin(a_array)
    row_idx = idx // a_array.shape[1]
    col_idx = idx % a_array.shape[1]
    return row_idx,col_idx

def reorthogonalize_matrix(matrix):
    """
    Ensure the matrix is a valid rotation matrix with determinant = 1.
    """
    U, _, Vt = np.linalg.svd(matrix)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    return R

def sort_solver_by_cost(cost_matrix,pairs):
    #find the row and column index of the minimum value in the cost matrix
    #get the solver of hungarian algorithm (but #the solver is from the first row)
    #sort by cost_matrix[i,j] sort the result from the minimum cost to the maximum cost
    #the result is the pair of the row and column index
    costs = []
    for i in range(len(pairs)):
        row,column = pairs[i]
        costs.append(cost_matrix[row,column])
    sorted_idx = np.argsort(costs)
    sorted_pairs = [pairs[idx] for idx in sorted_idx]
    return sorted_pairs


#after test, we find that we cannot get an exclusive pair, because of the bad initial guess
##def update_pairs(pairs,atom_positions,i,j):
##    nodeA_idx_set = atom_positions[i][:,0]
##    nodeB_idx_set = atom_positions[j][:,0]
##    correct_idx_pair =[]
##    for k in range(len(pairs)):
##        idx_A,idx_B = nodeA_idx_set[pairs[k][0]],nodeB_idx_set[pairs[k][1]]
##        correct_idx_pair.append((idx_A,idx_B))
##    return correct_idx_pair

#def _find_edge_pairings(sorted_edges, atom_positions):
#    """
#    Identify optimal pairings for each edge in the graph.
#
#    Parameters:
#        G (networkx.Graph): Graph structure with edges between nodes.
#        atom_positions (dict): Positions of X atoms for each node.
#
#    Returns:
#        dict: Mapping of edges to optimal atom pairs.
#              Example: {(0, 1): [(0, 3), (1, 2)], ...}
#    """
#
#    edge_pairings = {}
#    
#    for i, j in sorted_edges:
#        node_i_positions = atom_positions[i] #[index,x,y,z]
#        node_j_positions = atom_positions[j] #[index,x,y,z]
#
#
#        # Find optimal pairings for this edge
#        
#        pairs = find_optimal_pairings(node_i_positions, node_j_positions)
#        print(sorted_nodes[i],sorted_nodes[j],pairs)
#        edge_pairings[(i, j)] = pairs #update_pairs(pairs,atom_positions,i,j)
#        #idx_0,idx_1 = pairs[0]
#        #x_idx_0 = atom_positions[i][idx_0][0]
#        #x_idx_1 = atom_positions[j][idx_1][0]
# #
#        #edge_pairings[(i, j)] = update_pairs(pairs,atom_positions,i,j) #but only first pair match
#        #atom_positions[i] = np.delete(atom_positions[i], idx_0, axis=0)
#        #atom_positions[j] = np.delete(atom_positions[j], idx_1, axis=0)
#
#    return edge_pairings


def axis_rotation_matrix(axis, theta):
    """
    Compute the rotation matrix for a rotation around an axis by an angle theta.

    Parameters:
        axis (tuple): The axis vector (a, b, c).
        theta (float): The rotation angle in radians.

    Returns:
        numpy.ndarray: The 3x3 rotation matrix.
    """
    a, b, c = axis
    axis = np.array([a, b, c])
    axis = axis / np.linalg.norm(axis)  # Normalize the axis vector
    a, b, c = axis

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    I = np.eye(3)
    K = np.array([
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0]
    ])
    
    R = I + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
    return R

def axis_objective_function(thetas, axis, static_atom_positions, G,sorted_nodes,sorted_edges):
    """
    Objective function to minimize distances between paired atoms along edges.

    Parameters:
        theta (float): The rotation angle in radians.
        axis (tuple): The axis vector (a, b, c).
        G (networkx.Graph): Graph structure.
        static_atom_positions (dict): Original positions of X atoms for each node.
        edge_pairings (dict): Precomputed pairings for each edge.

    Returns:
        float: Total distance metric to minimize.
    """
    total_distance = 0.0

    for (i, j) in sorted_edges:
        R_i = reorthogonalize_matrix(axis_rotation_matrix(axis, thetas[i]))
        R_j = reorthogonalize_matrix(axis_rotation_matrix(axis, thetas[j]))

        com_i = G.nodes[sorted_nodes[i]]['ccoords']
        com_j = G.nodes[sorted_nodes[j]]['ccoords']

        # Rotate positions around their mass center
        rotated_i_positions = np.dot(static_atom_positions[i][:,1:] - com_i, R_i.T) + com_i
        rotated_j_positions = np.dot(static_atom_positions[j][:,1:] - com_j, R_j.T) + com_j

        for idx_i in range(len(rotated_i_positions)):
            for idx_j in range(len(rotated_j_positions)):
                dist = np.linalg.norm(rotated_i_positions[int(idx_i)] - rotated_j_positions[int(idx_j)])
                total_distance += dist ** 2

    return total_distance

def _axis_objective_function(thetas, axis, static_atom_positions, G,sorted_nodes,edge_pairings):
    """
    Objective function to minimize distances between paired atoms along edges.

    Parameters:
        theta (float): The rotation angle in radians.
        axis (tuple): The axis vector (a, b, c).
        G (networkx.Graph): Graph structure.
        static_atom_positions (dict): Original positions of X atoms for each node.
        edge_pairings (dict): Precomputed pairings for each edge.

    Returns:
        float: Total distance metric to minimize.
    """
    total_distance = 0.0

    for (i, j), pairs in edge_pairings.items():
        R_i = reorthogonalize_matrix(axis_rotation_matrix(axis, thetas[i]))
        R_j = reorthogonalize_matrix(axis_rotation_matrix(axis, thetas[j]))

        com_i = G.nodes[sorted_nodes[i]]['ccoords']
        com_j = G.nodes[sorted_nodes[j]]['ccoords']

        # Rotate positions around their mass center
        rotated_i_positions = np.dot(static_atom_positions[i][:,1:] - com_i, R_i.T) + com_i
        rotated_j_positions = np.dot(static_atom_positions[j][:,1:] - com_j, R_j.T) + com_j

        for idx_i, idx_j in pairs:
            dist = np.linalg.norm(rotated_i_positions[int(idx_i)] - rotated_j_positions[int(idx_j)])
            total_distance += dist ** 2

    return total_distance

def axis_optimize_rotations(axis, num_nodes,G,sorted_nodes,sorted_edges, atom_positions,opt_methods="L-BFGS-B",maxfun=15000):
    """
    Optimize the rotation angles around a given axis to minimize the difference between
    rotated and target positions for each node.

    Parameters:
        axis (tuple): The axis vector (a, b, c).
        G (networkx.Graph): Graph structure.
        static_atom_positions (dict): Original positions of X atoms for each node.
        edge_pairings (dict): Precomputed pairings for each edge.
        initial_thetas (numpy.ndarray): Initial guesses for the rotation angles.

    Returns:
        numpy.ndarray: The optimized rotation angles for each node.
    """
    initial_thetas = np.zeros(num_nodes) # Initial guess for rotation angles
    static_atom_positions = atom_positions.copy()
    # Precompute edge-specific pairings
    #edge_pairings = find_edge_pairings(sorted_edges, atom_positions)
    result = minimize(axis_objective_function, initial_thetas, 
                      args=(axis,  static_atom_positions,G,sorted_nodes,sorted_edges), 
                      method=opt_methods,options={"maxiter": 5000, "disp": True,"maxfun": maxfun})
    optimized_thetas = result.x

    # Compute the rotation matrices for each node
    optimized_rotations = [reorthogonalize_matrix(axis_rotation_matrix(axis, theta)) for theta in optimized_thetas]
    
    
    # Return the optimized rotation matrices 

    return optimized_rotations

def compute_rotation_with_pairing(connected_nodes, atom_positions,current_rotation_matrix):
    """
    Compute the optimal rotation matrix for node pairs, starting from the current rotation matrix.
    
    Parameters:
        node_i_positions (numpy.ndarray): Positions of X atoms in node i (Nx3 array).
        node_j_positions (numpy.ndarray): Positions of X atoms in node j (Mx3 array).
        current_rotation_matrix (numpy.ndarray): The current 3x3 rotation matrix for node i.

    Returns:
        rotation_matrix (numpy.ndarray): Optimized 3x3 rotation matrix for node i.
    """
    
    i, j = connected_nodes
    # Extract paired positions
    paired_node_i = atom_positions[i][:,1:]
    paired_node_j = atom_positions[j][:,1:]

    # Compute the centers of mass for both sets
    com_i = np.mean(paired_node_i, axis=0)
    com_j = np.mean(paired_node_j, axis=0)

    # Translate positions to the center of mass
    translated_i = paired_node_i - com_i
    translated_j = paired_node_j - com_j

    # Apply the current rotation matrix to node_i's positions
    rotated_translated_i = np.dot(translated_i, current_rotation_matrix.T)

    # Compute covariance matrix and SVD
    H = np.dot(rotated_translated_i.T, translated_j)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)

    # Ensure the resulting matrix is a valid rotation matrix (det = 1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)

    # Update the rotation matrix by combining the current and incremental rotation
    optimized_rotation_matrix = np.dot(R, current_rotation_matrix)

    return optimized_rotation_matrix
def objective_function_pre(params, G, static_atom_positions, sorted_nodes,sorted_edges):
    """
    Objective function to minimize distances between paired node to paired node_com along edges.

    Parameters:
        params (numpy.ndarray): Flattened array of rotation matrices.
        G (networkx.Graph): Graph structure.
        atom_positions (dict): Original positions of X atoms for each node.


    Returns:
        float: Total distance metric to minimize.
    """
    num_nodes = len(G.nodes())
    rotation_matrices = params.reshape(num_nodes, 3, 3)
    total_distance = 0.0

    for (i, j) in sorted_edges:
        R_i = reorthogonalize_matrix(rotation_matrices[i])

        com_i = G.nodes[sorted_nodes[i]]['ccoords']
        com_j = G.nodes[sorted_nodes[j]]['ccoords']
        # Rotate positions around their mass center
        rotated_i_positions = np.dot(static_atom_positions[i][:,1:] - com_i, R_i.T) + com_i


        dist_matrix = np.empty((len(rotated_i_positions), 1))    
        for idx_i in range(len(rotated_i_positions)):
                dist = np.linalg.norm(rotated_i_positions[idx_i] - com_j)
                dist_matrix[idx_i,0] = dist
                #total_distance += dist ** 2
        if np.argmin(dist_matrix) > 1:
            total_distance += 1e4 #penalty for the distance difference
        else:
            total_distance += (np.min(dist_matrix) ** 2)
#
        for idx_i in range(len(rotated_i_positions)):
            #second min and min distance difference not max
            if len(dist_matrix[idx_i, :]) > 1:
                second_min_dist = np.partition(dist_matrix[idx_i, :], 1)[1]
            else:
                second_min_dist = np.partition(dist_matrix[idx_i, :], 0)[0]
            diff = second_min_dist - np.min(dist_matrix[idx_i, :])

            if diff <4:
                total_distance += 1e4
        
        total_distance+= 1e3/(np.max(dist_matrix) - np.min(dist_matrix) ) #reward for the distance difference

   

    return total_distance

def objective_function_after(params, G, static_atom_positions, sorted_nodes,sorted_edges):
    """
    Objective function to minimize distances between paired atoms along edges. just use minimum distance

    Parameters:
        params (numpy.ndarray): Flattened array of rotation matrices.
        G (networkx.Graph): Graph structure.
        atom_positions (dict): Original positions of X atoms for each node.
        edge_pairings (dict): Precomputed pairings for each edge.

    Returns:
        float: Total distance metric to minimize.
    """
    num_nodes = len(G.nodes())
    rotation_matrices = params.reshape(num_nodes, 3, 3)
    total_distance = 0.0

    for (i, j) in sorted_edges:
        R_i = reorthogonalize_matrix(rotation_matrices[i])
        R_j = reorthogonalize_matrix(rotation_matrices[j])

        com_i = G.nodes[sorted_nodes[i]]['ccoords']
        com_j = G.nodes[sorted_nodes[j]]['ccoords']

        # Rotate positions around their mass center
        rotated_i_positions = np.dot(static_atom_positions[i][:,1:] - com_i, R_i.T) + com_i
        rotated_j_positions = np.dot(static_atom_positions[j][:,1:] - com_j, R_j.T) + com_j

        dist_matrix = np.empty((len(rotated_i_positions), len(rotated_j_positions)))    
        for idx_i in range(len(rotated_i_positions)):
            for idx_j in range(len(rotated_j_positions)):
                dist = np.linalg.norm(rotated_i_positions[idx_i] - rotated_j_positions[idx_j])
                dist_matrix[idx_i, idx_j] = dist
 
        if np.argmin(dist_matrix) > 1:
            total_distance += 1e4 #penalty for the distance difference
        else:
            total_distance += (np.min(dist_matrix) ** 2)





        for idx_i in range(len(rotated_i_positions)):
            #second min and min distance difference not max
            if len(dist_matrix[idx_i, :]) > 1:
                second_min_dist = np.partition(dist_matrix[idx_i, :], 1)[1]
            else:
                second_min_dist = np.partition(dist_matrix[idx_i, :], 0)[0]
            diff = second_min_dist - np.min(dist_matrix[idx_i, :])
            if diff <3:
                total_distance += 1e4
        for idx_j in range(len(rotated_j_positions)):
            #second min and min distance difference not max
            if len(dist_matrix[:, idx_j]) > 1:
                second_min_dist = np.partition(dist_matrix[:, idx_j], 1)[1]
            else:
                second_min_dist = np.partition(dist_matrix[:, idx_j], 0)[0]
            diff = second_min_dist - np.min(dist_matrix[:, idx_j])

            if diff <3:
                total_distance += 1e4
        

    return total_distance


def optimize_rotations_pre(num_nodes, G, sorted_nodes, sorted_edges, atom_positions, initial_rotations, opt_method,
                               maxfun, maxiter, disp, eps, iprint):
    """
    Optimize rotations for all nodes in the graph.

    Parameters:
        G (networkx.Graph): Graph structure with edges between nodes.
        atom_positions (dict): Positions of X atoms for each node.

    Returns:
        list: Optimized rotation matrices for all nodes.
    """
    print('optimize_rotations_step1')
    #initial_rotations = np.tile(np.eye(3), (num_nodes, 1)).flatten()
    #get a better initial guess, use random rotation matrix combination
    #initial_rotations  = np.array([reorthogonalize_matrix(np.random.rand(3,3)) for i in range(num_nodes)]).flatten()
    static_atom_positions = atom_positions.copy()
    # Precompute edge-specific pairings
    #edge_pairings = find_edge_pairings(sorted_edges, atom_positions)

    result = minimize(
        objective_function_pre,
        initial_rotations,
        args=(G, static_atom_positions, sorted_nodes, sorted_edges),
        method=opt_method,
        options={'maxfun': maxfun, 
                 'maxiter': maxiter, 
                 'disp': disp, 
                 'eps': eps, 
                 'iprint': iprint, },
    )

    

    #optimized_rotations = result.x.reshape(num_nodes, 3, 3)
    #optimized_rotations = [reorthogonalize_matrix(R) for R in optimized_rotations]
    
    optimized_rotations = result.x
    #optimized_rotations = [reorthogonalize_matrix(R) for R in optimized_rotations]
   ## # Print the optimized pairings after optimization
   ## print("Optimized Pairings (after optimization):")
   ## for (i, j), pairs in edge_pairings.items():
   ##     print(f"Node {i} and Node {j}:")
   ##     for idx_i, idx_j in pairs:
   ##         print(f"  node{i}_{idx_i} -- node{j}_{idx_j}")
   ## print()

    return optimized_rotations,static_atom_positions


def optimize_rotations_after(num_nodes, G, sorted_nodes, sorted_edges, atom_positions, initial_rotations, opt_method,
                               maxfun, maxiter, disp, eps, iprint,):
    """
    Optimize rotations for all nodes in the graph.

    Parameters:
        G (networkx.Graph): Graph structure with edges between nodes.
        atom_positions (dict): Positions of X atoms for each node.

    Returns:
        list: Optimized rotation matrices for all nodes.
    """
    print('optimize_rotations_step2')
    #get a better initial guess, use random rotation matrix combination
    #initial_rotations  = np.array([reorthogonalize_matrix(np.random.rand(3,3)) for i in range(num_nodes)]).flatten()
    static_atom_positions = atom_positions.copy()
    # Precompute edge-specific pairings
    #edge_pairings = find_edge_pairings(sorted_edges, atom_positions)

    result = minimize(
        objective_function_after,
        initial_rotations,
        args=(G, static_atom_positions, sorted_nodes,sorted_edges),
        method= opt_method,
        options={'maxfun': maxfun, 
            'maxiter': maxiter, 
            'disp': disp, 
            'eps': eps, 
            'iprint': iprint, },
    )

    

    optimized_rotations = result.x.reshape(num_nodes, 3, 3)
    optimized_rotations = [reorthogonalize_matrix(R) for R in optimized_rotations]
    
   ## # Print the optimized pairings after optimization
   ## print("Optimized Pairings (after optimization):")
   ## for (i, j), pairs in edge_pairings.items():
   ##     print(f"Node {i} and Node {j}:")
   ##     for idx_i, idx_j in pairs:
   ##         print(f"  node{i}_{idx_i} -- node{j}_{idx_j}")
   ## print()

    return optimized_rotations,static_atom_positions





def apply_rotations_to_atom_positions(optimized_rotations, G,sorted_nodes, atom_positions):
    """
    Apply the optimized rotation matrices to the atom positions.

    Parameters:
        optimized_rotations (list): Optimized rotation matrices for each node.
        G (networkx.Graph): Graph structure.
        atom_positions (dict): Original positions of X atoms for each node.

    Returns:
        dict: Rotated positions for each node.
    """
    rotated_positions = {}

    for i, node in enumerate(sorted_nodes):
        #if node type is V
       # if 'DV' in G.nodes[node]['type']:
            #continue
        R = optimized_rotations[i]
        
        original_positions = atom_positions[i]
    
        com = G.nodes[node]['ccoords']

        # Translate, rotate, and translate back to preserve the mass center
        translated_positions = original_positions - com
        rotated_translated_positions = np.dot(translated_positions, R.T)
        rotated_positions[node] = rotated_translated_positions + com
        
    return rotated_positions




def find_optimal_pairings(node_i_positions, node_j_positions):
    """
    Find the optimal one-to-one pairing between atoms in two nodes using the Hungarian algorithm.
    """
    num_i, num_j = len(node_i_positions), len(node_j_positions)
    cost_matrix = np.zeros((num_i, num_j))
    for i in range(num_i):
        for j in range(num_j):
            cost_matrix[i, j] = np.linalg.norm(node_i_positions[i,1:] - node_j_positions[j,1:])

    #row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #print(cost_matrix.shape) #DEBUG
    row_ind, col_ind = locate_min_idx(cost_matrix)
    #print(row_ind,col_ind,cost_matrix) #DEBUG
  

    return [row_ind,col_ind]

#after test, we find that we cannot get an exclusive pair, because of the bad initial guess
##def update_pairs(pairs,atom_positions,i,j):
##    nodeA_idx_set = atom_positions[i][:,0]
##    nodeB_idx_set = atom_positions[j][:,0]
##    correct_idx_pair =[]
##    for k in range(len(pairs)):
##        idx_A,idx_B = nodeA_idx_set[pairs[k][0]],nodeB_idx_set[pairs[k][1]]
##        correct_idx_pair.append((idx_A,idx_B))
##    return correct_idx_pair




#after test, we find that we cannot get an exclusive pair, because of the bad initial guess
##def update_pairs(pairs,atom_positions,i,j):
##    nodeA_idx_set = atom_positions[i][:,0]
##    nodeB_idx_set = atom_positions[j][:,0]
##    correct_idx_pair =[]
##    for k in range(len(pairs)):
##        idx_A,idx_B = nodeA_idx_set[pairs[k][0]],nodeB_idx_set[pairs[k][1]]
##        correct_idx_pair.append((idx_A,idx_B))
##    return correct_idx_pair





def find_edge_pairings(sorted_nodes,sorted_edges, atom_positions):
    """
    Identify optimal pairings for each edge in the graph.

    Parameters:
        G (networkx.Graph): Graph structure with edges between nodes.
        atom_positions (dict): Positions of X atoms for each node.

    Returns:
        dict: Mapping of edges to optimal atom pairs.
              Example: {(0, 1): [(0, 3), (1, 2)], ...}
    """

    edge_pairings = {}
    
    for i, j in sorted_edges:
        node_i_positions = atom_positions[i] #[index,x,y,z]
        node_j_positions = atom_positions[j] #[index,x,y,z]


        # Find optimal pairings for this edge
        
        pairs = find_optimal_pairings(node_i_positions, node_j_positions)
        #print(sorted_nodes[i],sorted_nodes[j],pairs) #DEBUG
        edge_pairings[(i, j)] = pairs #update_pairs(pairs,atom_positions,i,j)
        #idx_0,idx_1 = pairs[0]
        #x_idx_0 = atom_positions[i][idx_0][0]
        #x_idx_1 = atom_positions[j][idx_1][0]
 #
        #edge_pairings[(i, j)] = update_pairs(pairs,atom_positions,i,j) #but only first pair match
        #atom_positions[i] = np.delete(atom_positions[i], idx_0, axis=0)
        #atom_positions[j] = np.delete(atom_positions[j], idx_1, axis=0)

    return edge_pairings

def apply_rotations_to_Xatoms_positions(optimized_rotations, G,sorted_nodes,sorted_edges_of_sortednodeidx, Xatoms_positions_dict):
    """
    Apply the optimized rotation matrices to the atom positions.

    Parameters:
        optimized_rotations (list): Optimized rotation matrices for each node.
        G (networkx.Graph): Graph structure.
        atom_positions (dict): Original positions of X atoms for each node.

    Returns:
        dict: Rotated positions for each node.
    """
    rotated_positions = Xatoms_positions_dict.copy()

    for i, node in enumerate(sorted_nodes):
        #if node type is V
        #if 'DV' in G.nodes[node]['type']:
            #continue
        R = optimized_rotations[i]

        
        original_positions = rotated_positions[i][:,1:]
        com = G.nodes[node]['ccoords']

        # Translate, rotate, and translate back to preserve the mass center
        translated_positions = original_positions - com
        rotated_translated_positions = np.dot(translated_positions, R.T)
        rotated_positions[i][:,1:] = rotated_translated_positions + com
    edge_pair=find_edge_pairings(sorted_nodes, sorted_edges_of_sortednodeidx, rotated_positions)
    #print("Optimized Pairings (after optimization):") #DEBUG
    
    optimized_pair = {}

    for (i, j), pair in edge_pair.items():
        #print(f"Node {sorted_nodes[i]} and Node {sorted_nodes[j]}:") #DEBUG
        idx_i, idx_j = pair
        #print(f"  node{sorted_nodes[i]}_{int(idx_i)} -- node{sorted_nodes[j]}_{int(idx_j)}") #DEBUG
        optimized_pair[sorted_nodes[i],sorted_nodes[j]] = (int(idx_i),int(idx_j))
 


    return rotated_positions,optimized_pair

#use optimized_params to update all of nodes ccoords in G, according to the fccoords
def update_ccoords_by_optimized_cell_params(G,optimized_params):
    sG = G.copy()
    a,b,c,alpha,beta,gamma = optimized_params
    T_unitcell = unit_cell_to_cartesian_matrix(a,b,c,alpha,beta,gamma)
    updated_ccoords = {}
    for n in sG.nodes():
        updated_ccoords[n] = fractional_to_cartesian(T_unitcell,sG.nodes[n]['fcoords'].T).T
        sG.nodes[n]['ccoords'] = updated_ccoords[n]
    return sG,updated_ccoords





def save_xyz(filename, rotated_positions_dict,sorted_nodes):
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
