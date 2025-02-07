import numpy as np
import re

def fetch_node_withidx(placed_node,idx_list):
    res_id_list = [i+1 for i in idx_list ]
    res=[]
    for n in res_id_list:
        res.append(placed_node[placed_node[:,6]==n])
    return np.vstack(res)

    
def fetch_edge_withidx(placed_edge,idx_list):
    res_id_list = [-1*i-1 for i in idx_list ]
    res=[]
    for n in res_id_list:
        res.append(placed_edge[placed_edge[:,6]==n])
    return np.vstack(res)

def fetch_edge_withidx_sep(placed_edge,idx_list):
    res_id_list = [-1*i-1 for i in idx_list ]
    res=[]
    for n in res_id_list:
        res.append(placed_edge[placed_edge[:,6]==n])
    return res

def limit_x(x):
    while x>0.5:
        x=x-1
    while x< -0.5:
        x = x+1
    return x

def centerize_edges_cc(target_edges_list,target_node_c,sc_unit_cell):
    edges_update=[]
    edges_update_append = edges_update.append
    for te_ccord in target_edges_list:
        te = te_ccord[:,1:4]- target_node_c
        te_fvec = np.dot(np.linalg.inv(sc_unit_cell),te.T).T
        edge_c_fvec = np.mean(te_fvec,axis=0).tolist()
        cx,cy,cz = edge_c_fvec
        cx1 = limit_x(cx)
        cy1 = limit_x(cy)
        cz1 = limit_x(cz)
        differ = np.asarray([cx1,cy1,cz1])-np.asarray(edge_c_fvec)
        te_update = np.hstack((te_ccord[:,0:1],te+np.dot(sc_unit_cell,differ.T).T+target_node_c,te_ccord[:,4:]))
        edges_update_append(te_update)
        #print(differ)
    return np.vstack((edges_update))


def sort_points_clockwise_around_center(points, center):
    """
    Sorts 4 points in clockwise order around a given center point using arccos for angle comparison.
    
    :param points: List of 4 points [(x1, y1, z1), (x2, y2, z2), ...]
    :param center: The center point (cx, cy, cz)
    :return: Tuple (sorted_points, order), where:
             - sorted_points: Points sorted in clockwise order
             - order: Indices of the points in sorted order
    """
    # Convert points and center to NumPy arrays
    points = np.asarray(points,dtype=float)
    center = np.asarray(center,dtype=float)

    # Step 1: Compute vectors from the center to each point
    vectors = points - center

    # Step 2: Choose an initial reference vector (e.g., vector from center to the first point)
    reference_vector = vectors[0]

    # Step 3: Calculate the angles of all vectors relative to the reference vector
    angles = []
    for i in range(1, len(vectors)):
        # Compute the cosine of the angle between reference_vector and vectors[i]
        cos_theta = np.dot(reference_vector, vectors[i]) / (np.linalg.norm(reference_vector) * np.linalg.norm(vectors[i]))
        # Clamp the value to avoid any floating point errors that might result in an invalid arccos input
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.arccos(cos_theta)  # Angle between vectors in radians
        angles.append((angle, i))

    # Step 4: Sort points based on the angle (clockwise order)
    # We want the smallest angle to come first (since we're sorting counterclockwise by default, so we reverse the sorting)
    angles.sort(key=lambda x: x[0])

    # Start with the reference point (first point)
    sorted_indices = [0]  # Start with the first point
    # Add the other points in clockwise order based on the sorted angles
    for _, index in angles:
        sorted_indices.append(index)

    # Step 5: Return the sorted points
    sorted_points = points[sorted_indices]
    return sorted_points.tolist(), sorted_indices



def sort_points_followXorder(points,Xpoints,sc_unit_cell):
    order = []
    for x in Xpoints:
        dists=[]
        for p in range(len(points)):
            fvec = points[p]-x
            cvec = np.dot(sc_unit_cell,fvec)
            cdist = np.linalg.norm(cvec)
            dists.append(cdist)
        idx=dists.index(min(dists))
        if idx not in order:
            order.append(idx)
        else:
            raise ValueError("find more than one edge near center x ")
    print('edgeorder',order)
    return order



def centerize_edges_fc(target_edges_list,target_node_c_fc,target_nodes,sc_unit_cell):
    edges_update=[]
    edges_c_update=[]
    ordered_edges=[]
    target_nodes_X_fc=target_nodes[[row[5].startswith('X') for row in target_nodes]][:,1:4]
    print('targetnodes',target_nodes[[row[5].startswith('X') for row in target_nodes]][:,1:4]) 
    for te_fcord in target_edges_list:
        te_fvec = te_fcord[:,1:4]- target_node_c_fc
        edge_c_fvec = np.mean(te_fvec,axis=0).tolist()
        cx,cy,cz = edge_c_fvec
        cx1 = limit_x(cx)
        cy1 = limit_x(cy)
        cz1 = limit_x(cz)
        differ = np.asarray([cx1,cy1,cz1])-np.asarray(edge_c_fvec)
        te_update = np.hstack((te_fcord[:,0:1],te_fvec+differ+target_node_c_fc,te_fcord[:,4:]))
        edges_update.append(te_update)
        edges_c_update.append(np.mean(te_fvec+differ+target_node_c_fc,axis=0))
    
    edges_order=sort_points_followXorder(edges_c_update, target_nodes_X_fc,sc_unit_cell)

    for k in edges_order:
            ordered_edges.extend(edges_update[k])
    return np.vstack((ordered_edges))

def merge_multitopic_node_edge_fc(TG,sc_unit_cell,multi_node_name,placed_nodes_arr_fc,placed_edges_arr_fc):
    multitopics=[]
    edges_dict_list=list(TG.edges(data=True,keys=True))
    for c_node in multi_node_name:
        c_node_idx=TG.nodes[c_node]['index']
        linked_multitopic = []
        for i in range(len(edges_dict_list)):
            e_dict=edges_dict_list[i]
            check = (c_node in e_dict[0:2])
            if check:
                linked_multitopic.append(e_dict[2][0])
        multitopics.append((c_node,c_node_idx,linked_multitopic))

    multitopic_edges = []
    for i in range(len(multitopics)):
        c_node = multitopics[i]
        node_idx = [c_node[1]]
        linked_edge_idx = c_node[2]
        target_nodes = fetch_node_withidx(placed_nodes_arr_fc,node_idx)
        target_nodes_c_fc = np.mean(target_nodes[:,1:4],axis=0)
        moded_trans_fc = np.mod(target_nodes_c_fc,1) - target_nodes_c_fc
        target_nodes[:,1:4] = target_nodes[:,1:4] + moded_trans_fc

        target_edge_list = fetch_edge_withidx_sep(placed_edges_arr_fc,linked_edge_idx)
        target_nodes_c_fc = np.mean(target_nodes[:,1:4],axis=0)
        target_edges = centerize_edges_fc(target_edge_list,target_nodes_c_fc,target_nodes,sc_unit_cell)
        multitopic_edge = np.vstack((target_nodes,target_edges))

        for row_n in range(len(multitopic_edge)):
            multitopic_edge[row_n,5]= re.sub('[0-9]','',multitopic_edge[row_n,5]) + str(row_n+1)
        multitopic_edge[:,6]=[-1*i-2]*len(multitopic_edge)
        multitopic_edge[:,7]=['EDGE']*len(multitopic_edge)
        multitopic_edges.append(multitopic_edge)

    multitopic_edges_fcoords = np.vstack(multitopic_edges)

    return multitopic_edges_fcoords