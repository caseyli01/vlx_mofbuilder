import numpy as np
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import KDTree
from _readcif import extract_type_atoms_fcoords_in_primitive_cell
# use cell_info to generate the matrix for the unit cell to get cartesian coordinates
def make_supercell_3x3x3(array_xyz):
    array_x1 = array_xyz + np.array([1, 0, 0])
    array_x2 = array_xyz + np.array([-1, 0, 0])
    array_y1 = array_xyz + np.array([0, 1, 0])
    array_y2 = array_xyz + np.array([0, -1, 0])
    array_x1_y1 = array_xyz + np.array([1, 1, 0])
    array_x1_y2 = array_xyz + np.array([1, -1, 0])
    array_x2_y1 = array_xyz + np.array([-1, 1, 0])
    array_x2_y2 = array_xyz + np.array([-1, -1, 0])
    
    layer_3x3 = np.vstack((
        array_xyz,
        array_x1,
        array_x2,
        array_y1,
        array_y2,
        array_x1_y1,
        array_x1_y2,
        array_x2_y1,
        array_x2_y2
    ))
    
    layer_3x3_z1 = layer_3x3 + np.array([0, 0, 1])
    layer_3x3_z2 = layer_3x3 + np.array([0, 0, -1])
    
    supercell_3x3x3 = np.vstack((
        layer_3x3,
        layer_3x3_z1,
        layer_3x3_z2
    ))
    
    return supercell_3x3x3


def filter_overlapping_points(points, min_distance):
    """
    Filters out points that are too close to each other based on a minimum distance,
    using KDTree for efficient neighbor searching.

    Parameters:
        points (np.ndarray): Array of shape (N, D) where N is the number of points and D is the dimensionality.
        min_distance (float): Minimum distance allowed between points.

    Returns:
        np.ndarray: Array of filtered points.
    """
    # Build a KDTree for fast neighbor searching
    tree = KDTree(points)
    
    # Keep track of points to remove
    to_keep = np.ones(len(points), dtype=bool)
    
    for i in range(len(points)):
        if not to_keep[i]:
            continue  # Skip points already removed

        # Find neighbors within min_distance (excluding self)
        indices = tree.query_ball_point(points[i], min_distance)
        indices.remove(i)  # Remove self-index
        
        # Mark neighbors for removal
        to_keep[indices] = False
    
    return points[to_keep]

def extract_unit_cell(cell_info):
    pi = np.pi
    aL, bL, cL, alpha, beta, gamma = cell_info
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

# a function, given an array of atoms, use distance to find the clusters of atoms, and return the center of the cluster, we can set a distance thereshold to define the cluster
def find_cluster_center(array_atom):
    array_atom = np.array(array_atom,dtype=float)
    center = np.mean(array_atom,axis=0)
    return center


def clust_analysis_points(array_atom,distance_threshhold):
    #use scipy.cluster distance, to cluster the points
    #find the distance matrix
    dist = pdist(array_atom)
 
    #dist = dist_matrix_exclude_overlapping_pair(array_atom)
    #find the linkage matrix
    Z = linkage(dist, 'ward')
    #find the cluster
    cluster = fcluster(Z, distance_threshhold, criterion='distance')
    #find the center of the cluster
    cluster_center = []
    for i in range(1,max(cluster)+1):
        cluster_center.append(find_cluster_center(array_atom[cluster==i]))
    return cluster_center

#use pdist to calculate the distance between the points
#use squreform to get the distance matrix
def c2f_coords(coords,unit_cell):
    unit_cell_inv = np.linalg.inv(unit_cell)
    fc = np.dot(unit_cell_inv,coords.T).T
    return fc

def cluster_analysis_bridging_node(array_points,unit_cell,cluster_size,distance_threshold):
    fcoords = c2f_coords(array_points,unit_cell)

    #sep cell_points and cell_out points
    cell_in_points = [array_points[ind] for ind, i in enumerate(fcoords) if check_inside_unit_cell(i)]
    cell_out_points = [array_points[ind] for ind, i in enumerate(fcoords) if not check_inside_unit_cell(i)]

    #reorder array_points, cell_in_points should be at head,  cell_out_points at tail
    array_points = np.vstack((cell_in_points,cell_out_points))

    #find cluster in cell_in_points
    pdist_matrix = pdist(array_points)
    squareform_matrix = squareform(pdist_matrix)
    connection_map = {ind:[] for ind in range(len(cell_in_points))}
    #loop over the cell_in points, if the distance between two points in the cupercell is less than the distance_threshold, mark them as connected points
    for i in range(len(cell_in_points)):
        for j in range(len(array_points)):
            if squareform_matrix[i][j]<distance_threshold:
                connection_map[i].append(j)
    # Generate the center of clusters based on all possible combinations of the neighbors
    clusters = []
    for node, neighbors in connection_map.items():
        for combi in combinations(neighbors, cluster_size):
            if node in combi:
                clusters.append(tuple(sorted(combi)))
    set_clusters = set(clusters)
    clust_cc_centers =[]
    for clus in set_clusters:
        clus_atoms = [array_points[i] for i in clus]
        clust_cc_centers.append(find_cluster_center(clus_atoms))

    return clust_cc_centers


##def debug_cluster(cif_file, target_type,cluster_distance_threshhold):
##    cell_info, array_atom, array_target_atoms =extract_type_atoms_fcoords_in_primitive_cell(cif_file, target_type)
##    unit_cell = extract_unit_cell(cell_info)
##    #unit_cell = np.round(unit_cell,3)
##    metal333 = make_supercell_3x3x3(array_target_atoms)
##    metal333 = np.vstack(metal333)
##    #cluster analysis in cartesian coordinates
##    array_metal_ccords = np.dot(unit_cell,metal333.T).T
##    return array_metal_ccords,unit_cell 
  


def extract_bridge_point_cluster_center_from_templatecif(cif_file, target_type,cluster_size, cluster_distance_threshhold):
    cell_info, array_atom, array_target_atoms =extract_type_atoms_fcoords_in_primitive_cell(cif_file, target_type)
    unit_cell = extract_unit_cell(cell_info)
    #unit_cell = np.round(unit_cell,3)
    metal333 = make_supercell_3x3x3(array_target_atoms)
    metal333 = np.vstack(metal333)
    #cluster analysis in cartesian coordinates
    array_metal_ccords = np.dot(unit_cell,metal333.T).T
  
    #cluster_centers_ccoords=clust_analysis_points(array_metal_ccords,cluster_distance_threshhold)
    cluster_centers_ccoords=cluster_analysis_bridging_node(array_metal_ccords,unit_cell,cluster_size, cluster_distance_threshhold)
    if len(cluster_centers_ccoords)==0:
        raise ValueError('No cluster center found')
    cluster_centers_ccoords = np.vstack(cluster_centers_ccoords)
    #cluster_centers should return fractional coordinates
    cluster_centers_fcoords = np.dot(np.linalg.inv(unit_cell),cluster_centers_ccoords.T).T
    #filter cluster centers which is inside the unit cell, boundary condition is [-0.01,1.01]
    
    cluster_centers_fcoords = np.mod(cluster_centers_fcoords,1)
    cluster_centers_fcoords = filter_overlapping_points(cluster_centers_fcoords, 0.001)
    cluster_centers_fcoords = np.round(cluster_centers_fcoords,3)
    #cluster_centers_fcoords = [c for c in cluster_centers_fcoords if all([i>=-0.01 and i<=1.01 for i in c])]
    
    #not consider the overlapped or too close points, use numpy isclose to filter the points



    return cluster_centers_fcoords,cell_info,unit_cell

def extract_cluster_center_from_templatecif(cif_file, target_type,cluster_size, cluster_distance_threshhold):
    cell_info, array_atom, array_target_atoms =extract_type_atoms_fcoords_in_primitive_cell(cif_file, target_type)
    unit_cell = extract_unit_cell(cell_info)
    #unit_cell = np.round(unit_cell,3)
    metal333 = make_supercell_3x3x3(array_target_atoms)
    metal333 = np.vstack(metal333)
    #cluster analysis in cartesian coordinates
    array_metal_ccords = np.dot(unit_cell,metal333.T).T
    cluster_centers_ccoords=clust_analysis_points(array_metal_ccords,cluster_distance_threshhold)
    cluster_centers_ccoords = np.vstack(cluster_centers_ccoords)
    #cluster_centers should return fractional coordinates
    cluster_centers_fcoords = np.dot(np.linalg.inv(unit_cell),cluster_centers_ccoords.T).T
    #filter cluster centers which is inside the unit cell, boundary condition is [-0.01,1.01]
    #cluster_centers_fcoords = np.round(cluster_centers_fcoords,3)
    cluster_centers_fcoords = np.mod(cluster_centers_fcoords,1)
    #cluster_centers_fcoords = [c for c in cluster_centers_fcoords if all([i>=-0.01 and i<=1.01 for i in c])]
    cluster_centers_fcoords = filter_overlapping_points(cluster_centers_fcoords, 0.001)
    cluster_centers_fcoords = np.round(cluster_centers_fcoords,3)
    #cluster_centers_fcoords = np.unique(cluster_centers_fcoords,axis=0)
    return cluster_centers_fcoords,cell_info,unit_cell




#find pair of x, pass y, which means y is the edge center between two x 

#for each y, find nearest x in xxnode333, then check if the center of the pair of x is around y, if yes, the it is valid pair of x
def check_inside_unit_cell(point):
    return all([i>=-0.0 and i<1.0 for i in point])

#check if after np.mod, the fcoords is the same as before
def check_moded_fcoords(point):
    x,y,z = point[0],point[1],point[2]
    if np.mod(x,1)!=x:
        return False
    if np.mod(y,1)!=y:
        return False  
    if np.mod(z,1)!=z:
        return False
    return True

def find_pair_v_e(vvnode333, eenode333):
    G = nx.Graph()
    pair_ve = []
    for e in eenode333:
        dist = np.linalg.norm(vvnode333 - e, axis=1)
        # find two v which are nearest to e, and at least one v is in [0,1] unit cell
        v1 = vvnode333[np.argmin(dist)]
        v1_idx = np.argmin(dist)
        dist[np.argmin(dist)] = 1000
        v2 = vvnode333[np.argmin(dist)]
        v2_idx = np.argmin(dist)
        # find the center of the pair of v
        center = (v1 + v2) / 2
        # check if there is a v in [0,1] unit cell
        if check_inside_unit_cell(v1) or check_inside_unit_cell(v2):
            # check if the center of the pair of v is around e
            if np.linalg.norm(center - e) < 1e-3:
                G.add_node('V'+str(v1_idx), fcoords=v1)
                G.add_node('V'+str(v2_idx), fcoords=v2)
                G.add_edge('V'+str(v1_idx), 'V'+str(v2_idx), fcoords=(v1, v2),fc_center=e),
                pair_ve.append((v1, v2, e))
    return pair_ve, len(pair_ve), G

#add ccoords to the the nodes in the graph
def add_ccoords(G,unit_cell):
    for n in G.nodes():
        G.nodes[n]['ccoords'] = np.dot(unit_cell,G.nodes[n]['fcoords'])
    return G

def set_DV_V(G):
    for n in G.nodes():
        if G.degree(n) == max(dict(G.degree()).values()):
            #G.nodes[n]['type'] = 'V'
            #check if the moded ccoords is in the unit cell
            if check_moded_fcoords(G.nodes[n]['fcoords']):
                G.nodes[n]['type'] = 'V'
            else:
                G.nodes[n]['type'] = 'DV'
        else:
            G.nodes[n]['type'] = 'DV'
    return G

#check e_new in G, if e_new = e+[0,0,1] or e = e+[0,0,-1], [0,1,0],[0,-1,0] then this e_new is invalid not unique

#if make supercell333 of unique_e list cannot find the e_new, then this e_new should be appended to the unique_e list
#check if the e_new is in the unique_e list, if yes, then this e_new is valid, if no, then this e_new is invalid
def find_unitcell_e(all_e):
    cell_e = [all_e[0]]
    count=1
    while count < len(all_e):
        e_check = all_e[count]
        supercell_e = make_supercell_3x3x3(cell_e)
        if not np.any(np.all(np.isclose(e_check,supercell_e),axis=1)):
            cell_e.append(e_check)
        count+=1
    return cell_e

#check e in G, find e in unit_cell, use np.mod to filter the e in unit_cell, set the valid e with E type, others are DE type
def set_DE_E(G):
    all_e =[]
    for e in G.edges():
        all_e.append(G.edges[e]['fc_center'].copy())
    #all_e = np.vstack([limit_to_abs1(edge) for edge in all_e])
    #limit x,y,z of e to [-1,1]
    unique_e = np.vstack(find_unitcell_e(all_e))
    #print(unique_e) #debug
    for e in G.edges():
        if np.any(np.all(np.isclose(G.edges[e]['fc_center'],unique_e),axis=1)):
            G.edges[e]['type'] = 'E'
            #print(G.edges[e]['fc_center'],'E') #debug
        else:
            G.edges[e]['type'] = 'DE'
            #print(G.edges[e]['fc_center'],'DE') #debug
    return G

#firstly, check if all V nodes have highest connectivity
#secondly, sort all DV nodes by connectivity
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
                    sorted_edges.append((int(e[1]),int(e[0])))
                all_edges.pop(ei)
            else:
                ei += 1

    return sorted_edges    
    

#for test 
def check_connectivity(G):
    for n in G.nodes():
        print(n,G.nodes[n],G.degree(n))
#check_connectivity(G) #debug

#check edge information in G
def check_edge(G):
    for e in G.edges():
        print(e,G.edges[e])
##check_edge(G) #debug

if __name__ == '__main__':
    template_cif_file ='MIL53templatecif.cif'
    target_type = 'YY'
    cluster_distance_threshhold = 0.1

    vvnode,cell_info,unit_cell = extract_cluster_center_from_templatecif(template_cif_file, 'YY',1) # node com in template cif file, use fcluster to find cluster and the center of the cluster
    eenode,_,_ = extract_cluster_center_from_templatecif(template_cif_file, 'XX',1) # edge com in template cif file, use fcluster to find the cluster and center of the cluster

    #loop over super333xxnode and super333yynode to find the pair of x node in unicell which pass through the yynode
    vvnode333 = make_supercell_3x3x3(vvnode)
    eenode333 = make_supercell_3x3x3(eenode)
    pair_vv_e,_,G=find_pair_v_e(vvnode333,eenode333)
    G = add_ccoords(G,unit_cell)
    G = set_DV_V(G)
    G = set_DE_E(G)
    check_connectivity(G)
    check_edge(G)
    sorted_nodes = sort_nodes_by_type_connectivity(G)
    sorted_edges = find_and_sort_edges_bynodeconnectivity(G,sorted_nodes)

