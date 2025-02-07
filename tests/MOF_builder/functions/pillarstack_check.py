import numpy as np
from place_bbs import superimpose
def check_if_pillarstack(G):
    node_node_vec = []
    for n in G.nodes():
        if 'DV' in G.nodes[n]['type']:
            continue
        dist = []
        for nn in G.nodes():
            dist.append(np.linalg.norm(G.nodes[n]['fcoords']-G.nodes[nn]['fcoords']))
        dist = np.array(dist)
        if np.min(dist) < 1e-3:
            dist[np.argmin(dist)] = 1000
        if np.min(dist) > 1:
            continue

        nn = list(G.nodes())[np.argmin(dist)]
        node_node = G.nodes[n]['fcoords']-G.nodes[nn]['fcoords']
        node_node = node_node/np.linalg.norm(node_node)
        node_node_vec.append(node_node)
        
    #check if all of the pair vector are parallel
    for i in range(0,len(node_node_vec)):
        for j in range(i+1,len(node_node_vec)):
            if np.linalg.norm(np.cross(node_node_vec[i],node_node_vec[j])) > 1e-3:
                print('not pillar stack')
                return False,None
    pillar_vec = np.abs(node_node_vec[0])
    print('pillar stack',pillar_vec)
    return True,pillar_vec

def rotate_node_for_pillar(G,node_unit_cell,node_pillar_fvec,pillar_vec,node_x_vecs,chain_node_fcoords):
    beginning_point = [0.0,0.0,0.0]
    _,rot,_ = superimpose([beginning_point,node_pillar_fvec],[beginning_point,pillar_vec])
    pillar_oriented_nodexvec=np.dot(np.asarray(node_x_vecs),rot)
    pillar_oriented_node_xcoords = np.dot(node_unit_cell,pillar_oriented_nodexvec.T).T
    pillar_oriented_node_fcoords = np.dot(chain_node_fcoords,rot)
    pillar_oriented_node_coords = np.dot(node_unit_cell,pillar_oriented_node_fcoords.T).T
    return pillar_oriented_node_xcoords,pillar_oriented_node_coords



if __name__ == '__main__':
    import os
    from learn_template import extract_cluster_center_from_templatecif,make_supercell_3x3x3,find_pair_v_e,add_ccoords,set_DV_V,set_DE_E
    from chainnode import process_chain_node
    template_cif_file ='MIL53templatecif.cif'
    chain_node_cif = '21Alchain.cif'
    print(os.getcwd())
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
    PILLAR,pillar_vec = check_if_pillarstack(G) 
    if not PILLAR:
        print('not pillar stack')
    else:
        chainnode_target_type = 'Al'
        node_cell_info,node_pillar_fvec, node_x_vecs, chain_node_fcoords = process_chain_node(chain_node_cif, chainnode_target_type)
        pillar_oriented_node_xcoords,pillar_oriented_node_coords = rotate_node_for_pillar(G,unit_cell,node_pillar_fvec,pillar_vec,node_x_vecs,chain_node_fcoords)