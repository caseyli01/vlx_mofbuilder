import numpy as np
from _supercell import Carte_points_generator
from _cluster import split_diffs
import networkx as nx
import re

def pname(s):
    #return primitive_cell_vertex_node_name
    #extract V1 from 'V1_[-1.  0.  0.]'
    return s.split('_')[0]
def lname(s):
    #return array format of list of super node name
    #extract [-1.  0.  0.] from 'V1_[-1.  0.  0.]'
    if len(s.split('_'))<2:
        lis = np.array([0.,0.,0.])
    else:
        lis = np.asanyarray(s.split('_')[1][1:-1].split(),dtype=float)
    return lis


def replace_sG_dvnode_with_vnode(sG,diff_e,dvnode,vnode):
    #make name

    sG.add_node(vnode+'_'+str(diff_e),f_points = np.hstack((sG.nodes[vnode]['f_points'][:,0:2],sG.nodes[vnode]['f_points'][:,2:5].astype(float)+diff_e)), #NOTE:modified because of extra column of atom type
                            fcoords = sG.nodes[vnode]['fcoords']+diff_e,
                            type='SV',
                            note=sG.nodes[vnode]['note'])
    #process edge linked to dvnode and relink them to vnode_diff_e
    edges = [e for e in sG.edges(dvnode)]
    for e in edges:
        sG.add_edge(vnode+'_'+str(diff_e),e[1],fcoords=sG.edges[e]['fcoords'],
                    f_points=sG.edges[e]['f_points'],fc_center=sG.edges[e]['fc_center'],type=sG.edges[e]['type'])
        sG.remove_edge(e[0],e[1])
    sG.remove_node(dvnode)
    return sG

#find DV cooresponding to V and update the sG, remove dvnode and add vnode with diff_e
def replace_DV_with_corresponding_V(sG):
    #use distance matrix for moded_DV and V
    moded_dv_fc = []
    v_fc = []
    for n in sG.nodes():
        if sG.nodes[n]['type']=='DV':
            moded_dv_fc.append((n,np.mod(sG.nodes[n]['fcoords'],1)))
        else:
            v_fc.append((n,sG.nodes[n]['fcoords']))

    dist_matrix = np.zeros((len(moded_dv_fc),len(v_fc)))
    for i in range(len(moded_dv_fc)):
        for j in range(len(v_fc)):
            dist_matrix[i,j] = np.linalg.norm(moded_dv_fc[i][1]-v_fc[j][1])
   
    #check the distance is less than 1e-2
    dv_v_pairs = []
    for k in range(len(moded_dv_fc)):
        if np.min(dist_matrix[k,:]) < 1e-2:
            corresponding_v = np.argmin(dist_matrix[k,:])
            dvnode = moded_dv_fc[k][0]
            vnode = v_fc[corresponding_v][0]
            diff_e = sG.nodes[dvnode]['fcoords']-sG.nodes[vnode]['fcoords']
            dv_v_pairs.append((dvnode,vnode+'_'+str(diff_e)))
            sG = replace_sG_dvnode_with_vnode(sG,diff_e,moded_dv_fc[k][0],v_fc[corresponding_v][0])
        else:
            print('distance is larger than 1e-2',np.min(dist_matrix[k,:]) ,moded_dv_fc[k][0],v_fc[np.argmin(dist_matrix[k,:])][0])
    return dv_v_pairs,sG


#check if node is at the boundary of the supercell
#if at boundary, then add the diff_e to the node name and add the diff_e to the f_points
def update_supercell_node_fpoints_loose(sG,supercell):
   #boundary_node_res = []
    #incell_node_res = []	
    superG = nx.Graph()    
    for n in sG.nodes():
        if sG.nodes[n]['type']=='SV': #get rid of SV, sv will be pnode+i
            superG.add_node(n,f_points = sG.nodes[n]['f_points'],
                            fcoords = sG.nodes[n]['fcoords'],
                            type='SV',
                            note=sG.nodes[n]['note'])
            
                       
            continue

        #add the node to superG, if lname(n) is np.array([0,0,0])
        superG.add_node(n+'_'+str(lname(n)),f_points = sG.nodes[n]['f_points'],
                            fcoords = sG.nodes[n]['fcoords'],
                            type='V',
                            note=sG.nodes[n]['note'])

        arr = sG.nodes[n]['f_points'][:,2:5].astype(float)+supercell #NOTE:modified because of extra column of atom type
        moded_arr = np.mod(arr,1)
        arr = arr.astype(float)
        moded_arr = moded_arr.astype(float)
        row_diff = [i for i in range(len(arr)) if not np.allclose(arr[i],moded_arr[i])]
        diff = [arr[i]-moded_arr[i] for i in row_diff]

        diffs=np.round(np.unique(diff,axis=0),1)
        diff_ele = split_diffs(diffs)
       #if len(diff_ele) > supercell_Carte.shape[0]:
       #    boundary_node_res.append(n)
       #else:
       #    incell_node_res.append(n)    

        for diff_e in diff_ele:
            diff_e = np.asarray(diff_e)
            if (n+'_'+str(diff_e)) in superG.nodes():
                print('node already in superG',n+'_'+str(diff_e))
                continue
            superG.add_node((n+'_'+str(diff_e)),
                            f_points = np.hstack((sG.nodes[n]['f_points'][:,0:2],sG.nodes[n]['f_points'][:,2:5].astype(float)+diff_e)), #NOTE:modified because of extra column of atom type
                            fcoords = sG.nodes[n]['fcoords']+diff_e,
                            type='SV',
                            note=sG.nodes[n]['note'])
            
    return superG#,boundary_node_res,incell_node_res

'''
sG_node_note_set:{'CV', 'V'}
sG_node_type_set:{'DV', 'V'}
sG_edge_type_set:{'DE', 'E'}
'''

#check if edge is at the boundary of the supercell
def update_supercell_edge_fpoints(sG,superG,supercell):
    #boundary edge is DE
    #incell edge is E
    supercell_Carte = Carte_points_generator(supercell)		
    for e in sG.edges():
        for i in supercell_Carte:
            s_edge = (pname(e[0])+'_'+str(i+lname(e[0])),pname(e[1])+'_'+str(i+lname(e[1])))

            #check if node e[0]+'_'+str(diff_e) and e[1]+'_'+str(diff_e) in superG
            if (s_edge[0] in superG.nodes()) and (s_edge[1] in superG.nodes()):
                superG.add_edge(s_edge[0],s_edge[1],
                                f_points=np.hstack((sG.edges[e]['f_points'][:,0:2],sG.edges[e]['f_points'][:,2:5].astype(float)+i)),#NOTE:modified because of extra column of atom type
                                fcoords=sG.edges[e]['fcoords']+i,
                                type=sG.edges[e]['type'])
                
            elif (s_edge[0] in superG.nodes()) or (s_edge[1] in superG.nodes()):
                if s_edge[0] in superG.nodes():
                    superG.add_node(s_edge[1],
                                    f_points=np.hstack((sG.nodes[e[1]]['f_points'][:,0:2],sG.nodes[e[1]]['f_points'][:,2:5].astype(float)+i)),#NOTE:modified because of extra column of atom type
                                    fcoords=sG.nodes[e[1]]['fcoords']+i,
                                    type='DSV',
                                    note=sG.nodes[e[1]]['note'])
                    
                else:
                    superG.add_node(s_edge[0],
                                    f_points=np.hstack((sG.nodes[e[0]]['f_points'][:,0:2],sG.nodes[e[0]]['f_points'][:,2:5].astype(float)+i)),#NOTE:modified because of extra column of atom type
                                    fcoords=sG.nodes[e[0]]['fcoords']+i,
                                    type='DSV',
                                    note=sG.nodes[e[0]]['note'])
                    
                superG.add_edge(s_edge[0],s_edge[1],
                                f_points=np.hstack((sG.edges[e]['f_points'][:,0:2],sG.edges[e]['f_points'][:,2:5].astype(float)+i)),#NOTE:modified because of extra column of atom type
                                fcoords=sG.edges[e]['fcoords']+i,
                                type='DSE')
                    
    return superG

########## the below is to process the multiedge bundling in superG###########
#need to combine with the multiedge_bundling.py
#replace bundle dvnode with vnode+diff_e
def replace_bundle_dvnode_with_vnode(dv_v_pairs,multiedge_bundlings):
    for dv,v in dv_v_pairs:
        for bund in multiedge_bundlings:
            if dv in bund[1]:
                bund[1][bund[1].index(dv)]=v
            if dv in bund[0]:
                bund[0][bund[0].index(dv)]=v
    #update v if no list then add [0,0,0]
    #convert tuple to list
    updated_bundlings = []
    for bund in multiedge_bundlings:
        ec_node = pname(bund[0])+'_'+str(lname(bund[0]))
        con_nodes = [pname(i)+'_'+str(lname(i)) for i in bund[1]]
        updated_bundlings.append((ec_node,con_nodes))
    return updated_bundlings

#loop bundle and check if any element in the bundle is in the superG, if not, add the element to the superG
def make_super_multiedge_bundlings(prim_multiedge_bundlings,supercell):
    super_multiedge_bundlings = {}
    for i in Carte_points_generator(supercell):
        for bund in prim_multiedge_bundlings:
            ec_node = pname(bund[0])+'_'+str(i+lname(bund[0]))
            con_nodes = [pname(n)+'_'+str(i+lname(n)) for n in bund[1]]
            super_multiedge_bundlings[ec_node]=con_nodes
    return super_multiedge_bundlings


def update_supercell_bundle(superG,super_multiedge_bundlings):
    for ec_node in super_multiedge_bundlings.keys():
        con_nodes = super_multiedge_bundlings[ec_node]
        prim_ecname = pname(ec_node) + '_'+str(np.array([0.,0.,0.]))
        if ec_node not in superG.nodes():
            trans = lname(ec_node)
            superG.add_node(ec_node,f_points=np.hstack((superG.nodes[prim_ecname]['f_points'][:,0:2], #NOTE:modified because of extra column of atom type
                                                        superG.nodes[prim_ecname]['f_points'][:,2:5].astype(float)+trans)),#NOTE:modified because of extra column of atom type
                            fcoords=superG.nodes[prim_ecname]['fcoords']+trans,
                            type='SV',
                            note=superG.nodes[prim_ecname]['note'])
        for j in range(len(con_nodes)):
            cn = con_nodes[j]
            prim_cnname = super_multiedge_bundlings[prim_ecname][j] #find prim_ecname in super_multiedge_bundlings and then get the corresponding prim_cnname
            trans = lname(cn)-lname(prim_cnname)
            if cn not in superG.nodes():
                superG.add_node(cn,f_points=np.hstack((superG.nodes[prim_cnname]['f_points'][:,0:2], #NOTE:modified because of extra column of atom type
                                                       superG.nodes[prim_cnname]['f_points'][:,2:5].astype(float)+trans)), #NOTE:modified because of extra column of atom type
                                fcoords=superG.nodes[prim_cnname]['fcoords']+trans,
                                type='SV',
                                note=superG.nodes[prim_cnname]['note'])
                superG.add_edge(ec_node,cn,f_points=np.hstack((superG.edges[prim_ecname,prim_cnname]['f_points'][:,0:2], #NOTE:modified because of extra column of atom type
                                                               superG.edges[prim_ecname,prim_cnname]['f_points'][:,2:5].astype(float)+trans)), #NOTE:modified because of extra column of atom type
                                fcoords=superG.edges[prim_ecname,prim_cnname]['fcoords']+trans,
                                type='DSE')  
                

    return superG


def check_multiedge_bundlings_insuperG(super_multiedge_bundlings,superG):
    super_multiedge_bundlings_edges = []
    for ec_node in super_multiedge_bundlings:
        #check is all CV node in superG are in the super_multiedge_bundlings_edges first element
        cvnodes = [n for n in superG.nodes() if superG.nodes[n]['note']=='CV']
        #use set to check if all cvnodes are in the super_multiedge_bundlings_edges
        if set(cvnodes) == set([i[0] for i in super_multiedge_bundlings_edges]):
            return superG
        else:
            print('not all CV nodes in super_multiedge_bundlings_edges')
            diff_element = set(cvnodes).difference(set(list(super_multiedge_bundlings)))
            print('to remove diff_element',diff_element)
            #remove the diff_element from the superG
            for n in diff_element:
                superG.remove_node(n)
                #remove all edges linked to the node
                edges = [e for e in superG.edges(n)]
                for e in edges:
                    superG.remove_edge(e[0],e[1])
            
            return superG

########## the above is to process the multiedge bundling in superG###########



def locate_min_idx(matrix):
        min_idx = np.unravel_index(matrix.argmin(), matrix.shape)
        return min_idx[0],min_idx[1]

#remove charater from string
def nl(s):
    return int(re.sub('a-zA-Z','',s))

        

def add_virtual_edge(superG,bridge_node_distance,max_neighbor=2):
        #add pillar nodes virtual edges
    nodes_list = [n for n in superG.nodes() if superG.nodes[n]['note']=='V']
    n_n_distance_matrix = np.zeros((len(nodes_list),len(nodes_list)))
    
    for i in range(len(nodes_list)):
        for j in range(len(nodes_list)):
            n_n_distance_matrix[i,j] = np.linalg.norm(superG.nodes[nodes_list[i]]['fcoords'] - superG.nodes[nodes_list[j]]['fcoords'])
        n_n_distance_matrix[i,i] = 1000
    #use hungrain algorithm to find the shortest path between all nodes
    

    for i in range(len(nodes_list)):
        neighbor_count = 0
        while neighbor_count < max_neighbor:
            def add_virtual_edge(i,n_n_distance_matrix,superG,bridge_node_distance,count):
                n_n_min_distance = np.min(n_n_distance_matrix[i:i+1,:])
                if n_n_min_distance < bridge_node_distance:
                    _,n_j = locate_min_idx(n_n_distance_matrix[i:i+1,:])
                    superG.add_edge(nodes_list[i],nodes_list[n_j],type='virtual')
                    #print('add virtual edge between',nodes_list[i],nodes_list[n_j])
                    n_n_distance_matrix[i,n_j] = 1000
                    return True, count+1,n_n_distance_matrix,superG
                else:
                    return False, count,n_n_distance_matrix,superG
            added,neighbor_count,n_n_distance_matrix,superG = add_virtual_edge(i,n_n_distance_matrix,superG,bridge_node_distance,neighbor_count)
            if not added:
                break

    return superG