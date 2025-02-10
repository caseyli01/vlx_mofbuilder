import numpy as np
import networkx as nx
import re 



def fetch_X_atoms_ind_array(array,column,X):
    ind = [k for k in range(len(array)) if re.sub(r'\d','',array[k,column]) == X]
    x_array= array[ind]
    return ind,x_array


def get_rad_v1v2(v1,v2):
    cos_theta = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if cos_theta ==0:
        return 0
    else:
        rad = np.arccos(cos_theta)
        return rad

def filt_closest_x_angle(Xs_fc,edge_center_fc,node_center_fc):
    rds_list = []
    rads = []
    dists = []
    x_number = len(Xs_fc)
    half_x_number = int(0.5*x_number)
    for i in range(x_number):
        rad = get_rad_v1v2(Xs_fc[i]-edge_center_fc,node_center_fc-edge_center_fc)
        dist = np.linalg.norm(Xs_fc[i]-edge_center_fc)
        rds_list.append((i,rad,dist))
        rads.append(rad)
        dists.append(dist)
    rads.sort()
    dists.sort()
    x_idx=[i[0] for i in rds_list if (i[1]<0.6 and i[2]<dists[half_x_number])] #0.6 rad == 35degree
    x_info=[i for i in rds_list if (i[1]<0.6 and i[2]<dists[half_x_number])] #0.6 rad == 35degree
    if len(x_idx)==1:
        return x_idx,x_info
    elif len(x_idx)>1:
        min_d = min([j[2] for j in x_info])
        x_idx1=[i[0] for i in rds_list if  i[2]==min_d]
        x_info1=[i for i in rds_list if i[2]==min_d]
        return x_idx1,x_info1
    else:
        print("ERROR cannot find connected X")
        print(rds_list)

def filt_close_edgex(Xs_fc,edge_center_fc,linker_topics):
    '''
	find closest X for edge_center
	return the indices and distance
	'''
    lcs_list = []
    lcs = []
    for i in range(len(Xs_fc)):
        lc = np.linalg.norm(Xs_fc[i]-edge_center_fc)
        lcs_list.append((i,lc))
        lcs.append(lc)
    lcs.sort()
    outside_edgex_indices=[i[0] for i in lcs_list if i[1]<lcs[linker_topics]]
    outside_edgex_ind_dist=[i for i in lcs_list if i[1]<lcs[linker_topics]]
    return outside_edgex_indices,outside_edgex_ind_dist


def xoo_pair_ind_node(main_frag_nodes_fc,node_id,sc_unit_cell):
    xoo_ind_node = []
    single_node_fc=main_frag_nodes_fc[main_frag_nodes_fc[:,5]==node_id]
    single_node = np.hstack((single_node_fc[:,:-3],np.dot(sc_unit_cell,single_node_fc[:,-3:].T).T))
    xind,xs=fetch_X_atoms_ind_array(single_node,2,'X')
    oind,os=fetch_X_atoms_ind_array(single_node,2,'O')
    for i in xind:
        x = single_node[i][-3:]
        pair_oind,pair_o_info=filt_close_edgex(os[:,-3:],x,2)
        #pair_oind=[oind[j[0]] for j in pair_o_info]
        xoo_ind_node.append((i,[oind[po] for po in pair_oind]))
    return xoo_ind_node


def replace_Xbyx(single_edge):
    for i in range(len(single_edge)):
        if single_edge[i,2][0]=='X':
            single_edge[i,2] = re.sub('X','x',single_edge[i,2])
    return single_edge


def filt_xcoords(single_edge):
    xcoords=[]
    for i in range(len(single_edge)):
        if single_edge[i,2][0]=='X':
            single_edge_xcoord= single_edge[i,-3:]
            xcoords.append(single_edge_xcoord)
    return xcoords

def correct_neighbor_nodes_order_by_edge_xs_order(eG,edge_n,single_edge):
    neighbor_nodes = list(nx.neighbors(eG,edge_n))
    for inn in neighbor_nodes:
            c_nn = eG.nodes[inn]['fc']
    a = filt_xcoords(single_edge) #edge xs orders
    if len(a)==6:
          a=a[-3:]
    if len(a)==8:
          a=a[-4:]
          
    b=neighbor_nodes # neighbor node->_center fc
    ordered_neinodes=[]
    for xc_i in range(len(a)):
        min_l=100

        for n in b:
            value = eG.nodes[n]['fc']
            l = np.linalg.norm(value-a[xc_i])
            if l<min_l:
                min_l = l
                near_node = n
        
        ordered_neinodes.append(near_node) if near_node not in ordered_neinodes else  None
    return ordered_neinodes

def addxoo2edge(eG,main_frag_nodes,main_frag_nodes_fc,main_frag_edges,main_frag_edges_fc,sc_unit_cell):

    xoo_ind_node0 = xoo_pair_ind_node(main_frag_nodes_fc,main_frag_nodes[0],sc_unit_cell)
    xoo_ind_node1 = xoo_pair_ind_node(main_frag_nodes_fc,main_frag_nodes[1],sc_unit_cell)
    if xoo_ind_node0 == xoo_ind_node1:
        xoo_dict={}
        for xoo in xoo_ind_node0:
            xoo_dict[xoo[0]]=xoo[1]

    con_nodes_x_dict= {}
    xoo_main_frag_edge_fc=[]
    for i in main_frag_edges:
        cons_fc=[]
        #degree_of_edge=nx.degree(eG,i)
        #neighbor_nodes = list(nx.neighbors(eG,i))
        single_edge = main_frag_edges_fc[main_frag_edges_fc[:,5]==int(i[1:])]
        #print(i,neighbor_nodes,eG.nodes[i]['fc'])
        c_edge=eG.nodes[i]
        c_edge_fc = c_edge['fc']
        neighbor_nodes=correct_neighbor_nodes_order_by_edge_xs_order(eG,i,single_edge)
        single_edge = replace_Xbyx(single_edge)
        #print(neighbor_nodes,'neighbor_nodes')
        for inn in neighbor_nodes:
            c_nn = eG.nodes[inn]
            c_nn_fc = c_nn['fc']
            single_node = main_frag_nodes_fc[main_frag_nodes_fc[:,5]==inn]
            xind,xs=fetch_X_atoms_ind_array(single_node,2,'X') # filt X atoms indices and coords in this neighbor node
            con_x,con_x_info=filt_closest_x_angle(xs[:,-3:],c_edge_fc,c_nn_fc) #filt closest X atoms and info
            #print(i,con_x,xind[con_x[0]],neighbor_nodes)
            con_x_id=xind[con_x[0]]
            con_x_oo_id=xoo_dict[con_x_id]
            con_x_fc = single_node[con_x_id]
            con_o1_fc = single_node[con_x_oo_id[0]]
            con_o2_fc = single_node[con_x_oo_id[1]]
            con_xoo_fc = np.vstack((con_x_fc,con_o1_fc,con_o2_fc))
            con_xoo_fc[:,4] = 'EDGE'
            con_xoo_fc[:,5] = int(i[1:])
            cons_fc.append(con_xoo_fc)
            cons_fc_arr=np.vstack(cons_fc)
            if inn in con_nodes_x_dict.keys():
                con_nodes_x_dict[inn].append(con_x_id)
            else:
                con_nodes_x_dict[inn] = [con_x_id]
        single_edge_con_xoo = np.vstack((single_edge,cons_fc_arr))
        row_n = None
        for row_n in range(len(single_edge_con_xoo)):
            single_edge_con_xoo[row_n,2] = re.sub('[0-9]','',single_edge_con_xoo[row_n,2])+str(row_n+1)
        xoo_main_frag_edge_fc.append(single_edge_con_xoo)

    return np.vstack(xoo_main_frag_edge_fc),xoo_dict,con_nodes_x_dict
