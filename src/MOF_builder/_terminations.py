import numpy as np
import networkx as nx
import re
from _superimpose import superimpose
from _filtX import filt_outside_edgex
from _cluster import exposed_Xs_Os_boundary_node

def termpdb(filename):
        inputfile = str(filename)
        with open(inputfile, "r") as fp:
            content = fp.readlines()
            #linesnumber = len(content)
        data = []
        for line in content:
            line = line.strip()
            if len(line)>0: #skip blank line
                if line[0:6] == "ATOM" or line[0:6] == "HETATM":
                    value_atom = line[12:16].strip()  # atom_label
                    #resname
                    #value2 = 'MOL'  # res_name

                    value_x = float(line[30:38])  # x
                    value_y = float(line[38:46])  # y
                    value_z = float(line[46:54])  # z
                    value_charge = float(line[61:66]) 
                    value_note = line[67:80].strip() # atom_note
                    #resnumber
                    try:
                        value_res_num = int(line[22:26])
                    except ValueError:
                        value_res_num = 1 
                    data.append([value_atom,value_charge,value_note,value_res_num,'TERM',value_res_num,value_x,value_y,value_z])
        return np.vstack(data)




def terminate_nodes(term_file,boundary_connected_nodes_res,connected_nodeedge_fc_loose,sc_unit_cell,box_bound):
    ex_node_cxo_cc_loose = exposed_Xs_Os_boundary_node(boundary_connected_nodes_res,connected_nodeedge_fc_loose,sc_unit_cell,box_bound)
    terms_loose = add_terminations(term_file,ex_node_cxo_cc_loose)
    if len(terms_loose) > 0:
        terms_cc_loose = np.vstack((terms_loose))
        return terms_cc_loose
    else:
        return np.empty((0,9),dtype='O')

def Xpdb(data,X): 
        indices=[i for i in range(len(data)) if data[i,2][0] == X]
        X_term=data[indices]
        return X_term,indices

def convert_to_tuple(array):
    return (tuple(np.round(array[0],3).flatten()), tuple(np.round(array[1],3).flatten()),tuple(np.round(array[2],3).flatten()))

def is_list_A_in_B(A, B):
    # Convert both A and B into sets of tuple representations
    A_tuples = set(convert_to_tuple(a) for a in A)
    B_tuples = set(convert_to_tuple(b) for b in B)
    # Check if all elements in A are in B
    return A_tuples.issubset(B_tuples)

def add_terminations(term_file,ex_node_cxo_cc):
    tG=nx.Graph()
    terms=[]
    node_oovecs_record=[]
    terms_append = terms.append
    node_oovecs_record_append = node_oovecs_record.append

    term_data=termpdb(term_file)
    term_info = term_data[:,:-3]
    term_coords = term_data[:,-3:]
    xterm,_=Xpdb(term_data,'X')
    oterm,_=Xpdb(term_data,'Y')
    term_xvecs= xterm[:,-3:]
    term_ovecs= oterm[:,-3:]
    term_xvecs = term_xvecs.astype('float')
    term_ovecs = term_ovecs.astype('float')
    term_coords = term_coords.astype('float')

    term_ovecs_c = np.mean(np.asarray(term_ovecs),axis=0)
    term_coords = term_coords-term_ovecs_c
    term_xoovecs =np.vstack((term_xvecs,term_ovecs))
    term_xoovecs = term_xoovecs-term_ovecs_c

    for ex in range(len(ex_node_cxo_cc)):
        node_x = ex_node_cxo_cc[ex][3]
        node_opair=ex_node_cxo_cc[ex][5]
        node_opair_c = np.mean(np.asarray(node_opair),axis=0)
        node_xoo_vecs = np.vstack([(i-node_opair_c).astype('float') for i in (node_opair+[node_x])])

        indices = [index for index, value in enumerate(node_oovecs_record) if is_list_A_in_B(node_xoo_vecs,value[0])]
        if len(indices)==1: 
            #find index of node_oo_vecs in record 
            print(f"found one,{indices}")
            rot = node_oovecs_record[indices[0]][1]
        else:
            print("new_term",term_xoovecs.shape,node_xoo_vecs.shape)
            _,rot,_, = superimpose(term_xoovecs,node_xoo_vecs)
            node_oovecs_record_append((node_xoo_vecs,rot))
        adjusted_term_vecs = np.dot(term_coords,rot) + node_opair_c
        adjusted_term = np.hstack((np.asarray(term_info),adjusted_term_vecs))
        terms_append(adjusted_term)
    return terms


def exposed_xoo_cc(eG,unsaturated_main_frag_nodes,main_frag_nodes_cc,con_nodes_x_dict,xoo_dict):

    ex_node_cxo_cc = []

    for exnode_info in unsaturated_main_frag_nodes:
        exnode_id = exnode_info[0]
        node_center_fc = eG.nodes[exnode_id]['fc']
        ex_node = main_frag_nodes_cc[main_frag_nodes_cc[:,5]==exnode_id]
        #print(list(xoo_dict) ,con_nodes_x_dict[exnode_id])
        ex_x_idx=[i for i in list(xoo_dict) if i not in con_nodes_x_dict[exnode_id]]
        for ix in ex_x_idx:
            ex_x = ex_node[ix][-3:]
            ex_x=ex_x.astype('float')
            ex_oo=ex_node[xoo_dict[ix]][:,-3:]
            ex_oo=ex_oo.astype('float')

            ex_node_cxo_cc.append((node_center_fc,len(ex_x_idx),'exposed_X',ex_x,'node_Opair',ex_oo,(ix,xoo_dict[ix])))

    return ex_node_cxo_cc

def add_node_terminations(term_file,ex_node_cxo_cc):
    tG=nx.Graph()
    terms=[]
    node_oovecs_record=[]
    terms_append = terms.append
    node_oovecs_record_append = node_oovecs_record.append

    term_data=termpdb(term_file)
    term_info = term_data[:,:-3]
    term_coords = term_data[:,-3:]
    xterm,_=Xpdb(term_data,'X')
    oterm,_=Xpdb(term_data,'Y')
    term_xvecs= xterm[:,-3:]
    term_ovecs= oterm[:,-3:]
    term_coords = term_coords.astype('float')
    term_xvecs = term_xvecs.astype('float')
    term_ovecs = term_ovecs.astype('float')

    term_ovecs_c = np.mean(np.asarray(term_ovecs),axis=0)
    term_coords = term_coords-term_ovecs_c
    term_xoovecs =np.vstack((term_xvecs,term_ovecs))
    term_xoovecs = term_xoovecs-term_ovecs_c

    for ex in range(len(ex_node_cxo_cc)):
        node_x = ex_node_cxo_cc[ex][3]
        node_opair=ex_node_cxo_cc[ex][5]
        node_opair_c = np.mean(np.asarray(node_opair),axis=0)
        node_o1=node_opair[0]
        node_o2=node_opair[1]
        node_xoo_vecs = np.vstack([node_x-node_opair_c,node_o1-node_opair_c,node_o2-node_opair_c])
        #print(term_xoovecs,node_xoo_vecs)

        indices = [index for index, value in enumerate(node_oovecs_record) if is_list_A_in_B(node_xoo_vecs,value[0])]
        if len(indices)==1: 
            #find index of node_oo_vecs in record 
            #print(f"found one,{indices}")
            rot = node_oovecs_record[indices[0]][1]
        else:
            #print(term_xoovecs,node_xoo_vecs)
            _,rot,_, = superimpose(term_xoovecs,node_xoo_vecs)
            node_oovecs_record_append((node_xoo_vecs,rot))
        adjusted_term_vecs = np.dot(term_coords,rot) + node_opair_c
        adjusted_term = np.hstack((np.asarray(term_info),adjusted_term_vecs))
        adjusted_term[:,5] = ex+1
        for row_n in range(len(adjusted_term)):
            adjusted_term[row_n,2] = re.sub('[0-9]','',adjusted_term[row_n,2]) + str(row_n+1)
        terms_append(adjusted_term)
    return terms

def exposed_x_mainfrag_edge(unsaturated_main_frag_edges,eG,edge_cc,linker_topics):
    '''look for two nearest Oxys for every exposed(unsaturated X) in unsaturated main frag edges and then add 1A away H as termination'''
    ex_edge_x=[]

    for uE in unsaturated_main_frag_edges:
        i = uE[0] #res_number
        degree_of_edges=nx.degree(eG,i)
        neighbor_nodes = list(nx.neighbors(eG,i))
        print(f'degree of edge {i} is {degree_of_edges}, neighbor nodes are {neighbor_nodes}')
        edge=edge_cc[edge_cc[:,5]==int(i[1:])]
        edge_center_cc = np.mean(edge[:,-3:],axis=0)
        edgex_indices = [ i_ex for i_ex in range(len(edge)) if edge[i_ex,2][0]=='x']        
        edgeX_indices = [ i_eX for i_eX in range(len(edge)) if edge[i_eX,2][0]=='X']
        xs_cc = edge[edgex_indices][:,-3:]
        Xs_cc = edge[edgeX_indices][:,-3:]
        Xs_cc = Xs_cc.astype(float)
        xs_cc = xs_cc.astype(float)
        #filt out 3x for tritopic linker 
        out_x_ind, _ = filt_outside_edgex(xs_cc,edge_center_cc,linker_topics)
        out_x_cc = xs_cc[out_x_ind]

        xX_pair=[]
        paired_x_ind = []
        for ix in range(len(out_x_cc)):
             x = out_x_cc[ix]
             for iX in range(len(Xs_cc)):
                X= Xs_cc[iX]
                d = np.linalg.norm(x-X)
                #print('d',d,x,X)
                if d<3.5:
                    xX_pair.append(((ix,iX),d))
                    paired_x_ind.append(ix)
                    continue
                continue
        

        exposed_outx_ind = [ix for ix in range(len(out_x_cc)) if ix not in paired_x_ind]
        #print(uE,'exposedx',exposed_outx_ind,paired_x_ind,xX_pair)
        for j in exposed_outx_ind:
            x_edge_ind = edgex_indices[out_x_ind[j]]
            exposed_x_cc = edge[x_edge_ind]
            ex_edge_x.append((edge_center_cc,uE[0],'exposed_x',x_edge_ind,exposed_x_cc[-3:]))
    return ex_edge_x


def add_edge_termination(e_termfile,ex_edge_x):
    e_term = termpdb(e_termfile)
    e_term_other_ind = []
    for k in range(len(e_term)):
        atom = e_term[k]
        if atom[2] =='X':
            e_term_X_ind = k
            #print(f'e_term_X_ind: {e_term_X_ind}')
        elif atom[2] == 'x':
            e_term_x_ind = k
            #print(f'e_term_x_ind: {e_term_x_ind}')
        else:
            e_term_other_ind.append(k)
    #print(f'e_term_other_ind{e_term_other_ind}')
    e_term_other_ind.append(e_term_X_ind)
    e_term_x_cc= e_term[e_term_x_ind][-3:]
    e_term_X_cc= e_term[e_term_X_ind][-3:]
    e_term_otherXcoor_cc = e_term[e_term_other_ind][:,-3:]

    e_term_x_cc = e_term_x_cc.astype('float')
    e_term_X_cc = e_term_X_cc.astype('float')
    e_term_otherXcoor_cc = e_term_otherXcoor_cc.astype('float')

    e_term_xX = e_term_X_cc-e_term_x_cc
    norm_e_term_xX = e_term_xX/np.linalg.norm(e_term_xX)
    norm_e_term_xX = norm_e_term_xX.reshape((1,3))

    e_term_xN = np.vstack([e_term_x_cc,e_term_x_cc+norm_e_term_xX])

    edge_terms = []
    for i_eex in range(len(ex_edge_x)):
        #in edge:
        eex_cen_cc = ex_edge_x[i_eex][0]
        ex_cc = ex_edge_x[i_eex][-1]
        eex_cen_cc = eex_cen_cc.astype('float')
        ex_cc = ex_cc.astype('float')

        edge_xX_cc = ex_cc-eex_cen_cc
        norm_edge_xX_cc = edge_xX_cc/np.linalg.norm(edge_xX_cc)
        norm_edge_xX_cc = norm_edge_xX_cc.reshape((1,3))
        beginning = np.asarray([0.0,0.0,0.0])
        beginning = beginning.reshape((1,3))
        
        edge_xN = ex_cc+norm_edge_xX_cc
        e_edge_xN = np.vstack([ex_cc,edge_xN])
        #_,rot,trans = superimpose(e_term_xN,e_edge_xN)
        v_term = np.vstack((beginning,norm_e_term_xX))
        v_edge = np.vstack((beginning,norm_edge_xX_cc))
        #print(v_term.shape,v_edge.shape)
        _,rot,trans = superimpose(v_term,v_edge)
        flip_matrix = np.array(([-1,0,0],[0,-1,0],[0,0,-1]))
        edge_term = e_term[e_term_other_ind]
        rotated_norm_e_term = np.dot(norm_e_term_xX,rot)
        rotated_norm_e_term = rotated_norm_e_term.reshape((3,1))
        #print(np.dot(norm_edge_xX_cc,rotated_norm_e_term)[0,0]>0)
        if np.dot(norm_edge_xX_cc,rotated_norm_e_term)[0,0] > 0:
            #bottom--up then flip it 
           edge_term[:,-3:] = np.dot(e_term_otherXcoor_cc-e_term_x_cc,rot)+ex_cc

        else:
            edge_term[:,-3:] = np.dot(np.dot(e_term_otherXcoor_cc-e_term_x_cc,rot),flip_matrix)+ex_cc

        edge_term[:,4] = 'HEDGE'
        edge_term[:,5] = int(ex_edge_x[i_eex][1][1:])

        edge_terms.append(edge_term)

    return edge_terms,len(e_term_other_ind)



def terminate_unsaturated_edges(e_termfile,unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics):
    ex_edge_x = exposed_x_mainfrag_edge(unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics)
    unsaturated_edges_idx = [int(ue[0][1:]) for ue in unsaturated_main_frag_edges]
    if len(unsaturated_edges_idx) > 0:
        edge_terms_cc,e_term_atom_num = add_edge_termination(e_termfile,ex_edge_x)
        edge_term_dict = {}
        edge_term_cc_arr = np.vstack(edge_terms_cc)
        for idx in unsaturated_edges_idx:
            idx_term_arr = edge_term_cc_arr[edge_term_cc_arr[:,5]==str(idx)]
            edge_term_dict[idx] = idx_term_arr

        sa_edges=main_frag_edges_cc[~np.isin(main_frag_edges_cc[:, 5], unsaturated_edges_idx)] 
        usa_edges = main_frag_edges_cc[np.isin(main_frag_edges_cc[:, 5], unsaturated_edges_idx)] 

        t_usa_edges = []
        for i_exedge in unsaturated_edges_idx:
                exedge=usa_edges[usa_edges[:,5]==i_exedge]
                t_edge_lines = edge_term_dict[i_exedge]
                #print(i_exedge,len(t_edge_lines))
                t_usa_edge=np.vstack((exedge,t_edge_lines))
                t_usa_edge[:,5]=int(i_exedge)
                if int(len(t_edge_lines)/e_term_atom_num)==1:
                    t_usa_edge[:,4]='HEDGE'
                elif int(len(t_edge_lines)/e_term_atom_num)==2:
                    t_usa_edge[:,4]='HHEDGE'
                elif int(len(t_edge_lines)/e_term_atom_num)==3:
                    t_usa_edge[:,4]='HHHEDGE'
                for row_n in range(len(t_usa_edge)):
                    t_usa_edge[row_n,2] = re.sub('[0-9]','',t_usa_edge[row_n,2])+str(row_n+1)
                t_usa_edges.append(t_usa_edge)
        t_usa_edges_arr = np.vstack(t_usa_edges)
        t_edges=np.vstack((sa_edges,t_usa_edges_arr))
    
        return t_edges
    else:
        return main_frag_edges_cc
    
def terminate_unsaturated_edges_CCO2(e_termfile,unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics):
    ex_edge_x = exposed_x_mainfrag_edge(unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics)
    unsaturated_edges_idx = [int(ue[0][1:]) for ue in unsaturated_main_frag_edges]
    if len(unsaturated_edges_idx) > 0:
        edge_terms_cc,e_term_atom_num = add_edge_termination(e_termfile,ex_edge_x)
        edge_term_dict = {}
        edge_term_cc_arr = np.vstack(edge_terms_cc)
        for idx in unsaturated_edges_idx:
            idx_term_arr = edge_term_cc_arr[edge_term_cc_arr[:,5]==str(idx)]
            edge_term_dict[idx] = idx_term_arr

        sa_edges=main_frag_edges_cc[~np.isin(main_frag_edges_cc[:, 5], unsaturated_edges_idx)] 
        usa_edges = main_frag_edges_cc[np.isin(main_frag_edges_cc[:, 5], unsaturated_edges_idx)] 

        t_usa_edges = []
        for i_exedge in unsaturated_edges_idx:
                exedge=usa_edges[usa_edges[:,5]==i_exedge]
                t_edge_lines = edge_term_dict[i_exedge]
                #print(i_exedge,len(t_edge_lines))
                t_usa_edge=np.vstack((exedge,t_edge_lines))
                t_usa_edge[:,5]=int(i_exedge)
                if int(len(t_edge_lines)/e_term_atom_num)==1:
                    t_usa_edge[:,4]='EDGE'
                elif int(len(t_edge_lines)/e_term_atom_num)==2:
                    t_usa_edge[:,4]='EDGE'
                elif int(len(t_edge_lines)/e_term_atom_num)==3:
                    t_usa_edge[:,4]='EDGE'
                for row_n in range(len(t_usa_edge)):
                    t_usa_edge[row_n,2] = re.sub('[0-9]','',t_usa_edge[row_n,2])+str(row_n+1)
                t_usa_edges.append(t_usa_edge)
        t_usa_edges_arr = np.vstack(t_usa_edges)
        t_edges=np.vstack((sa_edges,t_usa_edges_arr))
        return t_edges
    else:
        return main_frag_edges_cc