import numpy as np
import re
import networkx as nx

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
#save eG to gro
def temp_save_eG_TERM_gro(eG,sc_unit_cell,unsaturated_vnode_xoo_dict):
    with open('eG_TERM.gro','w') as f:
        newgro = []
        newgro.append('eG_TERM\n')
        newgro.append('17470\n')
        res_num = 0
        line_num = 0
        for n in eG.nodes():
            #if pname(n) != 'EDGE':
                #continue

            postions = eG.nodes[n]['f_points']
            res_num+=1
            fc = np.asarray(postions[:,1:4],dtype=float)
            cc = np.dot(sc_unit_cell,fc.T).T
            positionss = np.hstack((postions[:,0:1],cc))
            for line in positionss:
                line_num+=1
                value_atom_number = int(line_num)  # atom_number
                value_label = re.sub('\d','',line[0]) # atom_label
                if 'X' in value_label:
                    value_label = 'C'
                elif 'x' in value_label:
                    value_label = 'C'
                    
                value_resname = str(n)[0:1]+str(eG.nodes[n]['index'])  # residue_name
                value_resnumber = int(res_num) # residue number
                value_x = 0.1*float(line[1])  # x
                value_y = 0.1*float(line[2])  # y
                value_z = 0.1*float(line[3])  # z
                formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
                    value_resnumber,
                    value_resname,
                    value_label,
                    value_atom_number,
                    value_x,
                    value_y,
                    value_z,
                )
                newgro.append(formatted_line + "\n")
        for k,v in unsaturated_vnode_xoo_dict.items():
            res_num+=1
            for line in v['node_term_c_points']:
                line_num+=1
                value_atom_number = int(line_num)
                value_label = re.sub('\d','',line[0])
                value_resname = 'T'+str(k[0])[0:2]
                value_resnumber = int(res_num)
                value_x = 0.1*float(line[1])
                value_y = 0.1*float(line[2])
                value_z = 0.1*float(line[3])
                formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
                    value_resnumber,
                    value_resname,
                    value_label,
                    value_atom_number,
                    value_x,
                    value_y,
                    value_z,
                )
                newgro.append(formatted_line + "\n")
        tail = "10 10 10 \n"
        newgro.append(tail)
        f.writelines(newgro)