import numpy as np
import re
import networkx as nx
from _place_node_edge import fractional_to_cartesian
from makesuperG import pname


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

def check_supercell_box_range(point,supercell,buffer):
    #to cleave eG to supercell box

    supercell_x = supercell[0]+buffer
    supercell_y = supercell[1]+buffer  
    supercell_z = supercell[2]+buffer
    if point[0] >= 0 and point[0] <= supercell_x and point[1] >= 0 and point[1] <= supercell_y and point[2] >= 0 and point[2] <= supercell_z:
        return True
    else:
        #print(point, 'out of supercell box range:  [',supercell_x,supercell_y,supercell_z, '],   will be excluded') #debug
        return False

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

def make_unsaturated_vnode_xoo_dict(unsaturated_node,xoo_dict,matched_vnode_xind,eG,sc_unit_cell):
        """
        make a dictionary of the unsaturated node and the exposed X connected atom index and the corresponding O connected atoms
        """
    
        #process matched_vnode_xind make it to a dictionary
        matched_vnode_xind_dict = {}
        for [k,v] in matched_vnode_xind:
            if k in matched_vnode_xind_dict.keys():
                matched_vnode_xind_dict[k].append(v)
            else:
                matched_vnode_xind_dict[k] = [v]
       

        unsaturated_vnode_xind_dict ={}
        xoo_keys = list(xoo_dict.keys())
        #for each unsaturated node, get the upmatched x index and xoo atoms
        for unsat_v in unsaturated_node:
            if unsat_v in matched_vnode_xind_dict.keys():
                unsaturated_vnode_xind_dict[unsat_v] = [i for i in xoo_keys if i not in matched_vnode_xind_dict[unsat_v]]
                #print(unsaturated_vnode_xind_dict[unsat_v],'unsaturated_vnode_xind_dict[unsat_v]') #DEBUG
            else:
                unsaturated_vnode_xind_dict[unsat_v] = xoo_keys
        
        #based on the unsaturated_vnode_xind_dict, add termination to the unsaturated node xoo
        #loop over unsaturated nodes, and find all exposed X atoms and use paied xoo atoms to form a termination
        unsaturated_vnode_xoo_dict = {}
        for vnode,exposed_x_indices in unsaturated_vnode_xind_dict.items():
            for xind in exposed_x_indices:  
                x_fpoints = eG.nodes[vnode]['f_points'][xind]
                x_cpoints = np.hstack((x_fpoints[0:2],fractional_to_cartesian(x_fpoints[2:5],sc_unit_cell))) #NOTE: modified add the atom type and atom name
                oo_ind_in_vnode = xoo_dict[xind]
                oo_fpoints_in_vnode = [eG.nodes[vnode]['f_points'][i] for i in oo_ind_in_vnode]
                oo_fpoints_in_vnode = np.vstack(oo_fpoints_in_vnode)
                oo_cpoints = np.hstack((oo_fpoints_in_vnode[:,0:2],fractional_to_cartesian(oo_fpoints_in_vnode[:,2:5],sc_unit_cell)))#NOTE: modified add the atom type and atom name

                unsaturated_vnode_xoo_dict[(vnode,xind)] = {'xind':xind,'oo_ind':oo_ind_in_vnode,
                                                    'x_fpoints':x_fpoints,
                                                    'x_cpoints':x_cpoints,
                                                    'oo_fpoints':oo_fpoints_in_vnode,
                                                    'oo_cpoints':oo_cpoints}
                

        return unsaturated_vnode_xind_dict,unsaturated_vnode_xoo_dict,matched_vnode_xind_dict

#functions for write 
# write gro file
def extract_node_edge_term(tG,sc_unit_cell):
    nodes_tG = []
    terms_tG = []
    edges_tG = []
    node_res_num = 0
    term_res_num = 0
    edge_res_num = 0
    nodes_check_set = set()
    edges_check_set = set()
    terms_check_set = set()
    for n in tG.nodes():
        if pname(n) != 'EDGE':
            postions = tG.nodes[n]['noxoo_f_points']
            nodes_check_set.add(len(postions))
            if len(nodes_check_set) >1:
                raise ValueError('node index is not continuous')
            node_res_num+=1
            nodes_tG.append(np.hstack((np.tile(np.array([node_res_num,'NODE']), (len(postions), 1)), #residue number and residue name
                                       postions[:, 1:2], #atom type (element)
                                       fractional_to_cartesian(postions[:, 2:5], sc_unit_cell), #Cartesian coordinates
                                       postions[:, 0:1], #atom name
                                       np.tile(np.array([n]), (len(postions), 1))))) #node name in eG is added to the last column
            for term_ind_key,c_positions in tG.nodes[n]['term_c_points'].items():
                terms_check_set.add(len(c_positions))
                if len(terms_check_set) >1:
                    raise ValueError('term index is not continuous')

                term_res_num+=1
                terms_tG.append(np.hstack((np.tile(np.array([term_res_num,'TERM']), (len(c_positions), 1)),  #residue number and residue name
                                           c_positions[:, 1:2], #atom type (element)
                                           c_positions[:, 2:5], #Cartesian coordinates
                                           c_positions[:, 0:1], #atom name
                                           np.tile(np.array([term_ind_key]), (len(c_positions), 1))))) #term name in eG is added to the last column

            
        elif pname(n) == 'EDGE':
            postions = tG.nodes[n]['f_points']
            edges_check_set.add(len(postions))
            if len(edges_check_set) >1:
                print(edges_check_set)
                raise ValueError('edge atom number is not continuous')
            edge_res_num+=1
            edges_tG.append(np.hstack((np.tile(np.array([edge_res_num,'EDGE']), (len(postions), 1)), #residue number and residue name
                                        postions[:, 1:2], #atom type (element)
                                        fractional_to_cartesian(postions[:, 2:5], sc_unit_cell), #Cartesian coordinates
                                        postions[:, 0:1], #atom name
                                        np.tile(np.array([n]), (len(postions), 1))))) #edge name in eG is added to the last column

    #nodes_tG = np.vstack(nodes_tG)
    #terms_tG = np.vstack(terms_tG)
    #edges_tG = np.vstack(edges_tG)
    return nodes_tG,edges_tG,terms_tG,node_res_num,edge_res_num,term_res_num



def convert_node_array_to_gro_lines(array,line_num_start,res_num_start,name):
    formatted_gro_lines = []
    for i in range(len(array)):
        line = array[i]
        ind_inres = i+1
        value_atom_number_in_gro = int(ind_inres+line_num_start)  # atom_number
        value_label = re.sub('\d','',line[2])+str(ind_inres) # atom_label       
        value_resname = str(name)[0:3]#+str(eG.nodes[n]['index'])  # residue_name
        value_resnumber = int(res_num_start+int(line[0])) # residue number
        value_x = 0.1*float(line[3])  # x
        value_y = 0.1*float(line[4])  # y
        value_z = 0.1*float(line[5])  # z
        formatted_line = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
            value_resnumber,
            value_resname,
            value_label,
            value_atom_number_in_gro,
            value_x,
            value_y,
            value_z,
        )
        formatted_gro_lines.append(formatted_line+"\n")
    return formatted_gro_lines,value_atom_number_in_gro

def merge_node_edge_term(nodes_tG,edges_tG,terms_tG,node_res_num,edge_res_num):
    merged_node_edge_term = []
    line_num = 0
    for node in nodes_tG:
        formatted_gro_lines,line_num = convert_node_array_to_gro_lines(node,line_num,0,'NOD')
        merged_node_edge_term+=formatted_gro_lines
    for edge in edges_tG:
        formatted_gro_lines,line_num = convert_node_array_to_gro_lines(edge,line_num,node_res_num,'EDG')
        merged_node_edge_term+=formatted_gro_lines
    for term in terms_tG:
        formatted_gro_lines,line_num = convert_node_array_to_gro_lines(term,line_num,node_res_num+edge_res_num,'TER')
        merged_node_edge_term+=formatted_gro_lines
    return merged_node_edge_term

def save_node_edge_term_gro(merged_node_edge_term,gro_name):
    with open(str(gro_name)+'.gro','w') as f:
        head =[]
        head.append("eG_NET\n")
        head.append(str(len(merged_node_edge_term))+"\n")
        f.writelines(head)
        f.writelines(merged_node_edge_term)
        tail = ["10 10 10 \n"]
        f.writelines(tail)




#debug write gro function
def convert_positions_to_gro_lines(line,line_num,res_num,name):
    line_num+=1
    value_atom_number = int(line_num)  # atom_number
    value_label = re.sub('\d','',line[0]) # atom_label
    value_resname = str(name)[0:3]#+str(eG.nodes[n]['index'])  # residue_name
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

    return formatted_line,line_num, res_num

def temp_save_eGterm_gro(eG,sc_unit_cell):
    with open('eG_TERM.gro','w') as f:
        newgro = []
        res_num = 0
        line_num = 0
        for n in eG.nodes():
            if pname(n) != 'EDGE':
                postions = eG.nodes[n]['f_points']
                res_num+=1
                fc = np.asarray(postions[:,2:5],dtype=float)
                cc = np.dot(sc_unit_cell,fc.T).T
                positionss = np.hstack((postions[:,1:2],cc))
                for line in positionss:
                    formatted_line,line_num, res_num = convert_positions_to_gro_lines(line,line_num,res_num,n)
                    newgro.append(formatted_line + "\n")
                for term_ind_key,c_positions in eG.nodes[n]['term_c_points'].items():
                    for line in c_positions:
                        formatted_line,line_num, res_num = convert_positions_to_gro_lines(line,line_num,res_num,term_ind_key)
                        newgro.append(formatted_line + "\n")
                
            elif pname(n) == 'EDGE':
                postions = eG.nodes[n]['f_points']
                res_num+=1
                fc = np.asarray(postions[:,2:5],dtype=float)
                cc = np.dot(sc_unit_cell,fc.T).T
                positionss = np.hstack((postions[:,1:2],cc))
                for line in positionss:
                    formatted_line,line_num, res_num = convert_positions_to_gro_lines(line,line_num,res_num,n)
                    newgro.append(formatted_line + "\n")
        head =[]
        head.append("eG_TERM\n")
        head.append(str(line_num)+"\n")
        tail = ["10 10 10 \n"]
        f.writelines(head)
        f.writelines(newgro)
        f.writelines(tail)
 