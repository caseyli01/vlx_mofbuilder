from pathlib import Path
import numpy as np
import os
import datetime
import veloxchem as vlx
import networkx as nx
import matplotlib.pyplot as plt


def create_lG(molecule):
    matrix=molecule.get_connectivity_matrix()
    coords=molecule.get_coordinates_in_angstrom()
    labels = molecule.get_labels()
    dist_matrix = molecule.get_distance_matrix_in_angstrom()
    mass_center_bohr=molecule.center_of_mass_in_bohr()
    bohr_to_angstrom = 0.529177
    mass_center_angstrom = np.asarray(mass_center_bohr)*bohr_to_angstrom 
    coords = coords - mass_center_angstrom


    METEL_ELEMENTS = ['Ac','Ag','Al','Am','Au','Ba','Be','Bi',
				  'Bk','Ca','Cd','Ce','Cf','Cm','Co','Cr',
				  'Cs','Cu','Dy','Er','Es','Eu','Fe','Fm',
				  'Ga','Gd','Hf','Hg','Ho','In','Ir',
				  'K','La','Li','Lr','Lu','Md','Mg','Mn',
				  'Mo','Na','Nb','Nd','Ni','No','Np','Os',
				  'Pa','Pb','Pd','Pm','Pr','Pt','Pu','Ra',
				  'Rb','Re','Rh','Ru','Sc','Sm','Sn','Sr',
				  'Ta','Tb','Tc','Th','Ti','Tl','Tm','U',
				  'V','W','Y','Yb','Zn','Zr']

    lG=nx.Graph()
    metals=[]
    for i in range(len(labels)):
            lG.add_nodes_from([(i,{'label': labels[i],'coords': coords[i]})])
            if labels[i] in METEL_ELEMENTS:
                metals.append(i)

    i=None
    for i in range(len(labels)):
        for j in np.where(matrix[i]==1)[0]:
            if i not in metals and j not in metals:
                lG.add_edge(i,j,weight = dist_matrix[i,j])    
    return lG,metals,mass_center_angstrom

def plot2dedge(lG,coords,cycle,EDGE_length=False):
    #pos = coords
    #nodes = np.array([pos[v] for v in lG])
    #edges = np.array([(pos[u], pos[v]) for u, v in lG.edges()])
    #pos = None
    pos = coords[:,:-1]
    # explicitly set positions
    #pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

    options = {
            "font_size": 10,
            "node_size": 150,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1,
        }
    nx.draw_networkx(lG,pos,**options)


    # edge weight labels
    if EDGE_length:
        edge_labels = nx.get_edge_attributes(lG, "weight")
        nx.draw_networkx_edge_labels(lG, pos, edge_labels)

    # Highlight the cycle in red
    if len(cycle)>0:
        nx.draw_networkx_edges(lG, pos, edgelist=cycle, edge_color="r", width=2)
    plt.show()

def find_center_cycle_nodes(lG):
    #To find center cycle
    target_nodes = set(nx.center(lG))
    cycles=list(nx.simple_cycles(lG,length_bound=80))
    for cycle in cycles:
        if target_nodes < set(cycle):
            return cycle
        
def distinguish_G_centers(lG):
    centers = nx.barycenter(lG)
    if len(centers)==1:
        print('center is a point')
        center_class = 'onepoint'
        center_nodes = centers
    elif len(centers)==2:
        if nx.shortest_path_length(lG,centers[0],centers[1])==1:
            print('center is two points')
            center_class = 'twopoints'
            center_nodes = centers
        else:
            print('center is a cycle')
            center_class = 'cycle'
            center_nodes = find_center_cycle_nodes(lG)
    else:
        print('center is a cycle')
        center_class = 'cycle'
        center_nodes = find_center_cycle_nodes(lG)
    return center_class,center_nodes

def classify_nodes(lG,center_nodes):
    # Step 1: Identify the center node(s) of the graph
    #center_nodes = nx.center(lG)

    for center_ind in range(len(center_nodes)):
        #to classify which node belonging to which center_node, add 'cnode_l' to lG nodes
        # Compute the shortest path length from the center node to all other nodes
        center = center_nodes[center_ind]
        lengths = nx.single_source_shortest_path_length(lG, center)
        if center_ind == 0:
            for k in lengths:
                    lG.nodes[k]['cnodes_l']=(center,lengths[k])
        elif center_ind >0:
                for k in lengths:  
                    if lengths[k] < lG.nodes[k]['cnodes_l'][1]:
                        lG.nodes[k]['cnodes_l']=(center,lengths[k])
                    elif lengths[k] == lG.nodes[k]['cnodes_l'][1]:
                        lG.nodes[k]['cnodes_l']=(-1,lengths[k]) #if the node is between any two center nodes
    return lG

def find_center_nodes_pair(lG,center_nodes):
    if len(center_nodes)>6:
        centers=nx.center(lG)
        
    pairs=[]
    for i in range(len(centers)):
        for j in range(i,len(centers)):
            l=nx.shortest_path_length(lG,centers[i],centers[j])
            if l==1:
                pairs.append([centers[i],centers[j]])

    # loop each pair to find center pair
    for p in pairs:
        a=p[0]
        b=p[1]
        ds=[]
        for n in centers:
            if n not in p:
                d=nx.shortest_path_length(lG,a,n)
                ds.append(d)
        if len(set(ds)) < len(ds):
            center_pair = p
        
    return center_pair

def get_pairX_outer_frag(connected_pairXs,outer_frag_nodes):
    for x in list(connected_pairXs):
            pairXs = [connected_pairXs[x][1],connected_pairXs[x][3]]
            if set(pairXs) < set(outer_frag_nodes):
                break
    return pairXs

def cleave_outer_frag_subgraph(lG,pairXs,outer_frag_nodes):

    subgraph_outer_frag = lG.subgraph(outer_frag_nodes)
    kick_nodes = []
    for i in list(outer_frag_nodes):
        if nx.shortest_path_length(subgraph_outer_frag,pairXs[0],i) > nx.shortest_path_length(subgraph_outer_frag,pairXs[0],pairXs[1]):
            kick_nodes.append(i)
        elif nx.shortest_path_length(subgraph_outer_frag,pairXs[1],i) > nx.shortest_path_length(subgraph_outer_frag,pairXs[0],pairXs[1]):
            kick_nodes.append(i)
        
    subgraph_single_frag = lG.subgraph(outer_frag_nodes-set(kick_nodes))
    return subgraph_single_frag

def lines_of_center_frag(subgraph_center_frag,Xs_indices,metals,labels,coords,mass_center_angstrom):
    count = 1
    lines = []
    Xs =[]
    for cn in list(subgraph_center_frag.nodes):
        label = subgraph_center_frag.nodes[cn]['label']
        coord = subgraph_center_frag.nodes[cn]['coords']
        if cn not in Xs_indices:
            name = label+str(count)
        else:
            name = 'X'+str(count)
            Xs.append(count-1)
        count+=1
        lines.append([name,label,coord[0],coord[1],coord[2]])
    for cm in metals:
        label = labels[cm]
        coord = coords[cm]-mass_center_angstrom
        name = label+str(count)
        lines.append([name,label,coord[0],coord[1],coord[2]])
    return lines,Xs

def lines_of_single_frag(subgraph_single_frag,Xs_indices):
    count =1
    rows = []
    Xs = []
    for sn in list(subgraph_single_frag.nodes):
        label = subgraph_single_frag.nodes[sn]['label']
        coord = subgraph_single_frag.nodes[sn]['coords']
        if sn not in Xs_indices:
            name = label+str(count)
        else:
            name = 'X'+str(count)
            Xs.append(count-1)
        count+=1
        rows.append([name,label,coord[0],coord[1],coord[2]])
    return rows,Xs

def get_atom_name_in_subgraph(subgraph,n_id,Xs_indices):
    for ind, value in enumerate(list(subgraph.nodes)):
        if value == n_id:
            if value not in Xs_indices:
                return (subgraph.nodes[n_id]['label']+ str(ind+1))
            else:
                return ('X'+ str(ind+1))
        
def get_bonds_from_subgraph(subgraph,Xs_indices):
    bonds =[]
    for e in list(subgraph.edges):
        atom1 = get_atom_name_in_subgraph(subgraph,e[0],Xs_indices)
        atom2 = get_atom_name_in_subgraph(subgraph,e[1],Xs_indices)
        length = subgraph.edges[e]['weight']/50  # 50 50 50 box
        sym = '.'
        if atom1[0]=='X' or atom2[0]=='X':
            bond_type = 'A'
        else:
            bond_type = 'S'
        bonds.append([atom1,atom2,length,sym,bond_type])

    return bonds

def create_cif(name_label_coords, bonds, foldername,cifname):
	opath = os.path.join(foldername, cifname)
	print(opath,'is writen')
	with open(opath, 'w') as out:
		out.write('data_' + cifname[0:-4] + '\n')
		out.write('_audit_creation_date              ' + datetime.datetime.today().strftime('%Y-%m-%d') + '\n')
		out.write("_audit_creation_method            'MOFbuilder'" + '\n')
		out.write("_symmetry_space_group_name_H-M    'P1'" + '\n')
		out.write('_symmetry_Int_Tables_number       1' + '\n')
		out.write('_symmetry_cell_setting            triclinic' + '\n')
		out.write('loop_' + '\n')
		out.write('_symmetry_equiv_pos_as_xyz' + '\n')
		out.write('  x,y,z' + '\n')
		out.write('_cell_length_a                    ' + '50' + '\n')
		out.write('_cell_length_b                    ' + '50' + '\n')
		out.write('_cell_length_c                    ' + '50' + '\n')
		out.write('_cell_angle_alpha                 ' + '90' + '\n')
		out.write('_cell_angle_beta                  ' + '90' + '\n')
		out.write('_cell_angle_gamma                 ' + '90' + '\n')
		out.write('loop_' + '\n')
		out.write('_atom_site_label' + '\n')
		out.write('_atom_site_type_symbol' + '\n')
		out.write('_atom_site_fract_x' + '\n')
		out.write('_atom_site_fract_y' + '\n')
		out.write('_atom_site_fract_z' + '\n')

		for l in name_label_coords:

			vec = list(map(float, l[2:5]))
			m = np.array(([0.02,0,0],[0,0.02,0],[0,0,0.02]))
			cvec = np.dot(m, vec)
	
			
			cvec = np.mod(cvec, 1)
			extra = '   0.00000  Uiso   1.00       -0.000000'
			out.write('{:7} {:>4} {:>15} {:>15} {:>15}'.format(l[0], l[1], "%.10f" % np.round(cvec[0],10), "%.10f" % np.round(cvec[1],10), "%.10f" % np.round(cvec[2],10)))
			out.write(extra)
			out.write('\n')

		out.write('loop_' + '\n')
		out.write('_geom_bond_atom_site_label_1' + '\n')
		out.write('_geom_bond_atom_site_label_2' + '\n')
		out.write('_geom_bond_distance' + '\n')
		out.write('_geom_bond_site_symmetry_2' + '\n')
		out.write('_ccdc_geom_bond_type' + '\n')

		for b in bonds:
			out.write('{:7} {:>7} {:>5} {:>7} {:>3}'.format(b[0], b[1], "%.3f" % float(b[2]), b[3], b[4]))
			out.write('\n')


def create_pdb(filename,lines):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    newpdb = []
    newpdb.append("PDB file \n"+str(filename)+"   GENERATED BY MOF_builder\n")
    with open(str(filename)+".pdb", "w") as fp:
        # Iterate over each line in the input file
        for i in range(len(lines)):
            # Split the line into individual values (assuming they are separated by spaces)
            values = lines[i]
            # Extract values based on their positions in the format string
            value1 = "ATOM"
            value2 = int(i + 1)
            value3 = values[0]  # label
            value4 = 'MOL'  # residue
            value5 = 1  # residue number
            value6 = float(values[2])  # x
            value7 = float(values[3])  # y
            value8 = float(values[4])  # z
            value9 = "1.00"
            value10 = "0.00"
            value11 = values[1]  # note
            # Format the values using the specified format string
            formatted_line = "%-6s%5d%5s%4s%10d%8.3f%8.3f%8.3f%6s%6s%4s" % (
                value1,
                value2,
                value3,
                value4,
                value5,
                value6,
                value7,
                value8,
                value9,
                value10,
                value11,
            )
            newpdb.append(formatted_line + "\n")
        fp.writelines(newpdb)


def process_linker_molecule(molecule,linker_topic):
    coords=molecule.get_coordinates_in_angstrom()
    labels = molecule.get_labels()
    # To remove center metals
    lG,metals,mass_center_angstrom = create_lG(molecule)
    lG.remove_nodes_from(metals)
    center_class,center_nodes = distinguish_G_centers(lG)
    if linker_topic==2 and len(center_nodes)>6:
        center_nodes = find_center_nodes_pair(lG,center_nodes)


    lG = classify_nodes(lG,center_nodes)
    print(center_nodes)

    if center_class=='cycle' and linker_topic >2:
        print("tritopic/tetratopic/multitopic: center is a cycle")
        connected_pairXs = {}
        Xs_indices = []
        innerX_coords =[]
        for k in range(len(center_nodes)):
            linker_C_l = []
            l_list = []
            for n in lG.nodes:
                if lG.nodes[n]['cnodes_l'][0] == center_nodes[k] and lG.nodes[n]['label'] == 'C':
                    linker_C_l.append((n,lG.nodes[n]['cnodes_l']))
                    l_list.append(lG.nodes[n]['cnodes_l'][1]) 
            center_connected_C_ind = [ind for ind,value in enumerate(l_list) if value ==1]
            outer_connected_C_ind = [ind for ind,value in enumerate(l_list) if value ==(max(l_list)-1)]
            if len(center_connected_C_ind) ==1 and len(outer_connected_C_ind)==1:
                inner_X = linker_C_l[center_connected_C_ind[0]]
                outer_X = linker_C_l[outer_connected_C_ind[0]]
                if center_nodes[k] not in [inner_X[0], outer_X[0]]:
                    print("find connected X in edge frag",inner_X[0],outer_X[0],center_nodes[k])
                    lG.remove_edge(inner_X[0],center_nodes[k])
                    connected_pairXs[center_nodes[k]]=('inner_X', inner_X[0],'outer_X', outer_X[0])
                    Xs_indices+=[center_nodes[k],inner_X[0],outer_X[0]]
                    innerX_coords.append(lG.nodes[inner_X[0]]['coords'])

        if nx.number_connected_components(lG) != linker_topic+1: #for check linker_topics+1  
            print("wrong fragments")
            raise ValueError
        
    elif linker_topic==2:
        if center_class == "twopoints":
            print("ditopic linker: center are two points")
            Xs_indices = []
            for k in range(len(center_nodes)):
                linker_C_l = []
                l_list = []
                for n in lG.nodes:
                    if lG.nodes[n]['cnodes_l'][0] == center_nodes[k]and lG.nodes[n]['label'] == 'C':
                        linker_C_l.append((n,lG.nodes[n]['cnodes_l']))
                        l_list.append(lG.nodes[n]['cnodes_l'][1]) 

                outer_connected_C_ind = [ind for ind,value in enumerate(l_list) if value ==(max(l_list)-1)]

                if len(outer_connected_C_ind)==1:
                    outer_X = linker_C_l[outer_connected_C_ind[0]]
                    if center_nodes[k] not in [ outer_X[0]]:
                        print("find connected X in edge:  ",outer_X[0])
                        Xs_indices+=[outer_X[0]]

        if center_class == "onepoint":
            print("ditopic linker: center is a point")
            Xs_indices = []
            linker_C_l = []
            l_list = []
            for n in lG.nodes:
                if lG.nodes[n]['cnodes_l'][0] == center_nodes[0] and lG.nodes[n]['label'] == 'C':
                    linker_C_l.append((n,lG.nodes[n]['cnodes_l']))
                    l_list.append(lG.nodes[n]['cnodes_l'][1]) 

            outer_connected_C_ind = [ind for ind,value in enumerate(l_list) if value ==(max(l_list)-1)]
            for m in range(len(outer_connected_C_ind)):
                    outer_X = linker_C_l[outer_connected_C_ind[m]]
                    print("find connected X in edge:  ",outer_X[0])
                    Xs_indices+=[outer_X[0]] 
        if center_class=='cycle' :
            print("ditopic linker: center is a cycle")
            connected_pairXs = {}
            Xs_indices = []
            for k in range(len(center_nodes)):
                linker_C_l = []
                l_list = []
                for n in lG.nodes:
                    if lG.nodes[n]['cnodes_l'][0] == center_nodes[k] and lG.nodes[n]['label'] == 'C':
                        linker_C_l.append((n,lG.nodes[n]['cnodes_l']))
                        l_list.append(lG.nodes[n]['cnodes_l'][1]) 
                outer_connected_C_ind = [ind for ind,value in enumerate(l_list) if value ==(max(l_list)-1)]
                
                if len(outer_connected_C_ind)==1:
                    outer_X = linker_C_l[outer_connected_C_ind[0]]
                    if center_nodes[k] not in [ outer_X[0]]:
                        print("find connected X in edge:  ",outer_X[0])
                        Xs_indices+=[outer_X[0]]
    else:
        raise ValueError("failed to recognize a multitopic linker whose center is not a cycle")

    #write cifs

    if linker_topic > 2: #multitopic
        frag_nodes = list(sorted(nx.connected_components(lG), key=len, reverse=True))
        for f in frag_nodes:
            if set(center_nodes) < set(f):
                center_frag_nodes = f
            else:
                outer_frag_nodes = f 
        
        subgraph_center_frag = lG.subgraph(center_frag_nodes)
        lines,center_Xs = lines_of_center_frag(subgraph_center_frag,Xs_indices,metals,labels,coords,mass_center_angstrom)
        center_frag_bonds = get_bonds_from_subgraph(subgraph_center_frag,Xs_indices)
        subgraph_center_frag_edges = list(subgraph_center_frag.edges)
        #plot2dedge(lG,coords,subgraph_center_frag_edges,True)
        #plot2dedge(lG,coords,subgraph_center_frag_edges,False)
        pairXs=get_pairX_outer_frag(connected_pairXs,outer_frag_nodes)
        subgraph_single_frag = cleave_outer_frag_subgraph(lG,pairXs,outer_frag_nodes)
        rows,frag_Xs = lines_of_single_frag(subgraph_single_frag,Xs_indices)
        single_frag_bonds = get_bonds_from_subgraph(subgraph_single_frag,Xs_indices)
        if linker_topic == 3:
            print('center_frag:',subgraph_center_frag.number_of_nodes(),center_Xs)
            print('outer_frag:' ,subgraph_single_frag.number_of_nodes(),frag_Xs)
            create_cif(lines,center_frag_bonds,'nodes','tricenter.cif')
            create_cif(rows,single_frag_bonds,'edges','triedge.cif')
            return (subgraph_center_frag.number_of_nodes(),center_Xs,subgraph_single_frag.number_of_nodes(),frag_Xs)
        elif linker_topic == 4:
            print('center_frag:',subgraph_center_frag.number_of_nodes(),center_Xs)
            print('outer_frag:' ,subgraph_single_frag.number_of_nodes(),frag_Xs)
            create_cif(lines,center_frag_bonds,'nodes','tetracenter.cif')
            create_cif(rows,single_frag_bonds,'edges','tetraedge.cif')
            return (subgraph_center_frag.number_of_nodes(),center_Xs,subgraph_single_frag.number_of_nodes(),frag_Xs)
        else:
            create_cif(lines,center_frag_bonds,'nodes','multicenter.cif')
            create_cif(rows,single_frag_bonds,'edges','multiedge.cif')
            return (subgraph_center_frag.number_of_nodes(),center_Xs,subgraph_single_frag.number_of_nodes(),frag_Xs)
        
    elif linker_topic == 2: #ditopic
        pairXs = Xs_indices
        subgraph_center_frag = cleave_outer_frag_subgraph(lG,pairXs,lG.nodes)
        subgraph_center_frag_edges = list(subgraph_center_frag.edges)
        #plot2dedge(lG,coords,subgraph_center_frag_edges,True)
        #plot2dedge(subgraph_center_frag,coords,subgraph_center_frag_edges,False)
        lines,center_Xs = lines_of_center_frag(subgraph_center_frag,Xs_indices,metals,labels,coords,mass_center_angstrom)
        center_frag_bonds = get_bonds_from_subgraph(subgraph_center_frag,Xs_indices)
        #create_cif(lines,center_frag_bonds,'edges','diedge.cif')
        create_pdb('edges/diedge',lines)
        print('center_frag:',subgraph_center_frag.number_of_nodes(),center_Xs)
        return (subgraph_center_frag.number_of_nodes(),center_Xs,0,[])

        