import numpy as np
import networkx as nx
import re 

from place_bbs import superimpose

# Function to apply periodic boundary conditions in 3D
def PBC3DF_sym(vec1, vec2):
	# vec1, vec2: input vectors

	dX, dY, dZ = vec1 - vec2
			
	if dX > 0.5:
		s1 = 1.0
		ndX = dX - 1.0
	elif dX < -0.5:
		s1 = -1.0
		ndX = dX + 1.0
	else:
		s1 = 0.0
		ndX = dX
				
	if dY > 0.5:
		s2 = 1.0
		ndY = dY - 1.0
	elif dY < -0.5:
		s2 = -1.0
		ndY = dY + 1.0
	else:
		s2 = 0.0
		ndY = dY
	
	if dZ > 0.5:
		s3 = 1.0
		ndZ = dZ - 1.0
	elif dZ < -0.5:
		s3 = -1.0
		ndZ = dZ + 1.0
	else:
		s3 = 0.0
		ndZ = dZ

	sym = np.array([s1, s2, s3])

	return np.array([ndX, ndY, ndZ]), sym

# Function to calculate new node coordinates based on periodic boundary conditions
def newno_fxnx(f_ex, nx, sc_unit_cell, no):
	# f_ex: fractional coordinates of the edge
	# nx: node coordinates
	# sc_unit_cell: supercell unit cell matrix
	# no: original node coordinates

	f_nx = np.dot(np.linalg.inv(sc_unit_cell), nx)
	fdist_vec, sym = PBC3DF_sym(f_ex, f_nx) 
	f_no = np.dot(np.linalg.inv(sc_unit_cell), no)
	new_no = np.dot(sc_unit_cell, f_no + sym)
	return new_no

# Function to adjust edges based on node positions
def _adjust_edges(placed_edges, placed_nodes, sc_unit_cell):
	# placed_edges: list of placed edges
	# placed_nodes: list of placed nodes
	# sc_unit_cell: supercell unit cell matrix

	adjusted_placed_edges = []
	adjusted_placed_edges_extend = adjusted_placed_edges.extend
	adjusted_placed_OXedges = []
	adjusted_placed_OXedges_extend = adjusted_placed_OXedges.extend

	placed_edges = np.asarray(placed_edges)
	edge_labels = set(map(int, placed_edges[:, -1]))

	edge_dict = dict((k, []) for k in edge_labels)

	# Filter nodes with 'X' in their labels
	node_connection_x = np.asarray([i for i in placed_nodes if re.sub('[0-9]', '', i[5]) == 'X'])
	nx_elems = node_connection_x[:, 0]
	node_connection_points = [list(map(float, i)) for i in node_connection_x[:, 1:4]]
	nx_charges = node_connection_x[:, 4]
	nx_cp = node_connection_x[:, 5]

	# Filter nodes with 'O' in their labels
	node_oxy = np.asarray([i for i in placed_nodes if re.sub('[0-9]', '', i[0]) == 'O'])
	no_elems = node_oxy[:, 0]
	node_oxy_points = [list(map(float, i)) for i in node_oxy[:, 1:4]]
	no_charges = node_oxy[:, 4]
	no_cp = node_oxy[:, 5]

	# Find pairs of 'O' nodes for each 'X' node
	X_Opair = []
	X_Opair_append = X_Opair.append
	for i in range(len(node_connection_points)):
		cdist_xos = []
		cdist_xos_sort = []
		cdist_xos_append = cdist_xos.append
		cdist_xos_sort_append = cdist_xos_sort.append
		fvec_x = node_connection_points[i]
		for j in range(len(node_oxy_points)):
			fvec_o = node_oxy_points[j]
			fvec_xo = np.asarray(fvec_o) - np.asarray(fvec_x) 	
			fdist_xo = np.dot(np.linalg.inv(sc_unit_cell), fvec_xo)
			cdist_xo = np.linalg.norm(np.dot(sc_unit_cell, fdist_xo))
			cdist_xos_append(cdist_xo)
			cdist_xos_sort_append(cdist_xo)
		cdist_xos_sort.sort()
		cdist_xos_sort3rd = cdist_xos_sort[2]
		opair = [index for index, value in enumerate(cdist_xos) if value < cdist_xos_sort3rd]
		X_Opair_append(('X' + str(i), node_connection_points[i], (opair), [node_oxy_points[k] for k in opair]))
	
	'''cleave placed_nodes remove X_Opair, but future need add dummy atom(can be applied in node cif file )'''
	print(f"X_Opair{X_Opair}")

	opairs_vec= [j for i in X_Opair for j in i[3]]
	xs_vec = [i[1]  for i in X_Opair]
	xos_vec = opairs_vec+xs_vec
	#cleaved_placed_nodes = []
	#cleaved_placed_nodes_append= cleaved_placed_nodes.append
	#for i in placed_nodes:
	#	if re.sub('[0-9]','',i[5]) == 'X':
	#		if list(map(float,i[1:4])) not in xos_vec:####remove xoo from node_fc
	#			cleaved_placed_nodes_append(i)

	#	elif re.sub('[0-9]','',i[5]) == 'O':
	#		if list(map(float,i[1:4])) not in xos_vec:####remove xoo from node_fc
	#			cleaved_placed_nodes_append(i)

	#	else:
	#		cleaved_placed_nodes_append(i)


	for edge in placed_edges:
		ty = int(edge[-1])
		edge_dict[ty].append(edge)

	for k in edge_dict:
		edge = np.asarray(edge_dict[k])
		elems = edge[:, 0]
		evecs = [list(map(float, i)) for i in edge[:, 1:4]]
		charges = edge[:, 4]
		cp = edge[:, 5]
		ty = edge[:, 6]

		xvecs = [list(map(float, i)) for (i, j) in zip(evecs, cp) if re.sub('[0-9]', '', j) == 'X']
		relevant_node_xvecs = []
		relevant_node_xvecs_append = relevant_node_xvecs.append
		corr_opair = []
		corr_opair_append = corr_opair.append
		for count in range(len(xvecs)):
			ex = xvecs[count]
			min_dist = (1e6, [], 0)

			f_ex = np.dot(np.linalg.inv(sc_unit_cell), ex)
			for i in range(len(node_connection_points)):
				nx = node_connection_points[i]
				f_nx = np.dot(np.linalg.inv(sc_unit_cell), nx)

				fdist_vec, sym = PBC3DF_sym(f_ex, f_nx) 
				cdist = np.linalg.norm(np.dot(sc_unit_cell, fdist_vec))

				if cdist < min_dist[0]:
					min_dist = (cdist, np.dot(sc_unit_cell, f_nx + sym), i)
					target_nx = nx
			relevant_node_xvecs_append(min_dist[1])
			idx_nx = min_dist[2]
			corresponding_o_vec = [newno_fxnx(f_ex, target_nx, sc_unit_cell, no) for no in X_Opair[idx_nx][3]]
			corresponding_x_vec = [newno_fxnx(f_ex, target_nx, sc_unit_cell, target_nx)]
			pairo_indices = X_Opair[idx_nx][2]
			elems_nopair = [no_elems[i] for i in pairo_indices]
			charges_nopair = [no_charges[i] for i in pairo_indices]
			cp_nopair = [no_cp[i] for i in pairo_indices]
			ty_nopair = [ty[0]] * len(pairo_indices)

			corresponding_o = np.c_[elems_nopair, corresponding_o_vec, charges_nopair, cp_nopair, ty_nopair]
			corresponding_x = np.c_[[nx_elems[min_dist[2]]], corresponding_x_vec, [nx_charges[min_dist[2]]], [nx_cp[min_dist[2]]], [ty[0]]]
			corr_opair_append(corresponding_o)
			corr_opair_append(corresponding_x)
		ecom = np.average(xvecs, axis=0)
		rnxcom = np.average(relevant_node_xvecs, axis=0)

		evecs = np.asarray(evecs - ecom)
		xvecs = np.asarray(xvecs - ecom)
		relevant_node_xvecs = np.asarray(relevant_node_xvecs)

		trans = rnxcom
		min_dist,rot,tran = superimpose(xvecs,relevant_node_xvecs)
		adjusted_evecs = np.dot(evecs,rot) + trans
		#add oox to in_edge array
		adjusted_edge_in = np.column_stack((elems,adjusted_evecs,charges,cp,ty))
		adjusted_edge_opair = np.vstack(corr_opair)
		adjusted_OXedge = np.vstack((adjusted_edge_opair, adjusted_edge_in))
		adjusted_placed_edges_extend(adjusted_edge_in)
		adjusted_placed_OXedges_extend(adjusted_OXedge)

	return adjusted_placed_edges, adjusted_placed_OXedges, placed_nodes, X_Opair

# Function to adjust edges based on node positions (simplified version)
def adjust_edges(placed_edges, placed_nodes, sc_unit_cell):
	# placed_edges: list of placed edges
	# placed_nodes: list of placed nodes
	# sc_unit_cell: supercell unit cell matrix

	adjusted_placed_edges = []
	adjusted_placed_edges_extend = adjusted_placed_edges.extend

	placed_edges = np.asarray(placed_edges)
	edge_labels = set(map(int, placed_edges[:, -1]))

	edge_dict = dict((k, []) for k in edge_labels)
	node_connection_points = [list(map(float, i[1:4])) for i in placed_nodes if re.sub('[0-9]', '', i[5]) == 'X']

	for edge in placed_edges:
		ty = int(edge[-1])
		edge_dict[ty].append(edge)

	for k in edge_dict:
		edge = np.asarray(edge_dict[k])
		elems = edge[:, 0]
		evecs = [list(map(float, i)) for i in edge[:, 1:4]]
		charges = edge[:, 4]
		cp = edge[:, 5]
		ty = edge[:, 6]

		xvecs = [list(map(float, i)) for (i, j) in zip(evecs, cp) if re.sub('[0-9]', '', j) == 'X']
		relavent_node_xvecs = []
		relavent_node_xvecs_append = relavent_node_xvecs.append

		for ex in xvecs:
			min_dist = (1e6, [], 0)

			f_ex = np.dot(np.linalg.inv(sc_unit_cell), ex)
			for i in range(len(node_connection_points)):
				nx = node_connection_points[i]
				f_nx = np.dot(np.linalg.inv(sc_unit_cell), nx)

				fdist_vec, sym = PBC3DF_sym(f_ex, f_nx) 
				cdist = np.linalg.norm(np.dot(sc_unit_cell, fdist_vec))

				if cdist < min_dist[0]:
					min_dist = (cdist, np.dot(sc_unit_cell, f_nx + sym), i)

			node_connection_points.pop(min_dist[2])
			relavent_node_xvecs_append(min_dist[1])

		ecom = np.average(xvecs, axis=0)
		rnxcom = np.average(relavent_node_xvecs, axis=0)

		evecs = np.asarray(evecs - ecom)
		xvecs = np.asarray(xvecs - ecom)
		relavent_node_xvecs = np.asarray(relavent_node_xvecs)

		trans = rnxcom
		min_dist, rot, tran = superimpose(xvecs, relavent_node_xvecs)
		adjusted_evecs = np.dot(evecs, rot) + trans
		adjusted_edge = np.column_stack((elems, adjusted_evecs, charges, cp, ty))
		adjusted_placed_edges_extend(adjusted_edge)

	return adjusted_placed_edges

# Function to fetch indices and coordinates of atoms with a specific label
def fetch_X_atoms_ind_array(array, column, X):
	# array: input array
	# column: column index to check for label
	# X: label to search for

	ind = [k for k in range(len(array)) if re.sub(r'\d', '', array[k, column]) == X]
	x_array = array[ind]
	return ind, x_array

# Function to calculate the angle between two vectors
def get_rad_v1v2(v1, v2):
	# v1, v2: input vectors

	cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
	if cos_theta == 0:
		return 0
	else:
		rad = np.arccos(cos_theta)
		return rad

# Function to filter the closest 'X' atoms based on angle and distance
def filt_closest_x_angle(Xs_fc, edge_center_fc, node_center_fc):
	# Xs_fc: coordinates of 'X' atoms
	# edge_center_fc: coordinates of the edge center
	# node_center_fc: coordinates of the node center

	rds_list = []
	rads = []
	dists = []
	x_number = len(Xs_fc)
	half_x_number = int(0.5 * x_number)
	for i in range(x_number):
		rad = get_rad_v1v2(Xs_fc[i] - edge_center_fc, node_center_fc - edge_center_fc)
		dist = np.linalg.norm(Xs_fc[i] - edge_center_fc)
		rds_list.append((i, rad, dist))
		rads.append(rad)
		dists.append(dist)
	rads.sort()
	dists.sort()
	x_idx = [i[0] for i in rds_list if (i[1] < 0.6 and i[2] < dists[half_x_number])] # 0.6 rad == 35 degrees
	x_info = [i for i in rds_list if (i[1] < 0.6 and i[2] < dists[half_x_number])]
	if len(x_idx) == 1:
		return x_idx, x_info
	elif len(x_idx) > 1:
		min_d = min([j[2] for j in x_info])
		x_idx1 = [i[0] for i in rds_list if i[2] == min_d]
		x_info1 = [i for i in rds_list if i[2] == min_d]
		return x_idx1, x_info1
	else:
		print("ERROR cannot find connected X")
		print(rds_list)

# Function to filter the closest 'X' atoms based on distance
def filt_close_edgex(Xs_fc, edge_center_fc, linker_topics):
	# Xs_fc: coordinates of 'X' atoms
	# edge_center_fc: coordinates of the edge center
	# linker_topics: number of linker topics

	lcs_list = []
	lcs = []
	for i in range(len(Xs_fc)):
		lc = np.linalg.norm(Xs_fc[i] - edge_center_fc)
		lcs_list.append((i, lc))
		lcs.append(lc)
	lcs.sort()
	outside_edgex_indices = [i[0] for i in lcs_list if i[1] < lcs[linker_topics]]
	outside_edgex_ind_dist = [i for i in lcs_list if i[1] < lcs[linker_topics]]
	return outside_edgex_indices, outside_edgex_ind_dist

# Function to find 'XOO' pairs for a specific node
def xoo_pair_ind_node(main_frag_nodes_fc, node_id, sc_unit_cell):
	# main_frag_nodes_fc: coordinates of main fragment nodes
	# node_id: ID of the node
	# sc_unit_cell: supercell unit cell matrix

	xoo_ind_node = []
	single_node_fc = main_frag_nodes_fc[main_frag_nodes_fc[:, 5] == node_id]
	single_node = np.hstack((single_node_fc[:, :-3], np.dot(sc_unit_cell, single_node_fc[:, -3:].T).T))
	xind, xs = fetch_X_atoms_ind_array(single_node, 2, 'X')
	oind, os = fetch_X_atoms_ind_array(single_node, 2, 'O')
	for i in xind:
		x = single_node[i][-3:]
		pair_oind, pair_o_info = filt_close_edgex(os[:, -3:], x, 2)
		xoo_ind_node.append((i, [oind[po] for po in pair_oind]))
	return xoo_ind_node

# Function to replace 'X' with 'x' in edge labels
def replace_Xbyx(single_edge):
	# single_edge: input edge

	for i in range(len(single_edge)):
		if single_edge[i, 2][0] == 'X':
			single_edge[i, 2] = re.sub('X', 'x', single_edge[i, 2])
	return single_edge

# Function to filter coordinates of 'X' atoms in an edge
def filt_xcoords(single_edge):
	# single_edge: input edge

	xcoords = []
	for i in range(len(single_edge)):
		if single_edge[i, 2][0] == 'X':
			single_edge_xcoord = single_edge[i, -3:]
			xcoords.append(single_edge_xcoord)
	return xcoords

# Function to correct the order of neighbor nodes based on edge 'X' atoms order
def correct_neighbor_nodes_order_by_edge_xs_order(eG, edge_n, single_edge):
	# eG: graph
	# edge_n: edge node
	# single_edge: input edge

	neighbor_nodes = list(nx.neighbors(eG, edge_n))
	for inn in neighbor_nodes:
		c_nn = eG.nodes[inn]['fc']
	a = filt_xcoords(single_edge) # edge 'X' atoms order
	if len(a) == 6:
		a = a[-3:]
	if len(a) == 8:
		a = a[-4:]
	b = neighbor_nodes # neighbor node -> center fc
	ordered_neinodes = []
	for xc_i in range(len(a)):
		min_l = 100
		for n in b:
			value = eG.nodes[n]['fc']
			l = np.linalg.norm(value - a[xc_i])
			if l < min_l:
				min_l = l
				near_node = n

		ordered_neinodes.append(near_node) if near_node not in ordered_neinodes else  None
	return ordered_neinodes

def addxoo2edge(eG,main_frag_nodes,main_frag_nodes_fc,main_frag_edges,main_frag_edges_fc,sc_unit_cell):
	#quick check the order of xoo in every node are same 
    xoo_ind_node0 = xoo_pair_ind_node(main_frag_nodes_fc,main_frag_nodes[0],sc_unit_cell) #pick node one and get xoo_ind pair
    xoo_ind_node1 = xoo_pair_ind_node(main_frag_nodes_fc,main_frag_nodes[1],sc_unit_cell) #pick node two and get xoo_ind pair
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
        print(neighbor_nodes,'neighbor_nodes', i)
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
