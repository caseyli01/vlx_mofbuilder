import networkx as nx
import numpy as np
import re

def cycle_cocyle(TG):
	'''
	The cycle_cocycle function calculates the cycle basis and node-associated edges 
	for a given graph TG. This function leverages concepts from graph theory, 
	specifically the Minimum Spanning Tree (MST) and cycles within the graph.
	The cycle_cocyle function performs the following steps:

	Computes the Minimum Spanning Tree (MST) of the target graph TG.
	Initializes a scaffold graph and adds MST edges to it.
	Iterates through all edges in TG to find cycles and constructs the cycle basis.
	Identifies and stores edges associated with each node in the target graph.
	Returns the cycle basis and node-associated edges.
	This function is useful for understanding the fundamental cycles in a graph and 
	the connectivity of nodes through their associated edges.
	@return:
	cycle_basis: List of cycles found in the graph.
	node_out_edges: List of edges associated with each node
	'''
	MST = nx.minimum_spanning_tree(TG)
	scaffold = nx.MultiGraph()

	used_keys = []
	for e in MST.edges(data=True):
		edict = e[2]
		lbl = edict['label']
		ind = edict['index']
		ke = (ind,lbl[0],lbl[1],lbl[2])
		scaffold.add_edge(e[0],e[1],key=ke)
		used_keys.append(ke)

	cycle_basis = []
	cycle_basis_append = cycle_basis.append
	nxfc = nx.find_cycle

	for e0 in TG.edges(data=True):

		edict = e0[2]
		lbl = edict['label']
		ind = edict['index']
		ke = (ind,lbl[0],lbl[1],lbl[2])
		scaffold.add_edge(e0[0],e0[1],key=ke)

		if ke not in used_keys:

			cycles = list(nxfc(scaffold))
			cy_list = [(i[0], i[1], i[2]) for i in cycles]

			if cy_list not in cycle_basis:
				cycle_basis_append(cy_list)

			scaffold.remove_edge(e0[0],e0[1],key=ke)

	node_out_edges = []
	node_out_edges_append = node_out_edges.append
	node_list = list(TG.nodes())
	
	for n in range(len(TG.nodes()) - 1):

		node = node_list[n]
		noe = [node]

		for e in TG.edges(data=True):
			if node == e[0] or node == e[1]:
				edict = e[2]
				lbl = edict['label']
				ke = (edict['index'],lbl[0],lbl[1],lbl[2])
				positive_direction = edict['pd']
				noe.append(positive_direction + ke)

		node_out_edges_append(noe)

	return cycle_basis, node_out_edges

def Bstar_alpha(CB, CO, TG, num_edges):
	'''
	The Bstar_alpha function constructs the cycle and cocycle basis matrices (Bstar and alpha) 
	for a given graph TG based on its cycle basis (CB) and node out edges (CO).

	CB: Cycle basis, a list of cycles where each cycle is a list of edges.
	CO: Node out edges, a list of nodes where each node has a list of outgoing edges.
	TG: A networkx graph object representing the target graph.
	num_edges: The number of edges in the graph TG
	'''
	#edge_keys: A dictionary to map each edge in the graph to its attributes using a tuple of its start node, end node, and index.
	edge_keys = dict(((k[0],k[1],k[2]['index']),[]) for k in TG.edges(data=True))

	for e in TG.edges(keys=True, data=True):
		edge_keys[(e[0],e[1],e[3]['index'])] = e[2] 

	#Bstar: List to store the cycle and cocycle basis vectors.
	#a: List to store the net voltages for cycles and zero vectors for cocycles.
	#q: Counter for cycles.
	Bstar = []
	a = []
	Bstar_append = Bstar.append
	a_append = a.append
	q = 0

	#Constructing Cycle Basis Vectors
	'''
	For each cycle in CB:
	Initialize a cycle vector cycle_vec of zeros with length equal to the number of edges.
	Initialize net_voltage to a zero vector.
	For each edge in the cycle:
	Extract the start node, end node, and label (containing the index and voltage).
	Determine the edge direction using positive_direction.
	Assign the direction (+1 or -1) to the appropriate position in cycle_vec.
	Update net_voltage based on the edge direction and voltage.
	Append the constructed cycle_vec and net_voltage to Bstar and a, respectively.
	'''
	for cycle in CB:

		q += 1
		cycle_vec = [0] * num_edges
		net_voltage = np.array([0,0,0])

		for edge in cycle:

			s,e,lv = edge
			ind = lv[0]
			voltage = np.asarray(lv[1:])

			try:
				key = edge_keys[(s,e,ind)]
			except:
				key = edge_keys[(e,s,ind)]

			positive_direction = TG[s][e][key]['pd']

			if (s,e) == positive_direction:
				direction = 1
			elif (e,s) == positive_direction:
				direction = -1
			else:
				raise ValueError('Error in B* cycle vector construction, edge direction cannot be defined for:',s,e)

			cycle_vec[ind - 1] = direction
			net_voltage = net_voltage + (direction * voltage)

		Bstar_append(cycle_vec)
		a_append(net_voltage)

	
	#Constructing Cocycle Basis Vectors
	'''
	For each vertex in CO (sorted by numerical order):
	Initialize a cocycle vector cocycle_vec of zeros.
	Extract the vertex name and outgoing edges (ooa).
	For each outgoing edge:
	Determine the edge direction and assign it to the appropriate position in cocycle_vec.
	Check the direction assignment to ensure it follows the convention.
	Append the constructed cocycle_vec to Bstar and a zero vector to a.

	'''
	for vertex in sorted(CO, key = lambda x: int(re.sub('[A-Za-z]','',x[0]))):
		cocycle_vec = [0] * num_edges
		v = vertex[0]
		ooa = [[i[2],(i[0],i[1]),np.array(i[3:])] for i in vertex[1:]]
		for out_edge in ooa:
			ke = (out_edge[0], out_edge[2][0], out_edge[2][1], out_edge[2][2])
			ind = out_edge[0]
			s,e = out_edge[1]
			positive_direction = TG[s][e][ke]['pd'] 
			if '_a' not in s and '_a' not in e:
				if s == v:
					o = e
					print('s == v &o==e',s,v,o,e)
				else:
					o = s
					print('s == v&o==s',s,v,o,e)
				v_ind = int(re.sub('[A-Za-z]','',v))
				o_ind = int(re.sub('[A-Za-z]','',o))

				if v_ind < o_ind:
					cd = 1
				else:
					cd = -1
				if v == s:
					direction = 1
				else:
					direction = -1
				if direction != cd:
					raise ValueError('The direction assignment for the co-cycle vector', s,e, 'may is incorrect... \n The direction assignment does not follow the low-index to high-index = positive convention')
			
			cocycle_vec[ind - 1] = direction

		Bstar_append(cocycle_vec)
		a_append(np.array([0,0,0]))
		
	#Final Check and Return
	if len(Bstar) != len(a):
		raise ValueError('Error in cycle_cocycle.py, the row ranks of Bstar and alpha do not match.')

	return np.asarray(Bstar), np.asarray(a)
