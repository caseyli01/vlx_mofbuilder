from bbcif_properties import calc_edge_len
import numpy as np

def SBU_coords(TG, ea_dict, csbl):
	'''
	The SBU_coords function calculates the coordinates of Secondary Building Units (SBUs) in a given graph TG. 
	This function uses edge assignments and the lengths of edges to determine the spatial configuration of SBUs around each node.

	TG: A networkx graph object representing the target graph.
	ea_dict: Edge assignment dictionary containing vector information for each node.
	csbl: Constant to be added to the calculated edge lengths, CONNECTION_SITE_BOND_LENGTH.
	'''
	SBU_coords = [] #SBU_coords: List to store the coordinates of SBUs.
	SBU_coords_append = SBU_coords.append

	for node in TG.nodes(data=True):

		vertex = node[0]
		xvecs = [] #xvecs: List to store vectors associated with the current vertex.
		xvecs_append = xvecs.append
		
		#Loop through each edge in the graph.
		#If the current vertex is part of the edge, proceed to calculate the required vectors.
		#ecif: CIF file name associated with the edge.
		#positive_direction: Directional information for the edge.
		#ind: Index of the edge.
		#length: Length of the edge, calculated using calc_edge_len.
		#direction: Determines if the edge direction is positive or negative relative to the vertex.
		#ov: Opposite vertex in the edge.
		#xvecname, dx_v, xvec: Vector information from the edge assignment dictionary.
		#dx_ov: Distance vector for the opposite vertex.
		
		for e0, e1, edict in TG.edges(data=True):

			if vertex in (e0,e1):

				ecif = edict['cifname']
				positive_direction = edict['pd']
				ind = edict['index']
				length = calc_edge_len(ecif,'edges')

				if vertex == positive_direction[0]:
					direction = 1
					ov = positive_direction[1]
				else:
					direction = -1
					ov = positive_direction[0]
				
				xvecname,dx_v,xvec = ea_dict[vertex][ind]
				dx_ov = ea_dict[ov][ind][1]

				if length < 0.1:
					total_length = dx_v + dx_ov + csbl
				else:
					total_length = dx_v + dx_ov + length + 2*csbl
				
				svec = (xvec/np.linalg.norm(xvec)) * total_length * direction
				xvecs_append([ind, svec])

		SBU_coords_append((vertex, xvecs))

	return SBU_coords
