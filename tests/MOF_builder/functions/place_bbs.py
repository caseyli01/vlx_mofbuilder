import numpy as np
from numpy.linalg import norm
from _Bio import SVDSuperimposer
from bbcif_properties import bb2array, X_vecs, bbbonds, bbcharges


import itertools
import re

def match_vectors(a1,a2,num):

	dist1 = [(np.linalg.norm(a1[0]-a1[i]),i) for i in range(len(a1))]
	dist2 = [(np.linalg.norm(a2[0]-a2[i]),i) for i in range(len(a2))]

	dist1.sort(key=lambda x: x[0])
	dist2.sort(key=lambda x: x[0])

	vecs1 = np.array([a1[i] for i in [dist1[j][1] for j in range(num)]])
	vecs2 = np.array([a2[i] for i in [dist2[j][1] for j in range(num)]])
	
	return vecs1,vecs2

def mag_superimpose(a1,a2):
	
	sup = SVDSuperimposer()

	a1 = np.asarray(a1)
	a2 = np.asarray(a2)
	mags = [norm(v) for v in a2]

	if len(a1) <= 7:

		min_dist = (1.0E6, 'foo', 'bar')
		
		for l in itertools.permutations(a1):

			p = np.array([m*v/norm(v) for m,v in zip(mags,l)])
			sup.set(a2,p)
			sup.run()
			rot,tran = sup.get_rotran()
			rms = sup.get_rms()

			if rms < min_dist[0]:
				min_dist = (rms,rot,tran)
	
	else:

		a1,a2 = match_vectors(a1,a2,6)
		mags = [norm(v) for v in a2]
		
		min_dist = (1.0E6, 'foo', 'bar')
		
		for l in itertools.permutations(a1):

			p = np.array([m*v/norm(v) for m,v in zip(mags,l)])
			sup.set(a2,p)
			sup.run()
			rot,tran = sup.get_rotran()
			rms = sup.get_rms()
		
			if rms < min_dist[0]:
				min_dist = (rms,rot,tran)

	return min_dist

def superimpose(a1,a2):
	sup = SVDSuperimposer()

	a1 = np.asarray(a1)
	a2 = np.asarray(a2)

	if len(a1) <= 7:

		min_dist = (1.0E6, 'foo', 'bar')
		
		for l in itertools.permutations(a1):

			p = np.asarray(l)
			sup.set(a2,p)
			sup.run()
			rot,tran = sup.get_rotran()
			rms = sup.get_rms()

			if rms < min_dist[0]:
				min_dist = (rms,rot,tran)
	
	else:

		a1,a2 = match_vectors(a1,a2,6)		
		min_dist = (1.0E6, 'foo', 'bar')
		
		for l in itertools.permutations(a1):

			p = np.asarray(l)
			sup.set(a2,p)
			sup.run()
			rot,tran = sup.get_rotran()
			rms = sup.get_rms()
		
			if rms < min_dist[0]:
				min_dist = (rms,rot,tran)

	return min_dist

def scaled_node_and_edge_vectors(sc_coords, sc_omega_plus, sc_unit_cell, ea_dict):

	nvecs = []
	evecs = []
	already_placed_edges = []
	nvecs_append = nvecs.append
	evecs_append = evecs.append
	already_placed_edges_append = already_placed_edges.append
	# add node_placed_edges for boundary edge and node check 
	node_placed_edges = []
	node_placed_edges_append = node_placed_edges.append
	for n in sc_coords:
		vertex,vcif,vfvec,indicent_edges = n
		vcvec = np.dot(sc_unit_cell, vfvec)

		ie = []
		ie_append = ie.append

		for e in indicent_edges:
			ind = e[0]
			positive_direction = e[1]
			ecif = e[2]

			if vertex == positive_direction[0]:
				direction = 1
				on = positive_direction[1]
			else:
				direction = -1
				on = positive_direction[0]

			dxn = ea_dict[vertex][ind][1]
			dxon = ea_dict[on][ind][1]

			ie_append((ind, direction, ecif, dxn, dxon))

		efvec = []
		ecvec = []
		efvec_append = efvec.append
		ecvec_append = ecvec.append
		#e_pair = []
		#e_pair_append = e_pair.append
		for e in ie:

			ind, d, ecif, dxn, dxon = e
			cs = np.dot(sc_unit_cell, sc_omega_plus[ind - 1])
			fvec = vfvec + d * sc_omega_plus[ind - 1]
			cvec = vcvec + d * cs

			ec1 = vcvec + d * dxn  * (cs/np.linalg.norm(cs))
			ec2 = cvec  - d * dxon * (cs/np.linalg.norm(cs))

			ecoords = np.average([ec1,ec2],axis=0)

			ecvec_append(cvec)
			efvec_append(fvec)
			#e_pair_append([[ec1,ec2,ind]])

			if ind not in already_placed_edges:
				evecs_append((ind, ecif, ecoords, np.array([vcvec,cvec])))
				already_placed_edges_append(ind)
				node_placed_edges_append((vertex,ind,vfvec,vcvec,positive_direction,ecoords, np.array([vcvec,cvec])))
		#print(f"n{n}\n,e_pair_append{e_pair}\n,already_placed_edges_append{already_placed_edges}")

		nvecs_append((vertex, vcvec, vcif, np.asarray(ecvec)))

	return nvecs, evecs,node_placed_edges

def place_nodes_ditopic(nvecs, nodes_dir):

	placed_nbb_coords = []
	placed_nbb_coords_extend = placed_nbb_coords.extend
	all_bonds = []
	all_bonds_extend = all_bonds.extend
	ind_seg = 0
	bbind = 1

	for n in nvecs:
		bbind = bbind + 1
		name,cvec,cif,nvec = n
		#ll = 0
		#for v in nvec:
		#	mag = np.linalg.norm(v - np.average(nvec, axis = 0))
		#	if mag > ll:
		#		ll = mag

		bbxvec = np.array(X_vecs(cif,nodes_dir,False))
		#if ORIENTATION_DEPENDENT_NODES:
		nbbxvec = bbxvec
		#else:
		#	nbbxvec = np.array([ll*(v / np.linalg.norm(v)) for v in bbxvec])

		min_dist,rot,tran = superimpose(nbbxvec,nvec)

		all_bb = bb2array(cif, nodes_dir)
		all_coords = np.array([v[1] for v in all_bb])
		all_inds = np.array([v[0] for v in all_bb])
		chg, elem = bbcharges(cif, nodes_dir)
		all_names = [o + re.sub('[A-Za-z]','',p) for o,p in zip(elem,all_inds)]
		#print(f'all_name{all_names}')

		all_names_indices = np.array([int(re.sub('[A-Za-z]','',e)) for e in all_names]) + ind_seg

		elem_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, elem):
			elem_dict[i] = j

		ind_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, all_names_indices):
			ind_dict[i] = j

		bonds = bbbonds(cif, nodes_dir)

		anf = [str(elem_dict[n]) + str(ind_dict[n]) for n in all_inds]

		abf = []
		for b in bonds:
			b1 = str(elem_dict[b[0]]) + str(ind_dict[b[0]])
			b2 = str(elem_dict[b[1]]) + str(ind_dict[b[1]])
			abf.append([b1,b2] + b[2:])

		aff_all = np.dot(all_coords,rot) + cvec
		
		laff_all = np.c_[anf, aff_all, chg, all_inds, [bbind] * len(anf)]
		
		placed_nbb_coords_extend(laff_all)
		all_bonds_extend(abf)
		ind_seg = ind_seg + len(all_names)

	return placed_nbb_coords, all_bonds




def place_nodes_tetra(nvecs, nodes_dir):
	placed_nbb_coords = []
	placed_edge_center_coords = []
	frame_nbb_coords =[]
	all_bonds = []
	tetra_node_name=[]
	ind_seg = 0
	bbind = 1

	for n in nvecs:
		bbind = bbind + 1
		name,cvec,cif,nvec = n
		#ll = 0
		#
		#for v in nvec:
		#	mag = np.linalg.norm(v - np.average(nvec, axis = 0))
		#	if mag > ll:
		#		ll = mag

		bbxvec = np.array(X_vecs(cif,nodes_dir,False))
		#if ORIENTATION_DEPENDENT_NODES:
		nbbxvec = bbxvec
		#else:
		#	nbbxvec = np.array([ll*(v / np.linalg.norm(v)) for v in bbxvec])

		min_dist,rot,tran = superimpose(nbbxvec,nvec)

		all_bb = bb2array(cif, nodes_dir)
		all_coords = np.array([v[1] for v in all_bb])
		all_inds = np.array([v[0] for v in all_bb])
		chg, elem = bbcharges(cif, nodes_dir)
		all_names = [o + re.sub('[A-Za-z]','',p) for o,p in zip(elem,all_inds)]
		#print(f'all_name{all_names}')

		all_names_indices = np.array([int(re.sub('[A-Za-z]','',e)) for e in all_names]) + ind_seg

		elem_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, elem):
			elem_dict[i] = j

		ind_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, all_names_indices):
			ind_dict[i] = j

		bonds = bbbonds(cif, nodes_dir)

		anf = [str(elem_dict[n]) + str(ind_dict[n]) for n in all_inds]

		abf = []
		for b in bonds:
			b1 = str(elem_dict[b[0]]) + str(ind_dict[b[0]])
			b2 = str(elem_dict[b[1]]) + str(ind_dict[b[1]])
			abf.append([b1,b2] + b[2:])

		aff_all = np.dot(all_coords,rot) + cvec
		
		laff_all = np.c_[anf, aff_all, chg, all_inds, [bbind] * len(anf)]
		if "tetracenter" in cif:
			placed_edge_center_coords.extend(laff_all)
			placed_nbb_coords.extend(laff_all)
			tetra_node_name.append(name)
		else:
			frame_nbb_coords.extend(laff_all)
			placed_nbb_coords.extend(laff_all)
		all_bonds.extend(abf)
		ind_seg = ind_seg + len(all_names)

	return placed_nbb_coords, placed_edge_center_coords,frame_nbb_coords,tetra_node_name,all_bonds





def place_nodes_tri(nvecs, nodes_dir):

	placed_nbb_coords = []
	placed_nbb_coords_extend = placed_nbb_coords.extend
	placed_edge_center_coords = []
	placed_edge_center_coords_extend = placed_edge_center_coords.extend
	frame_nbb_coords =[]
	frame_nbb_coords_extend = frame_nbb_coords.extend
	all_bonds = []
	all_bonds_extend = all_bonds.extend
	tri_node_name=[]
	tri_node_name_append= tri_node_name.append
	ind_seg = 0
	bbind = 1

	for n in nvecs:
		bbind = bbind + 1
		name,cvec,cif,nvec = n
		#ll = 0
		#
		#for v in nvec:
		#	mag = np.linalg.norm(v - np.average(nvec, axis = 0))
		#	if mag > ll:
		#		ll = mag

		bbxvec = np.array(X_vecs(cif,nodes_dir,False))
		#if ORIENTATION_DEPENDENT_NODES:
		nbbxvec = bbxvec
		#else:
		#	nbbxvec = np.array([ll*(v / np.linalg.norm(v)) for v in bbxvec])

		min_dist,rot,tran = superimpose(nbbxvec,nvec)

		all_bb = bb2array(cif, nodes_dir)
		all_coords = np.array([v[1] for v in all_bb])
		all_inds = np.array([v[0] for v in all_bb])
		chg, elem = bbcharges(cif, nodes_dir)
		all_names = [o + re.sub('[A-Za-z]','',p) for o,p in zip(elem,all_inds)]
		#print(f'all_name{all_names}')

		all_names_indices = np.array([int(re.sub('[A-Za-z]','',e)) for e in all_names]) + ind_seg

		elem_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, elem):
			elem_dict[i] = j

		ind_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, all_names_indices):
			ind_dict[i] = j

		bonds = bbbonds(cif, nodes_dir)

		anf = [str(elem_dict[n]) + str(ind_dict[n]) for n in all_inds]

		abf = []
		for b in bonds:
			b1 = str(elem_dict[b[0]]) + str(ind_dict[b[0]])
			b2 = str(elem_dict[b[1]]) + str(ind_dict[b[1]])
			abf.append([b1,b2] + b[2:])

		aff_all = np.dot(all_coords,rot) + cvec
		
		laff_all = np.c_[anf, aff_all, chg, all_inds, [bbind] * len(anf)]
		if "tricenter" in cif:
			placed_edge_center_coords_extend(laff_all)
			placed_nbb_coords_extend(laff_all)
			tri_node_name_append(name)
		else:
			frame_nbb_coords_extend(laff_all)
			placed_nbb_coords_extend(laff_all)
		all_bonds_extend(abf)
		ind_seg = ind_seg + len(all_names)

	return placed_nbb_coords, placed_edge_center_coords,frame_nbb_coords,tri_node_name,all_bonds



def place_edges(evecs, edges_dir,CHARGES,nnodes):

	placed_ebb_coords = []
	placed_ebb_coords_extend = placed_ebb_coords.extend
	all_bonds = []
	all_bonds_extend = all_bonds.extend
	ind_seg = nnodes
	bbind = -1
	#NOTE: bbind = -1
	for e in evecs:

		bbind = bbind - 1
		index,cif,ecoords,evec=e
		ll = 0
		#adjust evec length based on edge.cif
		for v in evec:
			mag = np.linalg.norm(v - np.average(evec, axis = 0))
			if mag > ll:
				ll = mag	

		bbxvec  = np.array(X_vecs(cif,edges_dir,False))
		nbbxvec = np.array([ll*(v / np.linalg.norm(v)) for v in bbxvec])

		min_dist,rot,tran = superimpose(nbbxvec,evec)

		all_bb = bb2array(cif, 'edges')
		all_coords = np.array([v[1] for v in all_bb])
		all_inds = np.array([v[0] for v in all_bb])
		chg, elem = bbcharges(cif, 'edges')
		all_names = [o + re.sub('[A-Za-z]','',p) for o,p in zip(elem,all_inds)]

		all_names_indices = np.array([int(re.sub('[A-Za-z]','',e)) for e in all_names]) + ind_seg

		elem_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, elem):
			elem_dict[i] = j

		ind_dict = dict((k,'') for k in all_inds)
		for i,j in zip(all_inds, all_names_indices):
			ind_dict[i] = j

		bonds = bbbonds(cif, 'edges')
		anf = [str(elem_dict[n]) + str(ind_dict[n]) for n in all_inds]
		abf = []
		for b in bonds:
			b1 = str(elem_dict[b[0]]) + str(ind_dict[b[0]])
			b2 = str(elem_dict[b[1]]) + str(ind_dict[b[1]])
			abf.append([b1,b2] + b[2:])

		'''!!!'''
		aff_all = np.dot(all_coords,rot) + ecoords
		laff_all = np.c_[anf, aff_all, chg, all_inds, [bbind] * len(anf)]  
		#print(f"anf, aff_all, chg, all_inds, [bbind] * len(anf){anf, aff_all, chg, all_inds, [bbind] * len(anf)}")
		placed_ebb_coords_extend(laff_all)
		all_bonds_extend(abf)
		ind_seg = ind_seg + len(all_names)

	return placed_ebb_coords, all_bonds
