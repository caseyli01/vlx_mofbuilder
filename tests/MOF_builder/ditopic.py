import os
import networkx as nx
import numpy as np
import re
import glob
import py3Dmol as p3d
from functions.ciftemplate2graph import ct2g
from functions.vertex_edge_assign import vertex_assign, assign_node_vecs2edges
from functions.cycle_cocyle import cycle_cocyle, Bstar_alpha
from functions.bbcif_properties import cncalc, bbelems
from functions.SBU_geometry import SBU_coords
from functions.scale import scale
from functions.scaled_embedding2coords import omega2coords
from functions.place_bbs import scaled_node_and_edge_vectors,place_edges,place_nodes_ditopic
#from functions.remove_net_charge import fix_charges
from functions.remove_dummy_atoms import remove_Fr
from functions.adjust_edges import adjust_edges,addxoo2edge,superimpose
from functions.write_cifs import write_cif_nobond, merge_catenated_cifs #write_check_cif,bond_connected_components, distance_search_bond, fix_bond_sym, merge_catenated_cifs
#from functions.scale_animation import scaling_callback_animation, write_scaling_callback_animation, animate_objective_minimization
import itertools
#from random import choice
from functions.cluster import cluster_supercell,placed_arr
from functions.supercell import find_new_node_beginning,Carte_points_generator
from functions.output import tempgro,temp_xyz,viewgro#,viewxyz
from functions.terminations import terminate_nodes,terminate_unsaturated_edges,add_node_terminations,exposed_xoo_cc,Xpdb,terminate_unsaturated_edges_CCO2
#from functions.filtX import filt_nodex_fvec,filt_closest_x_angle,filt_outside_edgex
#unctions.multitopic import 
from functions.isolated_node_cleaner import reindex_frag_array,get_frag_centers_fc,calculate_eG_net_ditopic
from functions.replace import fetch_by_idx_resname,sub_pdb




####### Global options #######
import configuration

pi = np.pi

vname_dict = {'V':1,'Er':2,'Ti':3,'Ce':4,'S':5,
			  'H':6,'He':7,'Li':8,'Be':9,'B':10,
			  'C':11,'N':12,'O':13,'F':14,'Ne':15,
			  'Na':16,'Mg':17,'Al':18,'Si':19,'P':20 ,
			  'Cl':21,'Ar':22,'K':23,'Ca':24,'Sc':24,
			  'Cr':26,'Mn':27,'Fe':28,'Co':29,'Ni':30}

metal_elements = ['Ac','Ag','Al','Am','Au','Ba','Be','Bi',
				  'Bk','Ca','Cd','Ce','Cf','Cm','Co','Cr',
				  'Cs','Cu','Dy','Er','Es','Eu','Fe','Fm',
				  'Ga','Gd','Hf','Hg','Ho','In','Ir',
				  'K','La','Li','Lr','Lu','Md','Mg','Mn',
				  'Mo','Na','Nb','Nd','Ni','No','Np','Os',
				  'Pa','Pb','Pd','Pm','Pr','Pt','Pu','Ra',
				  'Rb','Re','Rh','Ru','Sc','Sm','Sn','Sr',
				  'Ta','Tb','Tc','Th','Ti','Tl','Tm','U',
				  'V','W','Y','Yb','Zn','Zr']


####### Global options #######
IGNORE_ALL_ERRORS = configuration.IGNORE_ALL_ERRORS
#PRINT = configuration.PRINT
PRINT =True
CONNECTION_SITE_BOND_LENGTH = configuration.CONNECTION_SITE_BOND_LENGTH
WRITE_CHECK_FILES = configuration.WRITE_CHECK_FILES
WRITE_CIF = configuration.WRITE_CIF
ALL_NODE_COMBINATIONS = configuration.ALL_NODE_COMBINATIONS
USER_SPECIFIED_NODE_ASSIGNMENT = configuration.USER_SPECIFIED_NODE_ASSIGNMENT
COMBINATORIAL_EDGE_ASSIGNMENT = configuration.COMBINATORIAL_EDGE_ASSIGNMENT
#CHARGES = configuration.CHARGES
CHARGES = False
SCALING_ITERATIONS = configuration.SCALING_ITERATIONS
SYMMETRY_TOL = configuration.SYMMETRY_TOL
BOND_TOL = configuration.BOND_TOL
ORIENTATION_DEPENDENT_NODES = configuration.ORIENTATION_DEPENDENT_NODES
PLACE_EDGES_BETWEEN_CONNECTION_POINTS = configuration.PLACE_EDGES_BETWEEN_CONNECTION_POINTS
RECORD_CALLBACK = configuration.RECORD_CALLBACK
OUTPUT_SCALING_DATA = configuration.OUTPUT_SCALING_DATA
FIX_UC = configuration.FIX_UC
MIN_CELL_LENGTH = configuration.MIN_CELL_LENGTH
OPT_METHOD = configuration.OPT_METHOD
PRE_SCALE = configuration.PRE_SCALE
SINGLE_METAL_MOFS_ONLY = configuration.SINGLE_METAL_MOFS_ONLY
MOFS_ONLY = configuration.MOFS_ONLY
MERGE_CATENATED_NETS = configuration.MERGE_CATENATED_NETS
RUN_PARALLEL = configuration.RUN_PARALLEL
REMOVE_DUMMY_ATOMS = configuration.REMOVE_DUMMY_ATOMS


class MOF_ditopic:
	def __init__(self,templates_dir,nodes_dir,edges_dir,template,node_topics):
		self.templates_dir = templates_dir
		self.nodes_dir = nodes_dir
		self.edges_dir = edges_dir
		self.template = template
		self.linker_topics = 2 #ditopic class 
		self.node_topics = node_topics
	
	def load(self,WRITE_CIF):

		templates_dir = self.templates_dir 
		nodes_dir = self.nodes_dir 
		edges_dir = self.edges_dir 
		template = self.template 
		
		PRINT=False
		print()
		print('=========================================================================================================')
		print('template :',template)                                          
		print('=========================================================================================================')
		print()

		cat_count = 0
		for net in ct2g(template,templates_dir):

				cat_count += 1
				TG, start, unit_cell, TVT, TET, TNAME, a, b, c, ang_alpha, ang_beta, ang_gamma, max_le, catenation = net

				TVT = sorted(TVT, key=lambda x:x[0], reverse=True) # sort node with connected degree, the first one is the highest(full)-coordinated node
				TET = sorted(TET, reverse=True) #sort node_pair by the node_index
				#get node cif information from node dir
				print(os.listdir(nodes_dir))
				node_cns = [(cncalc(node, nodes_dir), node) for node in os.listdir(nodes_dir)]

				print('Number of vertices = ', len(TG.nodes()))
				print('Number of edges = ', len(TG.edges()))
				print()

				edge_counts = dict((data['type'],0) for e0,e1,data in TG.edges(data=True))
				for e0,e1,data in TG.edges(data=True):
					edge_counts[data['type']] += 1
				
				if PRINT:
			
					print('There are', len(TG.nodes()), 'vertices in the voltage graph:')
					print()
					v = 0
			
					for node in TG.nodes():
						v += 1
						print(v,':',node)
						node_dict = TG.nodes[node]
						print('type : ', node_dict['type'])
						print('cartesian coords : ', node_dict['ccoords'])
						print('fractional coords : ', node_dict['fcoords'])
						#print('degree : ', node_dict['cn'][0])
						print()
			
					print('There are', len(TG.edges()), 'edges in the voltage graph:')
					print()
			
					for edge in TG.edges(data=True,keys=True):
						edge_dict = edge[3]
						ind = edge[2]
						print(ind,':',edge[0],edge[1])
						print('length : ',edge_dict['length'])
						print('type : ',edge_dict['type'])
						print('label : ',edge_dict['label'])
						print('positive direction :',edge_dict['pd'])
						print('cartesian coords : ',edge_dict['ccoords'])
						print('fractional coords : ',edge_dict['fcoords'])
						print()
			
				vas = vertex_assign(nodes_dir,TG, TVT, node_cns, unit_cell, USER_SPECIFIED_NODE_ASSIGNMENT, SYMMETRY_TOL, ALL_NODE_COMBINATIONS)
				CB,CO = cycle_cocyle(TG)

				for va in vas:
					if len(va) == 0:
						print('At least one vertex does not have a building block with the correct number of connection sites.')
						print('Moving to the next template...')
						print()
						continue
			
				if len(CB) != (len(TG.edges()) - len(TG.nodes()) + 1):
					print('The cycle basis is incorrect.')
					print('The number of cycles in the cycle basis does not equal the rank of the cycle space.')
					print('Moving to the next template...')
					continue
				
				num_edges = len(TG.edges())
				Bstar, alpha = Bstar_alpha(CB,CO,TG,num_edges)

				if PRINT:
					print('B* (top) and alpha (bottom) for the barycentric embedding are:')
					print()
					for i in Bstar:
						print(i)
					print()
					for i in alpha:
						print(i)
					print()
			
				num_vertices = len(TG.nodes())
			
				if COMBINATORIAL_EDGE_ASSIGNMENT:
					eas = list(itertools.product([e for e in os.listdir(edges_dir)], repeat = len(TET)))
				else:
					edge_files = sorted([e for e in os.listdir(edges_dir)])
					eas = []
					i = 0
					while len(eas) < len(TET):
						eas.append(edge_files[i])
						i += 1
						if i == len(edge_files):
							i = 0
					eas = [eas]
			
				g = 0

				for va in vas:
					#check if assigned node has metal element 
					node_elems = [bbelems(i[1], nodes_dir) for i in va]
					metals = [[i for i in j if i in metal_elements] for j in node_elems]
					metals = list(set([i for j in metals for i in j]))
					#set node cif files as vertex assignment
					v_set0 = [('v' + str(vname_dict[re.sub('[0-9]','',i[0])]), i[1]) for i in va]
					v_set1 = sorted(list(set(v_set0)), key=lambda x: x[0])
					v_set = [v[0] + '-' + v[1] for v in v_set1]
			
					print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
					print('vertex assignment : ',v_set)
					print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
					print()

					if SINGLE_METAL_MOFS_ONLY and len(metals) != 1:
						print(v_set, 'contains no metals or multiple metal elements, no cif will be written')
						print()
						continue

					if MOFS_ONLY and len(metals) < 1:
						print(v_set, 'contains no metals, no cif will be written')
						print()
						continue
					
					# add cifname to TG.nodes
					for v in va:
						for n in TG.nodes(data=True):
							if v[0] == n[0]:
								n[1]['cifname'] = v[1]
					
					for ea in eas:
			
						g += 1
			
						print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
						print('edge assignment : ',ea)
						print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
						print()
						
						type_assign = dict((k,[]) for k in sorted(TET, reverse=True))
						for k,m in zip(TET,ea):
							type_assign[k] = m
						
						# add cifname to TG.edge
						for e in TG.edges(data=True):
							ty = e[2]['type']
							for k in type_assign:
								if ty == k or (ty[1],ty[0]) == k:
									e[2]['cifname'] = type_assign[k]

						num_possible_XX_bonds = 0
						for edge_type, cifname in zip(TET, ea):
							if cifname == 'ntn_edge.cif':
								factor = 1
							else:
								factor = 2
							edge_type_count = edge_counts[edge_type]
							num_possible_XX_bonds += factor * edge_type_count

						ea_dict = assign_node_vecs2edges(nodes_dir,TG, unit_cell, SYMMETRY_TOL, template)
						all_SBU_coords = SBU_coords(TG, ea_dict, CONNECTION_SITE_BOND_LENGTH)
						sc_a, sc_b, sc_c, sc_alpha, sc_beta, sc_gamma, sc_covar, Bstar_inv, max_length, callbackresults, ncra, ncca, scaling_data = scale(all_SBU_coords,a,b,c,ang_alpha,ang_beta,ang_gamma,max_le,num_vertices,Bstar,alpha,num_edges,FIX_UC,SCALING_ITERATIONS,PRE_SCALE,MIN_CELL_LENGTH,OPT_METHOD)
				
						print('*******************************************')
						print('The scaled unit cell parameters are : ')
						print('*******************************************')
						print('a    :', np.round(sc_a, 5))
						print('b    :', np.round(sc_b, 5))
						print('c    :', np.round(sc_c, 5))
						print('alpha:', np.round(sc_alpha, 5))
						print('beta :', np.round(sc_beta, 5))
						print('gamma:', np.round(sc_gamma, 5))
						print()
			
						for sc, name in zip((sc_a, sc_b, sc_c), ('a', 'b', 'c')):
							cflag = False
							if sc == MIN_CELL_LENGTH:
								print('unit cell parameter', name, 'may have collapsed during scaling!')
								print('try re-running with', name, 'fixed or a larger MIN_CELL_LENGTH')
								print('no cif will be written')
								cflag = True
			
						if cflag:
							continue
			
						scaled_params = [sc_a,sc_b,sc_c,sc_alpha,sc_beta,sc_gamma]
					
						sc_Alpha = np.r_[alpha[0:num_edges-num_vertices+1,:], sc_covar]
						sc_omega_plus = np.dot(Bstar_inv, sc_Alpha)
					
						ax = sc_a
						ay = 0.0
						az = 0.0
						bx = sc_b * np.cos(sc_gamma * pi/180.0)
						by = sc_b * np.sin(sc_gamma * pi/180.0)
						bz = 0.0
						cx = sc_c * np.cos(sc_beta * pi/180.0)
						cy = (sc_c * sc_b * np.cos(sc_alpha * pi/180.0) - bx * cx) / by
						cz = (sc_c ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5
						sc_unit_cell = np.asarray([[ax,ay,az],[bx,by,bz],[cx,cy,cz]]).T
						
						scaled_coords = omega2coords(start, TG, sc_omega_plus, (sc_a,sc_b,sc_c,sc_alpha,sc_beta,sc_gamma), num_vertices,templates_dir, template, g, WRITE_CHECK_FILES)
						nvecs,evecs,node_placed_edges = scaled_node_and_edge_vectors(scaled_coords, sc_omega_plus, sc_unit_cell, ea_dict)
						placed_nodes, node_bonds = place_nodes_ditopic(nvecs, nodes_dir)
						placed_edges, edge_bonds = place_edges(evecs, edges_dir,CHARGES,len(placed_nodes))


						placed_edges= adjust_edges(placed_edges, placed_nodes, sc_unit_cell)
						
						# add classifination 
	
						# add classifination 
						placed_nodes = np.c_[placed_nodes, np.full((len(placed_nodes),1),'NODE')]
						placed_edges = np.c_[placed_edges, np.full((len(placed_edges),1),'EDGE')]

						placed_all = list(placed_nodes) + list(placed_edges)
						bonds_all = node_bonds + edge_bonds
						
				
						##if WRITE_CHECK_FILES:
						##	write_check_cif(template, placed_nodes, placed_edges, g, scaled_params, sc_unit_cell)
					
						if REMOVE_DUMMY_ATOMS:
							placed_all, bonds_all, nconnections = remove_Fr(placed_all,bonds_all)
						
						##print('computing X-X bonds...')
						##print()
						##print('*******************************************')
						##print('Bond formation : ')
						##print('*******************************************')
						##
						###fixed_bonds, nbcount, bond_check_passed = bond_connected_components(placed_all, bonds_all, sc_unit_cell, max_length, BOND_TOL, nconnections, num_possible_XX_bonds)
						#print('there were ', nbcount, ' X-X bonds formed')
						##bond_check_passed =False
						##if bond_check_passed:
						##	print('bond check passed')
						##	bond_check_code = ''
						##else:
						##	print('bond check failed, attempting distance search bonding...')
						##	fixed_bonds, nbcount = distance_search_bond(placed_all, bonds_all, sc_unit_cell, 2.5)
						##	bond_check_code = '_BOND_CHECK_FAILED'
						##	print('there were', nbcount, 'X-X bonds formed')
						##print()
				
						##if CHARGES:
						##	fc_placed_all, netcharge, onetcharge, rcb = fix_charges(placed_all)
						##else:
						##	fc_placed_all = placed_all
					##
						##fc_placed_all = placed_all
						##fixed_bonds = fix_bond_sym(fixed_bonds, placed_all, sc_unit_cell)
			##
						##if CHARGES:
						##	print('*******************************************')
						##	print('Charge information :                       ')
						##	print('*******************************************')
						##	print('old net charge                  :', np.round(onetcharge, 5))
						##	print('rescaling magnitude             :', np.round(rcb, 5))
					##
						##	remove_net = choice(range(len(fc_placed_all)))
						##	fc_placed_all[remove_net][4] -= np.round(netcharge, 4)
					##
						##	print('new net charge (after rescaling):', np.sum([li[4] for li in fc_placed_all]))
						##	print()

						vnames = '_'.join([v.split('.')[0] for v in v_set])
						enames_list = [e[0:-4] for e in ea]
						enames_grouped = [list(edge_gr) for ind,edge_gr in itertools.groupby(enames_list)]
						enames_grouped = [(len(edge_gr), list(set(edge_gr))) for edge_gr in enames_grouped]
						enames_flat = [str(L) + '-' + '_'.join(names) for L,names in enames_grouped]
						enames = '_'.join(enames_flat)
						bond_check_code = 'nobond'

						if catenation:
							outcifname = template[0:-4] + '_' +  vnames + '_' + enames + bond_check_code + '_' + 'CAT' + str(cat_count) + '.cif'
						else:
							outcifname = template[0:-4] + '_' +  vnames + '_' + enames + bond_check_code + '.cif'
				
						##if WRITE_CIF:
						##	print('writing cif...')
						##	print()
						##	if len(cifname) > 255:
						##		cifname = cifname[0:241]+'_truncated.cif'
						##	write_cif(fc_placed_all, fixed_bonds, scaled_params, sc_unit_cell, outcifname, CHARGES, wrap_coords=False)
						if WRITE_CIF:
							print('writing cif...')
							print()
							if len(cifname) > 255:
								cifname = cifname[0:241]+'_truncated.cif'
							write_cif_nobond(placed_all, scaled_params, sc_unit_cell, outcifname, CHARGES, wrap_coords=False)

		if catenation and MERGE_CATENATED_NETS:
			
			print('merging catenated cifs...')
			cat_cifs = glob.glob('output_cifs/*_CAT*.cif')

			for comb in itertools.combinations(cat_cifs, cat_count):

				builds = [name[0:-9] for name in comb]

				print(set(builds))

				if len(set(builds)) == 1:
					pass
				else:
					continue

				merge_catenated_cifs(comb, CHARGES)

			for cif in cat_cifs:
				os.remove(cif)

		self.TG = TG
		self.placed_all = placed_all
		self.sc_unit_cell = sc_unit_cell
		self.placed_nodes = placed_nodes
		self.placed_edges = placed_edges

	def basic_supercell(self,supercell,term_file = 'data/methyl.pdb',boundary_cut_buffer = 0.00,edge_center_check_buffer = 0.0,cutxyz=[True,True,True]):
		linker_topics = self.linker_topics
		cutx,cuty,cutz = cutxyz
		TG = self.TG
		scalar = boundary_cut_buffer
		boundary_scalar = edge_center_check_buffer
		placed_edges = self.placed_edges
		placed_nodes = self.placed_nodes
		sc_unit_cell = self.sc_unit_cell
		
		frame_node_name= list(TG.nodes())
		frame_node_fc=np.asarray([TG.nodes[fn]['fcoords']for fn in frame_node_name])
    
		new_beginning_fc = find_new_node_beginning(frame_node_fc)		
		placed_nodes_arr,nodes_id=placed_arr(placed_nodes)
		placed_edges_arr,edges_id=placed_arr(placed_edges)		
		placed_nodes_fc = np.hstack((placed_nodes_arr[:,0:1],np.dot(np.linalg.inv(sc_unit_cell),placed_nodes_arr[:,1:4].T).T-new_beginning_fc,placed_nodes_arr[:,4:]))
		placed_edges_fc = np.hstack((placed_edges_arr[:,0:1],np.dot(np.linalg.inv(sc_unit_cell),placed_edges_arr[:,1:4].T).T-new_beginning_fc,placed_edges_arr[:,4:]))		
			
		target_all_fc = np.vstack((placed_nodes_fc,placed_edges_fc))
        #target_all_fc = np.vstack((placed_nodes_fc,tetratopic_edges_fcoords)) # the reason for use above version node is because we need xoo in node for terminations adding
		box_bound= supercell+1
		supercell_Carte = Carte_points_generator(supercell)		
		connected_nodeedge_fc, boundary_connected_nodes_res,eG,bare_nodeedge_fc_loose=cluster_supercell(sc_unit_cell,supercell_Carte,linker_topics,target_all_fc,box_bound,scalar,cutx,cuty,cutz,boundary_scalar)		
		terms_cc_loose = terminate_nodes(term_file,boundary_connected_nodes_res,connected_nodeedge_fc,sc_unit_cell,box_bound)

		connected_nodeedge_cc = np.hstack((connected_nodeedge_fc[:,:-3],np.dot(sc_unit_cell,connected_nodeedge_fc[:,-3:].T).T))
		#print(connected_nodeedge_cc.shape,terms_cc_loose.shape)

		node_edge_term_cc_loose = np.vstack((connected_nodeedge_cc,terms_cc_loose))		
		self.all_connected_node_edge_cc = connected_nodeedge_cc
		self.all_terms_cc_loose = terms_cc_loose
		self.all_N_E_T_cc = node_edge_term_cc_loose		
		self.bare_nodeedge_fc = bare_nodeedge_fc_loose
		
	def write_basic_supercell(self,gro,xyz):
		tempgro('20test.gro',self.all_connected_node_edge_cc[self.all_connected_node_edge_cc[:,5]==1])
		tempgro(gro,self.all_N_E_T_cc)
		temp_xyz(xyz,self.all_N_E_T_cc)
		viewgro("20test.gro")
		viewgro(gro)
	
	
	'''
        #Defective model: node/edge missing
            #delete nodes or edges
            #terminate nodes ROUND1
            # find main fragment
                # find unstaturated node uN1
                    # find uN1 neighbors and extract X in neighbor edge(E+int)
                        # filt exposed X sites in uN1
                            # add terminations

            #terminate edge ROUND2
                #find unsaturaed edge uE1
                    #find uE1 neighbors and extract X in neighbor node 'int'
                        # filt exposed X sites in uE1
                            # add terminations (-OH)

        #Defective model: linker exchange
        #   termination OO don't change, use X to set a range 
        #   atoms in outX_range stay
        #    then superimpose by X


            #superimpose for replacement
                #find X AND super impose
    '''
	def defect_missing(self,remove_node_list = [],remove_edge_list = []):
		bare_nodeedge_fc_loose = self.bare_nodeedge_fc
		linker_topics = self.linker_topics
		node_topics = self.node_topics
		sc_unit_cell =self.sc_unit_cell

		renode1_fcarr=reindex_frag_array(bare_nodeedge_fc_loose,'NODE')
		reedge1_fcarr=reindex_frag_array(bare_nodeedge_fc_loose,'EDGE')
		defective_node_fcarr = np.vstack(([i for i in renode1_fcarr if i[5] not in remove_node_list]))
		defective_edge_fcarr = np.vstack(([i for i in reedge1_fcarr if i[5] not in remove_edge_list]))
		renode_fcarr = reindex_frag_array(defective_node_fcarr,'NODE')
		reedge_fcarr = reindex_frag_array(defective_edge_fcarr,'EDGE')
		edgefc_centers = get_frag_centers_fc(reedge_fcarr)
		nodefc_centers = get_frag_centers_fc(renode_fcarr)

		eG = calculate_eG_net_ditopic(edgefc_centers,nodefc_centers,linker_topics)
		eG_subparts=[len(c) for c in sorted(nx.connected_components(eG), key=len, reverse=True)]

		if len(eG_subparts)>1:
			print(f'this MOF has {len(eG_subparts)} seperated fragments: {eG_subparts}')
		else:
			print(f'this MOF has {len(eG_subparts)} fragment')

		unsaturated_nodes = [(n,d) for n, d in eG.degree() if d <node_topics and isinstance(n,int)]
		unsaturated_edges = [(n,d) for n, d in eG.degree() if d <linker_topics and isinstance(n,str)]
		if len(unsaturated_edges) > 0 :
			print(f"UNsaturated edges(linkers) exist, need linker_termination <= {len(unsaturated_edges)}")
		else:
			print("only saturated edges(linkers) exist")

		if len(unsaturated_nodes) > 0 :
			print(f"UNsaturated nodes exist, <={len(unsaturated_nodes)} nodes need node_termination")
		else:
			print("only saturated nodes exist")

		frags=[(len(c),c) for c in sorted(nx.connected_components(eG), key=len, reverse=True)]
		main_frag=list(sorted(nx.connected_components(eG), key=len, reverse=True)[0])
		main_frag_nodes = [i for i in main_frag if isinstance(i,int)]
		main_frag_edges = [i for i in main_frag if re.sub('[0-9]','',str(i)) == 'E']
		
		unsaturated_main_frag_nodes = [i for i in unsaturated_nodes if i[0] in main_frag_nodes]
		unsaturated_main_frag_edges = [i for i in unsaturated_edges if i[0] in main_frag_edges]

		main_frag_half_edges_fc = np.vstack(([reedge_fcarr[reedge_fcarr[:,5]==int(ei[1:])]for ei in main_frag_edges]))
		main_frag_nodes_fc = np.vstack(([renode_fcarr[renode_fcarr[:,5]==ni]for ni in main_frag_nodes]))
		main_frag_edges_fc,xoo_dict,con_nodes_x_dict = addxoo2edge(eG,main_frag_nodes,main_frag_nodes_fc,main_frag_edges,main_frag_half_edges_fc,sc_unit_cell)

		main_frag_nodes_cc = np.hstack((main_frag_nodes_fc[:,:-3],np.dot(main_frag_nodes_fc[:,-3:],sc_unit_cell)))
		main_frag_edges_cc = np.hstack((main_frag_edges_fc[:,:-3],np.dot(main_frag_edges_fc[:,-3:],sc_unit_cell)))
		self.eG = eG
		self.main_frag_nodes = main_frag_nodes
		self.main_frag_edges = main_frag_edges
		self.unsaturated_main_frag_nodes = unsaturated_main_frag_nodes
		self.unsaturated_main_frag_edges = unsaturated_main_frag_edges
		self.node_xoo_dict = xoo_dict
		self.con_nodes_x_dict = con_nodes_x_dict
		self.main_frag_nodes_cc = main_frag_nodes_cc
		self.main_frag_edges_cc = main_frag_edges_cc   

	def term_defective_model(self,n_term_file = 'data/methyl.pdb',e_termfile = 'data/CCO2.pdb'):
		eG = self.eG
		unsaturated_main_frag_nodes = self.unsaturated_main_frag_nodes
		main_frag_nodes = self.main_frag_nodes
		main_frag_nodes_cc =self.main_frag_nodes_cc
		con_nodes_x_dict =self.con_nodes_x_dict
		xoo_dict = self.node_xoo_dict
		unsaturated_main_frag_edges = self.unsaturated_main_frag_edges
		main_frag_edges_cc = self.main_frag_edges_cc
		linker_topics = self.linker_topics
		
		# get indices for cleaved node atoms in main_frag_nodes  without xoo

		xoo_ind = []
		for key in list(xoo_dict):
			xoo_ind.append(key)
			xoo_ind += xoo_dict[key]
		single_node = main_frag_nodes_cc[main_frag_nodes_cc[:,5]==main_frag_nodes_cc[0,5]]
		node_nums = len(single_node)
		single_node_stay_ind=np.asarray([i for i in range(node_nums) if i not in xoo_ind])
		a =[]
		for node in range(len(main_frag_nodes)):
			a.append(node_nums*node+single_node_stay_ind)
		metal_node_indices = np.hstack(a)

		#add -methyl to terminate nodes find exposed xoo 
		ex_node_cxo_cc_loose=exposed_xoo_cc(eG,unsaturated_main_frag_nodes,main_frag_nodes_cc,con_nodes_x_dict,xoo_dict)
		n_terms_loose = add_node_terminations(n_term_file,ex_node_cxo_cc_loose)
		n_terms_cc = np.vstack((n_terms_loose))
		self.n_terms_cc = n_terms_cc

		#add -COOH term to exposed edge and change edge name to HEDGE
		if os.path.basename(e_termfile)=='CCO2.pdb':
			cleaved_metal_node = main_frag_nodes_cc[metal_node_indices]
			t_edges = terminate_unsaturated_edges_CCO2(e_termfile,unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics)
			node_edge_term_cc= np.vstack((cleaved_metal_node,t_edges,n_terms_cc))
		else:
			cleaved_metal_node = main_frag_nodes_cc[metal_node_indices]
			t_edges = terminate_unsaturated_edges(e_termfile,unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics)
			if len(t_edges) > 0:
				node_edge_term_cc= np.vstack((cleaved_metal_node,t_edges,n_terms_cc))
			else:
				node_edge_term_cc = np.empty((0,6))
		self.t_edges = t_edges
		self.tn_te_cc = node_edge_term_cc
			


	def write_tntemof(self,gro):
		tempgro(gro,self.tn_te_cc)
		#temp_xyz("303term_supercell.xyz",self.tn_te_cc)
		viewgro(gro)


	def defect_replace_linker(self,sub_file,sub_class,candidate_res_idx_list,sub_res_newname = 'SUB'):
		node_edge_term_cc = self.tn_te_cc
		for res_idx in candidate_res_idx_list:
			fetch_res_mask,other_mask = fetch_by_idx_resname(node_edge_term_cc,res_idx,sub_class)
			other_res = node_edge_term_cc[other_mask]
			selected_res = node_edge_term_cc[fetch_res_mask]
			if len(selected_res)==0:
				continue
			X_atoms_ind = [i for i in range(len(selected_res)) if selected_res[i,2][0] == 'X']
			outer_atoms_ind = [j for j in range(len(selected_res)) if (j > X_atoms_ind[0]) & (j not in X_atoms_ind)]
			stay_outer_atoms = selected_res[outer_atoms_ind]
			X_atoms = selected_res[X_atoms_ind]
			sub_data=sub_pdb(sub_file)
			subX,subX_ind= Xpdb(sub_data,'X')
			subX_coords_cc = subX[:,-3:]
			subX_coords_cc = subX_coords_cc.astype('float')
			X_atoms_coords_cc = X_atoms[:,-3:]
			X_atoms_coords_cc =  X_atoms_coords_cc.astype('float')

			_,rot,trans = superimpose(subX_coords_cc,X_atoms_coords_cc)
			sub_coords = sub_data[:,-3:]
			sub_coords = sub_coords.astype('float')
			placed_sub_data = np.hstack((sub_data[:,:-3],np.dot(sub_coords,rot)+trans))

			sub_edge = np.vstack((stay_outer_atoms,placed_sub_data))
			sub_edge[:,5]= stay_outer_atoms[0,5]
			sub_edge[:,4] = sub_res_newname
			for row in range(len(sub_edge)):
				sub_edge[row,2] = re.sub('[0-9]','',sub_edge[row,2])+str(row+1)
			node_edge_term_cc = np.vstack((other_res,sub_edge))
		self.replaced_tn_te_cc = node_edge_term_cc

	def write_view_replaced(self,gro):
		tempgro(gro,self.replaced_tn_te_cc)
		viewgro(gro)
			
			