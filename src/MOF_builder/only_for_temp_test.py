import numpy as np
import networkx as nx
import re
import os
from _implement_edges import addxoo2edge
from _cluster import cluster_supercell,placed_arr
from _output import tempgro,temp_xyz,viewgro#,viewxyz
from _terminations import terminate_nodes,terminate_unsaturated_edges,add_node_terminations,exposed_xoo_cc,Xpdb,terminate_unsaturated_edges_CCO2
from _isolated_node_cleaner import reindex_frag_array,get_frag_centers_fc,calculate_eG_net_ditopic
from _replace import fetch_by_idx_resname,sub_pdb
from _supercell import Carte_points_generator
from _superimpose import superimpose

####### Global options #######


pi = np.pi

class MOF_ditopic:
    #def __init__(self,template,node,edge,node_topics):
    #    self.template = template
    #    self.node = node
    #    self.edge = edge
    #    self.linker_topics = 2 #ditopic class 
    #    self.node_topics = node_topics
	
    def __init__(self):
        self.linker_topics = 2
        pass
    def load(self,TG,placed_all,sc_unit_cell,placed_nodes,placed_edges):
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
    
        #new_beginning_fc = find_new_node_beginning(frame_node_fc)		
        new_beginning_fc = np.array([0,0,0])
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

        #eG = calculate_eG_net_ditopic(edgefc_centers,nodefc_centers,linker_topics)
        eG = calculate_eG_net_ditopic(edgefc_centers,nodefc_centers,linker_topics)
        eG_subparts=[len(c) for c in sorted(nx.connected_components(eG), key=len, reverse=True)]
        #for pillar stack, add virtual edge to connect clostest nodes, set distance to 0.6
        if len(eG_subparts)>1:
            print(f'this MOF has {len(eG_subparts)} seperated fragments: {eG_subparts}')
        else:
            print(f'this MOF has {len(eG_subparts)} fragment')



        frags=[(len(c),c) for c in sorted(nx.connected_components(eG), key=len, reverse=True)]
        main_frag=list(sorted(nx.connected_components(eG), key=len, reverse=True)[0]) 
       

        main_frag_nodes = [i for i in main_frag if isinstance(i,int)]
        main_frag_edges = [i for i in main_frag if re.sub('[0-9]','',str(i)) == 'E']



        ###delete "virtual" edges #next version TODO:
        ##for edge_n in eG.edges():
        ##    if eG.edges[edge_n]['type'] == 'virtual':
        ##        eG.remove_edge(edge_n[0],edge_n[1])

        unsaturated_nodes = [(n,d) for n, d in eG.degree() if d <node_topics and isinstance(n,int)]
        unsaturated_edges = [(n,d) for n, d in eG.degree() if d <linker_topics and isinstance(n,str)]
        


        if len(unsaturated_edges) > 0 :
            print(f"UNsaturated edges(linkers) exist, need linker_termination <= {len(unsaturated_edges)}")
        else:
            print("only saturated edges(linkers) exist")

        if len(unsaturated_nodes) > 0 :
            print(f"UNsaturated nodes exist, <={len(unsaturated_nodes)} nodes need node_termination")
            print(unsaturated_nodes)
        else:
            print("only saturated nodes exist")



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
        if len(n_terms_loose) > 0:
            n_terms_cc = np.vstack((n_terms_loose))
            self.n_terms_cc = n_terms_cc
        else:
            n_terms_cc = np.empty((0,9))
            self.n_terms_cc = n_terms_cc

        #add -COOH term to exposed edge and change edge name to HEDGE
        if os.path.basename(e_termfile)=='CCO2.pdb':
            cleaved_metal_node = main_frag_nodes_cc[metal_node_indices]
            t_edges = terminate_unsaturated_edges_CCO2(e_termfile,unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics)
            node_edge_term_cc= np.vstack((cleaved_metal_node,t_edges,n_terms_cc))
        else:
            cleaved_metal_node = main_frag_nodes_cc[metal_node_indices]
            t_edges = terminate_unsaturated_edges(e_termfile,unsaturated_main_frag_edges,eG,main_frag_edges_cc,linker_topics)
            node_edge_term_cc= np.vstack((cleaved_metal_node,t_edges,n_terms_cc))
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
        