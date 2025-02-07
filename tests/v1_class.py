import os
import numpy as np
import veloxchem as vlx
from MOF_builder.functions.frag_recognizer import process_linker_molecule
from MOF_builder.functions.add_dummy2node import add_dummy_atoms_nodecif
from MOF_builder.gmxmd_prepare.fetchfile import fetch_file,read_mof_top_dict,find_data_folder,copy_file
from MOF_builder.multitopic import multitopic
from MOF_builder.functions.process_gro import split_dummy_node_tntte_arr
from MOF_builder.gmxmd_prepare.forcefield import get_residues_forcefield
from MOF_builder.gmxmd_prepare.gro_itps import get_gro,get_itps
from MOF_builder.gmxmd_prepare.top_combine import genrate_top_file
from MOF_builder.gmxmd_prepare.mdps import copy_mdps
import subprocess
from MOF_builder.gmxmd_prepare.solvate import solvate_model


class MOF_builder_v1_demo:
    def __init__(self):
        pass

    def set_MOF_family(self,MOF_family: str):
        self.MOF_family = MOF_family

    def set_MOF_node(self, MOF_node: str):
        self.MOF_node = MOF_node

    def set_MOF_node_metal(self,MOF_node_metal: str):
        self.MOF_node_metal = MOF_node_metal

    def set_MOF_supercell(self, supercell: list[int]):
        if isinstance(supercell, list) and len(supercell) == 3 and all(isinstance(i, int) for i in supercell):
            self.supercell = np.asarray(supercell)
        else:
            raise ValueError("supercell must be a list of 3 integers")
        
    def set_linker_file(self,linker_file: str):
        self.linker_file = linker_file

    def set_DUMMY_NODE(self,DUMMY_NODE: bool):
        self.DUMMY_NODE = DUMMY_NODE

    def set_node_termination(self,node_termination: str):
        self.node_termination = node_termination

    def set_sol_list(self,sol_list: list[str]):
        self.sol_list = sol_list

    def set_sol_num(self,sol_num: list[int]):
        self.sol_num = sol_num

    def set_linker_file_ff(self,linker_file_ff: str):
        self.linker_file_ff = linker_file_ff

    def set_model_name(self,model_name: str):
        self.model_name = model_name

    def set_templates_dif(self,templates_dir: str):
        self.templates_dir = templates_dir
    def set_nodes_dir(self,nodes_dir: str):
        self.nodes_dir = nodes_dir
    def set_edges_dir(self,edges_dir: str):
        self.edges_dir = edges_dir
    def set_save_cif(self,save_cif: bool):
        self.save_cif = save_cif

    def build_mof(self):
        mof_family = self.MOF_family
        mof_node_metal = self.MOF_node_metal
        model_name = self.model_name
        mof_linker_file = self.linker_file
        mof_DUMMY_NODE = self.DUMMY_NODE
        self.data_path =find_data_folder()
        mof_top_dict = read_mof_top_dict()
        node_connection= mof_top_dict[mof_family]['node_connection']
        linker_topic = mof_top_dict[mof_family]['linker_topic']
        self.linker_topic = linker_topic
        template = mof_top_dict[self.MOF_family]['topology']+'.cif'
        #if did not set template_dir then set it to default
        if not hasattr(self,'templates_dir'):
            self.templates_dir = '../data/template_database'

        if not hasattr(self,'edges_dir'):
            self.edges_dir = 'edges'
        if not hasattr(self,'nodes_dir'):
            self.nodes_dir = 'nodes'

        templates_dir = self.templates_dir
        nodes_dir = self.nodes_dir
        edges_dir = self.edges_dir

        data_path = find_data_folder()
        nodes_database_path = os.path.join(data_path,"nodes_database")
        node_cif= fetch_file(nodes_database_path,[str(node_connection)+'c',mof_node_metal], ['dummy'])
        for i in range(len(node_cif)):
            node_cif_database = os.path.join(data_path,'nodes_database/'+node_cif[i])
            target_node_path = os.path.join('nodes',node_cif[i])
            copy_file(node_cif_database,target_node_path)
        if mof_DUMMY_NODE:
            add_dummy_atoms_nodecif(target_node_path,mof_node_metal)
        if mof_DUMMY_NODE:
            os.remove(target_node_path)
        molecule = vlx.Molecule.read_xyz_file(mof_linker_file)
        self._linker_center_frag_nodes_num,self._linker_center_Xs,self._linker_single_frag_nodes_num,self._linker_frag_Xs = process_linker_molecule(molecule,linker_topic)

        if mof_DUMMY_NODE:
            keywords = [str(node_connection)+'c',mof_node_metal,'dummy']
            nokeywords = []
        else:
            keywords = [str(node_connection)+'c',mof_node_metal]
            nokeywords = ['dummy']
        self.dummy_node_name = fetch_file(nodes_dir,keywords, nokeywords)[0]

        new_mof = multitopic(templates_dir,nodes_dir,edges_dir,template,node_connection,linker_topic)
        
        if not hasattr(self,'save_cif'):
            self.save_cif = False
        new_mof.load(self.save_cif)

        self.new_mof = new_mof

    def set_boundary_cut_buffer(self,boundary_cut_buffer: float):
        self.boundary_cut_buffer = boundary_cut_buffer
    def set_edge_center_check_buffer(self,edge_center_check_buffer: float):
        self.edge_center_check_buffer = edge_center_check_buffer
    
    def build_terminated_primitimive_cell(self):
        if not hasattr(self,'node_termination'):
            self.node_termination = 'methyl'
            print('node_termination is set to default value: methyl')
        if self.node_termination == 'methyl' :
            self.n_term_file = '../data/terminations_database/methyl.pdb'
        if not hasattr(self,'boundary_cut_buffer'):
            self.boundary_cut_buffer = 0.0
        if not hasattr (self,'edge_center_check_buffer'):
            self.edge_center_check_buffer = 0.2
        self.new_mof.basic_supercell(self.supercell,term_file = self.n_term_file,
                                boundary_cut_buffer=self.boundary_cut_buffer,
                                edge_center_check_buffer=self.edge_center_check_buffer)
        
    def write_terminated_primitive_cell_file(self, file_name: str):
        self.new_mof.write_basic_supercell(file_name+'.gro',file_name+'.xyz')
    
    def terminate_mof(self):
        self.new_mof.defect_missing([],[])
        self.new_mof.term_defective_model(self.n_term_file,e_termfile = '../data/terminations_database/CCO2.pdb')

    def write_terminated_mof_gro(self, file_name: str):
        self.new_mof.write_tntemof(file_name+'.gro')
    
    def fetch_unsaturated_linkers_list(self):
        u_edge_idx=[]
        for i in self.new_mof.unsaturated_main_frag_edges:
            u_edge_idx.append(int(i[0][1:]))
        self.unsaturated_linkers_list = u_edge_idx
        return u_edge_idx

    def set_removed_linkers_list(self, removed_linkers_list: list[int]):
        self.removed_linkers_list = removed_linkers_list
    def set_removed_nodes_list(self, removed_nodes_list: list[int]):
        self.removed_nodes_list = removed_nodes_list

    def create_defective_mof(self,file_name: str): 
        if not hasattr(self,'removed_nodes_list'):
            self.removed_nodes_list = []
        if not hasattr(self,'removed_linkers_list'):
            self.removed_linkers_list = []
        self.new_mof.defect_missing(remove_node_list=self.removed_nodes_list,remove_edge_list=self.removed_linkers_list)
        self.new_mof.term_defective_model(self.n_term_file,e_termfile = '../data/terminations_database/CCO2.pdb')
        self.new_mof.write_tntemof(file_name+'.gro')
    

    def replace_linker(self,sub_pdb_file: str, sub_class: str, candidate_res_idx_list: list[int], file_name: str):
        self.new_mof.defect_replace_linker(sub_pdb_file,sub_class,candidate_res_idx_list)
        self.new_mof.write_view_replaced(file_name+'.gro')
    
    def md_prepare(self):
        if not hasattr(self,'linker_file_ff'):
            self.linker_file_ff = None
        arr,node_split_dict = split_dummy_node_tntte_arr(self.new_mof.tn_te_cc,self.dummy_node_name,self.DUMMY_NODE)
        new_arr,self.res_info,self.restypes=get_residues_forcefield(arr,node_split_dict,self.DUMMY_NODE,self.linker_file,self.linker_file_ff,
                                                          self.linker_topic,self._linker_center_frag_nodes_num,self._linker_center_Xs,
                                                          self._linker_single_frag_nodes_num,self._linker_frag_Xs)
        self.gro_path=get_gro(self.model_name,new_arr)
        self.itp_dir=get_itps(self.data_path,self.restypes,self.MOF_node_metal,self.node_termination,self.sol_list)
        self.top_path=genrate_top_file(self.itp_dir,self.data_path,self.res_info,self.model_name)
        self.mdp_dir=copy_mdps(self.data_path)
    
    def gmx_solvate(self,EXECUTE=False):
        init_gro=self.gro_path
        for i in range(len(self.sol_list)):
            EXECUTE=False
            sv=solvate_model(self.data_path)
            sv.set_maxsol(int(self.sol_num[i]))
            sv.set_solvent(self.sol_list[i])
            sv.set_grofile(init_gro)
            sv.set_top_file(self.top_path)
            command=sv.solvate()
            if EXECUTE:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                # Display the result
                #print("STDOUT:", result.stdout)
                #print("STDERR:", result.stderr)
            init_gro=sv.output_gro
            sv= None
            
    '''combine itps for topolgy file'''




