import time 
from v2_builder import net_optimizer

class mof_builder():
    def __init__(self, mof_family):
        self.mof_family = mof_family
        #seach mof_family name in the config file # if not found, raise error and ask user to generate the template cif and add it to the config file
        #from config get linker topic, node metal type, dummy node True or False, template cif, node cif
        #in database by calling MOF family name and node metal type, dummy node True or False
        self.template_cif = 'fcu.cif'
        self.node_cif = 'node2.cif'
        self.node_target_type = 'Zr'
    
    def set_supercell(self, supercell):
        self.supercell = supercell

    def set_linker_xyz(self, linker_xyz):
        self.linker_xyz = linker_xyz
        self.linker_cif = 'diedge.cif'
    
    def set_rotation_optimizer_maxfun(self, maxfun):
        self.rotation_optimizer_maxfun = maxfun
    
    def set_rotation_optimizer_maxiter(self, maxiter):
        self.rotation_optimizer_maxiter = maxiter
    
    def set_rotation_optimizer_tol(self, tol):
        self.rotation_optimizer_tol = tol
    
    def set_rotation_optimizer_method(self, method):
        self.rotation_optimizer_method = method
    
    def set_node_topic(self, node_topic):
        self.node_topic = node_topic
    
    def set_linker_topic(self, linker_topic):
        self.linker_topic = linker_topic
    
    def set_connection_constant_length(self, connection_constant_length):
        self.connection_constant_length = connection_constant_length
    
    def set_node_termination(self, node_termination):
        self.node_termination = node_termination
    
    def set_vitual_edge_search(self, vitual_edge_search):
        self.vitual_edge = True
    
    def set_dummy_node(self, dummy_node):
        self.dummy_node = dummy_node
    
    def set_remove_node_list(self, remove_node_list):
        self.remove_node_list = remove_node_list
    
    def set_remove_edge_list(self, remove_edge_list):
        self.remove_edge_list = remove_edge_list
    



    
    def build(self):
        #check before building
        if not hasattr(self, 'linker_xyz'):
            print('linker_xyz is not set')
            raise ValueError('linker_xyz is not set')
        if not hasattr(self, 'supercell'):
            self.supercell = [1,1,1]
        if not hasattr(self, 'template_cif'):
            print('template_cif is not set')
            raise ValueError('template_cif is not set')
        if not hasattr(self, 'node_cif'):
            print('node_cif is not set')
            raise ValueError('node_cif is not set')
        if not hasattr(self, 'node_target_type'):
            print('node_target_type is not set')
            raise ValueError('node_target_type is not set')
        if not hasattr(self, 'linker_cif'):
            print('linker_cif is not set')
            raise ValueError('linker_cif is not set')
        if not hasattr(self, 'linker_topic'):
            print('linker_topic is not set')
            raise ValueError('linker_topic is not set')
        if not hasattr(self, 'dummy_node'):
            self.dummy_node = False
        if not hasattr(self, 'mof_family'):
            print('mof_family is not set, the user defined topology will be used: ',self.template_cif)

        if self.linker_topic == 2:
            print('ditopic mof builder driver is called')
            start_time = time.time()
            linker_cif = self.linker_cif
            template_cif = self.template_cif
            node_cif = self.node_cif
            node_target_type = self.node_target_type
            supercell = self.supercell
            net = net_optimizer()
            net.analyze_template_ditopic(template_cif)
            net.node_info(node_cif,node_target_type)
            net.linker_info(linker_cif)
            net.optimize()
            print("--- %s seconds ---" % (time.time() - start_time))
            net.set_supercell(supercell)
            net.place_edge_in_net()
            net.make_supercell_ditopic()
            net.make_eG_from_supereG_ditopic()
            net.main_frag_eG()
            net.make_supercell_range_cleaved_eG()
            net.add_xoo_to_edge_ditopic()
            net.find_unsaturated_node_eG()
            net.set_node_terminamtion('methyl.pdb')
            net.add_terminations_to_unsaturated_node()
            net.remove_xoo_from_node()
            self.net = net


    def set_gro_name(self, gro_name):
        self.gro_name = gro_name

    def write_gro(self):
        if not hasattr(self, 'gro_name'):
            self.gro_name = 'mof_'+str(self.mof_family.split('.')[0])+'_'+str(self.linker_xyz.split('.')[0])
            print('gro_name is not set, will be saved as: ',self.gro_name,'.gro')

        self.net.write_node_edge_node_gro(self.gro_name)
            #temp_save_eGterm_gro(net.eG,net.sc_unit_cell) #debugging
    
    def make_defects_missing (self):
        defective_net = self.net
        if hasattr(self, 'remove_node_list'):
            remove_node_list = self.remove_node_list
        if hasattr(self, 'remove_edge_list'):
            remove_edge_list = self.remove_edge_list
        
        defective_net.remove_node_edge(remove_node_list, remove_edge_list) #TODO:
        defective_net.make_eG_from_supereG_ditopic()
        defective_net.main_frag_eG()
        defective_net.make_supercell_range_cleaved_eG()
        defective_net.add_xoo_to_edge_ditopic()
        defective_net.find_unsaturated_node_eG()
        defective_net.set_node_terminamtion(self.node_termination)
        defective_net.add_terminations_to_unsaturated_node()
        defective_net.remove_xoo_from_node()

        self.defective_net = defective_net
    
    def make_defects_exchange(self):
        defective_net = self.net
        if hasattr(self, 'exchange_node_list'):
            exchange_node_list = self.exchange_node_list
        if hasattr(self, 'exchange_edge_list'):
            exchange_edge_list = self.exchange_edge_list
        if hasattr(self, 'to_exchange_node_pdb'):
            to_exchange_node_pdb = self.to_exchange_node_pdb
        if hasattr(self, 'to_exchange_edge_pdb'):
            to_exchange_edge_pdb = self.to_exchange_edge_pdb
        #TODO:
        defective_net.exchange_node_edge(exchange_node_list, exchange_edge_list, to_exchange_node_pdb, to_exchange_edge_pdb)
        self.defective_net = defective_net
    
    def set_defect_gro_name(self, defect_gro_name):
        self.defect_gro_name = defect_gro_name

    def write_defect_gro(self): 
        if not hasattr(self, 'defective_net'):
            print('defective_net is not set')
            print('make_defects_missing() or make_defects_exchange() should be called before write_defect_gro(), or you can write with write_gro()')
            return

        if not hasattr(self, 'defect_gro_name'):
            self.defect_gro_name = 'defective_mof_'+str(self.mof_family.split('.')[0])+'_'+str(self.linker_xyz.split('.')[0])
            print('defect_gro_name is not set, will be saved as: ',self.defect_gro_name,'.gro')

        self.defective_net.write_node_edge_node_gro(self.defect_gro_name)
        
    


    