import os
'''solvate model and update topology file''' 
class solvate_model():
    def __init__(self,data_path):
        self.solvent = None
        self.gro_file = None
        self.output_gro = None
        self.top_file = None
        self.maxsol = None
        self.data_path = data_path
     
    
    def set_solvent(self,solvent):
        self.solvent = solvent
        self.full_path_solvent = os.path.join(self.data_path,'solvent_database/'+solvent+'.gro')

    def set_grofile(self,grofile):
        if self.solvent is None:
            raise ValueError('please set solvent before set grofile')
        self.gro_file = grofile
        self.output_gro = grofile.removesuffix('.gro')+'_'+self.solvent+'.gro'

    def set_top_file(self,top_file):
        self.top_file = top_file

    def set_maxsol(self,maxsol):
        self.maxsol = maxsol

    def solvate(self):
        comm_head = "gmx solvate -cp "+str(self.gro_file)
        comm_cs = " -cs "+str(self.full_path_solvent) if self.solvent != 'tip3p' else ''
        comm_o = " -o "+str(self.output_gro)
        comm_p = " -p "+str(self.top_file) if self.solvent is not None else ''
        comm_maxsol = " -maxsol "+str(self.maxsol) if self.maxsol is not None else ''
        comm_line = comm_head+comm_cs+comm_o+comm_p+comm_maxsol
        print(comm_line)
        return [comm_head,comm_cs,comm_o,comm_p,comm_maxsol]
