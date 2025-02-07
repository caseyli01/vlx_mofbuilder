import veloxchem as vlx
import os
import numpy as np
import re
import shutil
from MOF_builder.functions.map_forcefield import map_forcefield_by_xyz,parseff
from MOF_builder.functions.write_mapping_between import write_mapping_file,get_mapping_between_nometal_linker_xyz
from MOF_builder.functions.output import temp_xyz
from fetchfile import fetch_res_in_node_num

def ff_gen_xyz(f,charge=0,residue_name='MOL',show=True, scf_basis='sto-3g',scf_conv = 1e-2,scf_xcfun = 'b3lyp',scf_maxiter = 30,resp = False):
    m1 = vlx.Molecule.read_xyz_file(f)
    gromacs_file=os.path.basename(f).removesuffix('.xyz')
    res_name=residue_name
    if show:
        m1.show(atom_indices= True)
    m1.set_charge(charge)
    basis = vlx.MolecularBasis.read(m1, scf_basis)
    scf_drv = vlx.ScfUnrestrictedDriver() #TODO:
    #scf_drv.guess_unpaired_electrons = '1(1)' #TODO:
    scf_drv.conv_thresh = scf_conv
    scf_drv.max_iter = scf_maxiter
    scf_drv.xcfun = scf_xcfun
    scf_results = scf_drv.compute(m1, basis)
    scf_drv.ostream.mute()
    ff_gen = vlx.ForceFieldGenerator()
    ff_gen.ostream.mute()
    ff_gen.create_topology(m1, basis,scf_results,resp)
    ff_gen.write_gromacs_files(os.path.abspath("") + "/Residues/"+gromacs_file, res_name)
    return os.path.abspath("") + "/Residues/"+gromacs_file+'.itp'

# xtb optimize residues for forcefield generator
def xtb_residue(xyz_file,charge):
    molecule = vlx.Molecule.read_xyz_file(xyz_file)
    molecule.set_charge(charge)
    
    scf_drv=None 
    scf_drv = vlx.XtbDriver()
    scf_drv.ostream.mute()

    opt_drv = vlx.OptimizationDriver(scf_drv)
    opt_drv.ostream.mute()
    opt_results = opt_drv.compute(molecule)

    opt_geo = opt_results["final_geometry"]
    final_geometry = vlx.Molecule.read_xyz_string(opt_geo)
    final_geometry.show(atom_indices= True)
    print('Energy of optimized structure: ' + str(opt_results["opt_energies"][-1]))
    opt_geo = opt_results["final_geometry"]
    #print('XYZ coordinates of optimized structure:')
    #print(opt_results["final_geometry"])
    with open(xyz_file,'w') as f:
        f.write(opt_results["final_geometry"])
    return opt_geo

#1.ff_gen to generate ff for user_linker
#2. parse ff
#3. mapping and replace new mof_edge ff


###ff_name = ff_gen_xyz(linker_file,charge=-3)
##ff_name='/Users/chenxili/GitHub/MOFbuilder/tests/Residues/triph.itp'
##mapping = get_mapping_between_nometal_linker_xyz(linker_topic,center_frag_nodes_num,center_Xs,single_frag_nodes_num,frag_Xs,linker_file, new_xyz='Residues/EDGE.xyz')
##map_name = 'linker_ff_mapping'
##parsed_path=parseff(ff_name)
##map_path = write_mapping_file(parsed_path,mapping,map_name)
##map_forcefield_by_xyz(parsed_path,map_path,linker_file, new_xyz='EDGE.xyz')



def get_residues_forcefield(arr,node_split_dict,DUMMY_NODE,linker_file,linker_file_ff,linker_topic,center_frag_nodes_num,center_Xs,single_frag_nodes_num,frag_Xs):
    new_arr_list = []
    restypes = [name for name in np.unique(arr[:,4])]

    path = os.path.abspath("") + "/Residues/"
    os.makedirs(path, exist_ok=True)

    res_info={}

    res_previous_sum = 0
    for res in restypes:
        type_arr = arr[arr[:,4]==res]
        sample_idx = type_arr[-1,5]
        sample_arr = type_arr[type_arr[:,5]==sample_idx]
        res_numbers = len(sample_arr)

        if res in ['METAL','HHO','HO','O']:
            res_numbers = fetch_res_in_node_num(res,node_split_dict,DUMMY_NODE)
            #print(res_numbers,res)
            for i in range(len(type_arr)):
                type_arr[i,5]=i//res_numbers+1+res_previous_sum
                type_arr[i,2]=re.sub('[0-9]','',type_arr[i,2])+str(int(i%res_numbers+1))
                #print(type_arr[i,0:2],re.sub('[0-9]','',type_arr[i,2])+str(int(i%res_numbers+1)))
            sample_arr = type_arr[type_arr[:,5]==1+res_previous_sum]
            res_sum = int(len(type_arr)/res_numbers)
            res_info[res]=res_sum
            res_previous_sum +=res_sum
            #print(res_previous_sum,res_sum,res)
            xyz_file = os.path.join('Residues',str(res)+'.xyz')
            temp_xyz(xyz_file,sample_arr)
            if res  == 'O':
                #xtb_residue(xyz_file,-2)
                #ff_gen_xyz(xyz_file,charge=-2)
                print(res,"  fetched/optimized")
            elif res  ==  'HO':
                #xtb_residue(xyz_file,-1)
                #ff_gen_xyz(xyz_file,charge=-1)
                print(res,"  fetched/optimized")
            elif res == 'HHO':
                #xtb_residue(xyz_file,0)
                #ff_gen_xyz(xyz_file,charge=0)
                print(res,"  fetched/optimized")

            elif res == 'METAL':
                print(res,"  fetched")

            ##reindex resnumber
            new_arr_list.append(type_arr)
            #print('type_arr[i,5]',type_arr[0,3:6])

        else:
            for i in range(len(type_arr)):
                type_arr[i,5]=i//res_numbers+1+res_previous_sum

            res_sum = int(len(type_arr)/res_numbers)
            res_info[res]=res_sum
            res_previous_sum +=res_sum
            #print(res_previous_sum,res_sum,res)
            xyz_file = os.path.join('Residues',str(res)+'.xyz')
            temp_xyz(xyz_file,sample_arr)
            if res in ['EDGE']:
                #xtb_residue(xyz_file,-3)
                if linker_file_ff is None:
                    ff_name = ff_gen_xyz(linker_file,charge=-1*linker_topic)
                #ff_name = ff_gen_xyz(linker_file,charge=-1*linker_topic)
                
                ff_name=linker_file_ff
                mapping,metals,mol_metals = get_mapping_between_nometal_linker_xyz(linker_topic,center_frag_nodes_num,center_Xs,single_frag_nodes_num,frag_Xs,linker_file, new_xyz='Residues/EDGE.xyz')
                map_name = 'linker_ff_mapping'
                parsed_path=parseff(ff_name)
                map_path = write_mapping_file(parsed_path,mapping,metals,mol_metals,map_name)
                map_forcefield_by_xyz(parsed_path,map_path,linker_file, new_xyz='EDGE.xyz')
                shutil.rmtree(parsed_path)
                print(res,"  mapped")
            elif res in ['HEDGE']:
                #xtb_residue(xyz_file,-1*linker_topic+1)
                print(res,"   mapped/optimized")
            elif res in ['HHEDGE']:
                #xtb_residue(xyz_file,-1*linker_topic+2)
                print(res,"   mapped/optimized")
            elif res in ['TERM']:
                #xtb_residue(xyz_file,-1)
                #ff_gen_xyz(xyz_file,charge=-1,residue_name=res[:3])
                print(res,"  fetched/optimized")

            ##reindex resnumber
            new_arr_list.append(type_arr)
            
            #print('type_arr[i,5]',type_arr[0,3:6])
    #fetch_res_mask,_ = fetch_by_idx_resname(arr,1,sub_class)
    new_arr=np.vstack(new_arr_list)
    return new_arr,res_info,restypes