import os
import glob
from fetchfile import copy_file
from _output import tempgro
from atoms2C import atoms2c


def get_gro(model_name,new_arr):
    #tempgro('2testdummy.gro',new_arr)
    #copy_file('2testdummy.gro','MD_run/TEST.gro')
    os.makedirs('MD_run/',exist_ok=True)
    groname = 'MD_run/'+model_name+'.gro'
    tempgro(groname,new_arr)
    new_gro_path = atoms2c(groname,3,3,3,10,10,10)
    
    return  new_gro_path

def get_itps(data_path,restypes,metal,node_termination,sol_list):

  #itps nodes_database,edges,sol,gas
    itp_path='MD_run/itps'
    nodesitp_path = os.path.join(data_path,'nodes_itps')
    os.makedirs(itp_path,exist_ok=True)
    # copy nodes itps
    for i in glob.glob(os.path.join(data_path,'nodes_itps/*itp')):
        itp_name = os.path.basename(i)
        if itp_name.removesuffix('.itp') in restypes:
            copy_file(i,os.path.join(itp_path,itp_name))
        if itp_name.removesuffix('.itp') == metal:
            copy_file(i,os.path.join(itp_path,itp_name))
    # copy EDGE(/TERM) itps
    for j in glob.glob(os.path.join('Residues','*itp')):
        itp_name = os.path.basename(j)
        if itp_name.removesuffix('.itp') in restypes:
            copy_file(j,os.path.join(itp_path,itp_name))
    # copy TERM itp
    for k in glob.glob(os.path.join(data_path,'terminations_itps/*itp')):
        itp_name = os.path.basename(i)
        if itp_name.removesuffix('.itp')  == node_termination:
            copy_file(i,os.path.join(itp_path,'TERM.itp'))

    #copy solvent,ions,gas itps
    for sol in sol_list:
        copy_file(os.path.join(data_path,'solvent_database/'+sol+'.itp'),os.path.join(itp_path,sol+'.itp'))
    return itp_path