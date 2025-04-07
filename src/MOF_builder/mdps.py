import os
import glob
from fetchfile import copy_file
def copy_mdps(data_path):
    mdp_path = 'MD_run/mdp'
    os.makedirs(mdp_path,exist_ok=True)
    for i in glob.glob(os.path.join(data_path+'/mdp/','*')):       
        copy_file(i,'MD_run/mdp/'+os.path.basename(i))
    return mdp_path