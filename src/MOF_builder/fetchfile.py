import glob
import os
import shutil

def copy_file(old_path, new_path):
    # Check if the source file exists
    if not os.path.exists(old_path):
        raise FileNotFoundError(f"The source file does not exist: {old_path}")
    
    # Ensure the destination directory exists
    new_dir = os.path.dirname(new_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)  # Create the directory if it doesn't exist

    # Copy the file
    shutil.copy2(old_path, new_path)  # copy2 preserves metadata
    print(f"File copied from {old_path} to {new_path}")


def fetch_ciffile(dir,keywords,nokeywords):
    candidates = []
    for cif in glob.glob(os.path.join(dir, '*.cif'), recursive=True):
        name = os.path.basename(cif)
        if all(i in name for i in keywords) and all(j not in name for j in nokeywords):
            candidates.append(os.path.basename(cif))
    if len(candidates)==0:
        raise ValueError(f"Cannot find a file including '{keywords}' ")
    elif len(candidates)==1:
        print('found the file including', keywords)
        return candidates
    elif len(candidates) >1:
        print('found many files including', keywords)
        return candidates
    
def fetch_pdbfile(dir,keywords,nokeywords):
    candidates = []
    for pdb in glob.glob(os.path.join(dir, '*.pdb'), recursive=True):
        name = os.path.basename(pdb)
        if all(i in name for i in keywords) and all(j not in name for j in nokeywords):
            candidates.append(os.path.basename(pdb))
    if len(candidates)==0:
        raise ValueError(f"Cannot find a file including '{keywords}' ")
    elif len(candidates)==1:
        print('found the file including', keywords)
        return candidates
    elif len(candidates) >1:
        print('found many files including', keywords)
        return candidates
    
    
def find_data_folder():
    i=0
    current_p=os.path.abspath('')
    for i in range(5):
        if os.path.basename(current_p)=='vlx_mofbuilder': #TODO: need to ask Xin about this
            data_path_exist = os.path.exists(os.path.join(current_p,'data'))
            if data_path_exist:
                data_path = os.path.join(current_p,'data')
                return data_path
        parent_p=os.path.dirname(current_p)
        current_p = parent_p
        i+=1
    raise ValueError('cannot locate data folder')

def read_mof_top_dict(data_path):
    if os.path.exists(os.path.join(data_path,'MOF_topology_dict')):
        mof_top_dict_path = os.path.join(data_path,'MOF_topology_dict')
        with open(mof_top_dict_path,'r') as f:
            lines = f.readlines()
        titles = lines[0].split()
        mofs = lines[1:]
    mof_top_dict = {}
    for mof in mofs:
        mof_name = mof.split()[0]
        if mof_name not in mof_top_dict.keys():
            mof_top_dict[mof_name]={
                'node_connectivity':int(mof.split()[1]),'metal':[mof.split()[2]],
                'linker_topic':int(mof.split()[3]),'topology':mof.split()[-1]
                }
        else:
            mof_top_dict[mof_name]['metal'].append(mof.split()[2])
    return mof_top_dict


def fetch_res_in_node_num(res,node_split_dict,DUMMY_NODE):
    if res=='METAL':
        if DUMMY_NODE:
            return node_split_dict['dummy_res_len']
        else:
            return 1
    elif res=='HHO':
        return 3
    elif res=='HO':
        return 2
    elif res=='O':
        return 1
    elif res=='OOC':
        return 3

