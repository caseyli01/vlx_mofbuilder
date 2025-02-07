import numpy as np

def fetch_by_idx_resname(arr,idx,resname):
    mask1 = (arr[:,5]==int(idx))
    mask2 = (arr[:,4] == resname)
    fetch_res_mask = mask1&mask2
    other_mask = ~fetch_res_mask
    return fetch_res_mask,other_mask

def sub_pdb(filename):
        inputfile = str(filename)
        with open(inputfile, "r") as fp:
            content = fp.readlines()
            #linesnumber = len(content)
        data = []
        for line in content:
            line = line.strip()
            if len(line)>0: #skip blank line
                if line[0:6] == "ATOM" or line[0:6] == "HETATM":
                    value_atom = line[12:16].strip()  # atom_label
                    #resname
                    #value2 = 'MOL'  # res_name

                    value_x = float(line[30:38])  # x
                    value_y = float(line[38:46])  # y
                    value_z = float(line[46:54])  # z
                    value_charge = float(line[61:66]) 
                    value_note = line[67:80].strip() # atom_note
                    #resnumber
                    try:
                        value_res_num = int(line[22:26])
                    except ValueError:
                        value_res_num = 1 
                    data.append([value_atom,value_charge,value_note,value_res_num,'SUB',value_res_num,value_x,value_y,value_z])
        return np.vstack(data)

def Xpdb(data,X): 
        indices=[i for i in range(len(data)) if data[i,2][0] == X]
        X_term=data[indices]
        return X_term,indices