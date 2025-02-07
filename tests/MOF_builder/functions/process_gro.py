import numpy as np
import os

def make_dummy_split_node_dict(dummy_node_name):
    node_split_dict = {}
    dict_name = dummy_node_name.removesuffix('.cif')+'_dict'
    opath = os.path.join('nodes/',dict_name)
    with open(opath,'r') as f:
        lines = f.readlines()
    node_res_counts = 0
    for li in lines:
        li= li.strip('\n')
        key = li[:20].strip(' ')
        value = li[-4:].strip(' ')
        node_split_dict[key]=int(value)
    return node_split_dict


def rename_node_arr(dummy_node_name,node_arr,DUMMY_NODE):
    node_split_dict = make_dummy_split_node_dict(dummy_node_name)
    if DUMMY_NODE:
        metal_num = node_split_dict['METAL_count']*node_split_dict['dummy_res_len']
    else:
        metal_num = node_split_dict['METAL_count']
    hho_num = node_split_dict['HHO_count']*3
    ho_num = node_split_dict['HO_count']*2
    o_num = node_split_dict['O_count']*1
    metal_range = metal_num
    hho_range = metal_num+hho_num
    ho_range = metal_num+hho_num+ho_num
    o_range = metal_num+hho_num+ho_num+o_num
    new_node_arr_list=[]
    for idx in set(node_arr[:,5]):
        idx_arr = node_arr[node_arr[:,5]==idx]
        idx_arr[0:metal_range,4] = 'METAL'
        idx_arr[metal_range:hho_range,4] = 'HHO'
        idx_arr[hho_range:ho_range,4] = 'HO'
        idx_arr[ho_range:o_range,4] = 'O'
        new_node_arr_list.append(idx_arr)
    new_node_arr=np.vstack(new_node_arr_list)
    return new_node_arr

def split_dummy_node_tntte_arr(tnte_arr,dummy_node_name,DUMMY_NODE):
    old_node_arr = tnte_arr[tnte_arr[:,4]=='NODE']
    non_node_arr = tnte_arr[tnte_arr[:,4]!='NODE']
    new_node_arr=rename_node_arr(dummy_node_name,old_node_arr,DUMMY_NODE)
    new_arr = np.vstack((new_node_arr,non_node_arr))
    node_split_dict = make_dummy_split_node_dict(dummy_node_name)
    return new_arr,node_split_dict