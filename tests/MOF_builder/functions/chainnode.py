import numpy as np
from _readcif import extract_type_atoms_fcoords_in_primitive_cell,read_cif,extract_atoms_fcoords_from_lines,extract_atoms_ccoords_from_lines

def process_chain_node(chain_node_cif, target_type):
    _, _, node_target_atoms=extract_type_atoms_fcoords_in_primitive_cell(chain_node_cif, target_type)
    _,_, node_x_vecs=extract_type_atoms_fcoords_in_primitive_cell(chain_node_cif, 'X')

    node_cell_info, symmetry_sector, node_atom_site_sector = read_cif(chain_node_cif)
    node_atom, node_xyz = extract_atoms_fcoords_from_lines(node_atom_site_sector)
    node_unit_cell,node_atom, node_ccoords = extract_atoms_ccoords_from_lines(node_cell_info,node_atom_site_sector)
    node_com = np.mean(node_target_atoms, axis=0)
    chain_node_fcoords = node_xyz - node_com
    metal_fvec = node_target_atoms[0]-node_target_atoms[1]
    node_pillar_fvec = metal_fvec/np.linalg.norm(metal_fvec) 
    node_x_vecs = node_x_vecs - node_com

    return node_unit_cell,node_atom,node_pillar_fvec, node_x_vecs, chain_node_fcoords


def process_node(chain_node_cif, target_type):
    _, _, node_target_atoms=extract_type_atoms_fcoords_in_primitive_cell(chain_node_cif, target_type)
    _,_, node_x_vecs=extract_type_atoms_fcoords_in_primitive_cell(chain_node_cif, 'X')

    node_cell_info, symmetry_sector, node_atom_site_sector = read_cif(chain_node_cif)
    node_atom, node_xyz = extract_atoms_fcoords_from_lines(node_atom_site_sector)
    node_unit_cell,node_atom, node_ccoords = extract_atoms_ccoords_from_lines(node_cell_info,node_atom_site_sector)
    node_com = np.mean(node_target_atoms, axis=0)
    chain_node_fcoords = node_xyz - node_com
    #metal_fvec = node_target_atoms[0]-node_target_atoms[1]
    #node_pillar_fvec = metal_fvec/np.linalg.norm(metal_fvec) 
    node_x_vecs = node_x_vecs - node_com

    return node_unit_cell,node_atom, node_x_vecs, chain_node_fcoords

if __name__ == '__main__':
    chain_node_cif = '21Alchain.cif'
    chain_node_target_type = 'Al'
    node_unit_cell,node_pillar_fvec, node_x_vecs, chain_node_fcoords = process_chain_node(chain_node_cif, chain_node_target_type)
    print(node_unit_cell)
    print(node_pillar_fvec)
    print(node_x_vecs)
    print(chain_node_fcoords)