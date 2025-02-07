import re
import numpy as np
from _process_cifstr import remove_bracket,remove_tail_number,remove_quotes,extract_quote_lines,extract_xyz_coefficients_and_constant,extract_xyz_lines
from _supercell import make_supercell333,_make_supercell222
from _filter_distance import filter_neighbor_atoms_by_dist,O_filter_neighbor_atoms_by_dist

def find_keyword(keyword,s):
    m = re.search(keyword,s)
    if m:
        return True
    else:
        return False
def extract_unit_cell(cell_info):
    pi = np.pi
    aL, bL, cL, alpha, beta, gamma = cell_info
    aL,bL,cL,alpha,beta,gamma = list(map(float, (aL,bL,cL,alpha,beta,gamma)))
    ax = aL
    ay = 0.0
    az = 0.0
    bx = bL * np.cos(gamma * pi / 180.0)
    by = bL * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = cL * np.cos(beta * pi / 180.0)
    cy = (cL * bL * np.cos(alpha * pi /180.0) - bx * cx) / by
    cz = (cL ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5
    unit_cell = np.asarray([[ax,ay,az],[bx,by,bz],[cx,cy,cz]]).T
    return unit_cell

def read_cif(cif_file):
    with open(cif_file, "r") as f:
        lines = f.readlines()
        nonempty_lines = [line for line in lines if line.strip()]
    # nonempty_lines=lines
    keyword1 = "loop_"
    keyword2="x,\s*y,\s*z" 
    keyword3 = "-x"
    # find the symmetry sector begin with x,y,z, beteen can have space or tab and comma,but just x start, not '-x'
    #keyword2 = "x,\s*y,\s*z"

    loop_key = []
    loop_key.append(0)
    linenumber = 0
    for i in range(len(nonempty_lines)):  # search for keywords and get linenumber
        #m is find keywor1 or (find keyword2 without keyword3)
        m = find_keyword(keyword1, nonempty_lines[i]) or (find_keyword(keyword2, nonempty_lines[i]) and (not find_keyword(keyword3, nonempty_lines[i])))
       
        a = re.search("_cell_length_a", nonempty_lines[i])
        b = re.search("_cell_length_b", nonempty_lines[i])
        c = re.search("_cell_length_c", nonempty_lines[i])
        alpha = re.search("_cell_angle_alpha", nonempty_lines[i])
        beta = re.search("_cell_angle_beta", nonempty_lines[i])
        gamma = re.search("_cell_angle_gamma", nonempty_lines[i])

        if m:
            loop_key.append(linenumber)
        # if not nonempty_lines[i].strip():
        #    loop_key.append(linenumber)

        else:
            if a:
                cell_length_a = remove_bracket(nonempty_lines[i].split()[1])
            elif b:
                cell_length_b = remove_bracket(nonempty_lines[i].split()[1])
            elif c:
                cell_length_c = remove_bracket(nonempty_lines[i].split()[1])
            elif alpha:
                cell_angle_alpha = remove_bracket(nonempty_lines[i].split()[1])
            elif beta:
                cell_angle_beta = remove_bracket(nonempty_lines[i].split()[1])
            elif gamma:
                cell_angle_gamma = remove_bracket(nonempty_lines[i].split()[1])

        linenumber += 1
    loop_key.append(len(nonempty_lines))
    list(set(loop_key))
    loop_key.sort()

    cell_info = [
        cell_length_a,
        cell_length_b,
        cell_length_c,
        cell_angle_alpha,
        cell_angle_beta,
        cell_angle_gamma,
    ]

    # find symmetry sectors and atom_site_sectors
    cif_sectors = []
    for i in range(len(loop_key) - 1):
        cif_sectors.append(nonempty_lines[loop_key[i] : loop_key[i + 1]])
    for i in range(len(cif_sectors)):  # find '\s*x,\s*y,\s*z' symmetry sector
        if re.search(keyword2, cif_sectors[i][0]):
            symmetry_sector = cif_sectors[i]


        if len(cif_sectors[i]) > 1:
            if re.search("_atom_site_label\s+", cif_sectors[i][1]):  # line0 is _loop
                atom_site_sector = cif_sectors[i]

    return cell_info, symmetry_sector, atom_site_sector


def extract_atoms_fcoords_from_lines(atom_site_sector):
    atom_site_lines = []
    keyword = "_"
    for line in atom_site_sector:  # search for keywords and get linenumber
        m = re.search(keyword, line)
        if m is None:
            atom_site_lines.append(line)

    array_atom = np.zeros((len(atom_site_lines), 1), dtype=object)
    array_xyz = np.zeros((len(atom_site_lines), 3))

    for i in range(len(atom_site_lines)):
        for j in [0, 2, 3, 4]:
            if j == 0:
                array_atom[i, j] = remove_tail_number(atom_site_lines[i].split()[j])
            else:
                array_xyz[i, (j - 2)] = remove_bracket(atom_site_lines[i].split()[j])
    return array_atom, array_xyz

def extract_atoms_ccoords_from_lines(cell_info,atom_site_sector):
    atom_site_lines = []
    keyword = "_"
    for line in atom_site_sector:  # search for keywords and get linenumber
        m = re.search(keyword, line)
        if m is None:
            atom_site_lines.append(line)

    array_atom = np.zeros((len(atom_site_lines), 1), dtype=object)
    array_xyz = np.zeros((len(atom_site_lines), 3))

    for i in range(len(atom_site_lines)):
        for j in [0, 2, 3, 4]:
            if j == 0:
                array_atom[i, j] = remove_tail_number(atom_site_lines[i].split()[j])
            else:
                array_xyz[i, (j - 2)] = remove_bracket(atom_site_lines[i].split()[j])
    unit_cell = extract_unit_cell(cell_info)
    array_ccoords=np.dot(unit_cell,array_xyz.T).T   
    com = np.mean(array_ccoords,axis=0) 
    array_ccoords=array_ccoords-com
    return unit_cell,array_atom, array_ccoords


def extract_symmetry_operation_from_lines(symmetry_sector):
    symmetry_operation = []
    for i in range(len(symmetry_sector)):
        # Regular expression to match terms with coefficients and variables
        pattern = r"([+-]?\d*\.?\d*)\s*([xyz])"  # at least find a x/-x/y/-y/z/-z
        match = re.search(pattern, symmetry_sector[i])
        if match:
            string = remove_quotes(symmetry_sector[i].strip("\n"))
            no_space_string = string.replace(" ", "")
            symmetry_operation.append(no_space_string)
    if len(symmetry_operation) < 2:
        print(" no symmetry operation")
        symmetry_operation = ['x,y,z']
    else:
        print(f"apply {len(symmetry_operation)}  symmetry operation")

    return symmetry_operation


def _limit_value_0_1(new_array_metal_xyz):
    for i in range(new_array_metal_xyz.shape[0]):
        for j in range(new_array_metal_xyz.shape[1]):
            if new_array_metal_xyz[i][j] > 3:
                new_array_metal_xyz[i][j] = new_array_metal_xyz[i][j] - 3
            elif new_array_metal_xyz[i][j] > 2:
                new_array_metal_xyz[i][j] = new_array_metal_xyz[i][j] - 2
            elif new_array_metal_xyz[i][j] > 1:
                new_array_metal_xyz[i][j] = new_array_metal_xyz[i][j] - 1
            elif new_array_metal_xyz[i][j] < -2:
                new_array_metal_xyz[i][j] = new_array_metal_xyz[i][j] + 3
            elif new_array_metal_xyz[i][j] < -1:
                new_array_metal_xyz[i][j] = new_array_metal_xyz[i][j] + 2
            elif new_array_metal_xyz[i][j] < 0:
                new_array_metal_xyz[i][j] = new_array_metal_xyz[i][j] + 1
    return new_array_metal_xyz

def limit_value_0_1(new_array_metal_xyz):
    #use np.mod to limit the value in [0,1]
    new_array_metal_xyz=np.mod(new_array_metal_xyz,1)
    return new_array_metal_xyz


def apply_sym_operator(symmetry_operations, array_metal_xyz):
    cell_array_metal_xyz = np.empty((0, 3))
    for sym_line in range(len(symmetry_operations)):
        sym_operation = np.empty((3, 3))
        constants_xyz = np.empty((1, 3))
        for i in range(3):  # x,y,z columns for operation
            coeff_xyz, constant_term = extract_xyz_coefficients_and_constant(
                symmetry_operations[sym_line].split(",")[i]
            )
            # print(f"symmetry_operations[sym_line],{symmetry_operations[sym_line]}\ncoeff_xyz, constant_term,{coeff_xyz, constant_term}")
            sym_operation[:, i] = coeff_xyz
            constants_xyz[:, i] = constant_term
        new_xyz = np.dot(array_metal_xyz, sym_operation) + constants_xyz
        cell_array_metal_xyz = np.vstack((cell_array_metal_xyz, new_xyz))

    round_cell_array_metal_xyz = np.round(limit_value_0_1(cell_array_metal_xyz), 3)
    _, unique_indices = np.unique(round_cell_array_metal_xyz, axis=0, return_index=True)
    unique_indices.sort()
    unique_metal_array = round_cell_array_metal_xyz[unique_indices]

    return unique_metal_array, unique_indices


def extract_type_atoms_ccoords_in_primitive_cell(cif_file, target_type):
    cell_info, symmetry_sector, atom_site_sector = read_cif(cif_file)
    array_atom, array_fcoords= extract_atoms_fcoords_from_lines(atom_site_sector)
    unit_cell = extract_unit_cell(cell_info)
    if len(symmetry_sector) > 1:  # need to apply symmetry operations
        #print(f"apply {len(symmetry_sector)} symmetry operation")
        array_metal_xyz = array_fcoords[array_atom[:, 0] == target_type]
        array_metal_xyz = np.round(array_metal_xyz, 3)
        symmetry_sector_neat = extract_quote_lines(symmetry_sector)
        if len(symmetry_sector_neat) < 2:
            symmetry_sector_neat = extract_xyz_lines(symmetry_sector)
        symmetry_operations = extract_symmetry_operation_from_lines(
            symmetry_sector_neat
        )
        no_sym_array_metal_xyz, no_sym_indices = apply_sym_operator(
            symmetry_operations, array_metal_xyz
        )
        array_metal_fcoords_final = no_sym_array_metal_xyz
        array_ccoords = np.dot(unit_cell,array_fcoords.T).T
        array_metal_ccoords_final = np.dot(unit_cell,array_metal_fcoords_final.T).T
    else:  # P1
        print("P1 cell")
        array_metal_ccoords = array_fcoords[array_atom[:, 0] == target_type]
        array_metal_ccoords_final = np.dot(unit_cell,array_metal_ccoords.T).T
        array_ccoords = np.dot(unit_cell,array_fcoords.T).T
    return cell_info, array_ccoords, array_metal_ccoords_final

def extract_type_atoms_fcoords_in_primitive_cell(cif_file, target_type):
    cell_info, symmetry_sector, atom_site_sector = read_cif(cif_file)
    array_atom, array_xyz= extract_atoms_fcoords_from_lines(atom_site_sector)

    if len(symmetry_sector) > 1:  # need to apply symmetry operations
        #print(f"apply {len(symmetry_sector)} symmetry operation")
        array_metal_xyz = array_xyz[array_atom[:, 0] == target_type]
        array_metal_xyz = np.round(array_metal_xyz, 3)
        symmetry_sector_neat = extract_quote_lines(symmetry_sector)
        if len(symmetry_sector_neat) < 2: # if no quote, then find x,y,z
            symmetry_sector_neat = extract_xyz_lines(symmetry_sector)
        symmetry_operations = extract_symmetry_operation_from_lines(
            symmetry_sector_neat
        )
        no_sym_array_metal_xyz, no_sym_indices = apply_sym_operator(
            symmetry_operations, array_metal_xyz
        )
        array_metal_xyz_final = no_sym_array_metal_xyz

    else:  # P1
        print("P1 cell")
        array_metal_xyz = array_xyz[array_atom[:, 0] == target_type]

        array_metal_xyz_final = np.round(array_metal_xyz, 3)
    return cell_info, array_xyz, array_metal_xyz_final



def extract_node_center(array):
    #(node)array is 3d_array
    node_center=np.empty((len(array),3))
    for i in range(len(array)):
         node_center[i]=np.mean(array[i],axis=0)
    return node_center




def _extract_metal_node_center_cif(
    cif_file, metal_type, atom_number_in_cluster, cluster_distance_threshhold
):
    # array_atom only has the information of atom_type from the cif file and keep the same order with cif
    cell_info, array_atom, array_metal_xyz = extract_type_atoms_fcoords_in_primitive_cell(
        cif_file, metal_type
    )
    metal_xyz_supercell = _make_supercell222(array_metal_xyz)
    # find all possible metal atom in a node which fulfill the both creteria of distance and number in the distance
    node_list = filter_neighbor_atoms_by_dist(
        atom_number_in_cluster,
        metal_xyz_supercell,
        metal_xyz_supercell,
        cluster_distance_threshhold,
    )
   
    # the metal atom indices should only appear once to form the cluster in 222supercell
    node_cluster_indices = np.array(
        np.unique(node_list, axis=0), dtype=int
    )  # each line is a single metal cluster node in supercell
    print(node_cluster_indices.shape,'node_cluster_indices')
    # to get all of possible node centers in a primitive cell by filt array value [0,1]
    # cause the 222supercell is +1 so all of the values are > 0. then the left side do not need to take care
    node_center = np.empty(((len(node_cluster_indices), 3)))

    for i in range(node_cluster_indices.shape[0]):
            node_center[i] = (
                np.mean(metal_xyz_supercell[node_cluster_indices[i]], axis=0),
            )
            node_center[i] = limit_value_0_1(node_center[i])

   

   
    ##filtered_node_center = node_center[~np.any(node_center > 1.1, axis=1)]
    ##node_center[node_center >= 1] -= 1  # any value >=1 then -1
  
    # find unique node center in primitive cell
    #filtered_node_center = limit_value_0_1(node_center)
    node_in_frame, node_in_frame_indices = np.unique(
        np.round(node_center, 4), axis=0, return_index=True
    )
    if node_in_frame.shape[0] == (array_metal_xyz.shape[0] / atom_number_in_cluster):
        # unique_elements, indices = np.unique(node_in_frame, return_index=True,axis=0)
        # print(len(node_in_frame),node_in_frame,indices)
        print(
            f"from the cif_file {cif_file},"
            f" the the number of node centers in frame is {node_in_frame.shape[0]}"
        )

    else:
        print(
            f"please check the cif_file {cif_file},"
            f" the number of node centers in frame is {node_in_frame.shape[0]}"
            f"\nbut it is supposed to be {int(array_metal_xyz.shape[0]/atom_number_in_cluster)}"
        )
    return (
        cell_info,  # cell information: a,b,c,alpha,beta,gamma
        array_metal_xyz,  # all of metal atoms in cif input
        node_in_frame,# node center in a primitive cell in another function 
        metal_xyz_supercell,  # all the metal atoms positions in a 222supercell
        node_cluster_indices[node_in_frame_indices],
        # find the node center in primitive cell but the corresponding position
        # could be in supercell and is translated to primitive cell.
        # the indices can be used to extract the node in the supercell
        # node_center<-->node cluster, they share the same index, the former is the mean of the latter
    )


def search_O_cluster_in_node(
    cif_file, define_type, node_in_frame, O_atom_numbers_in_a_node, distance_threshhold
):
    # find cluster of O in node
    array_node_center_in_frame = node_in_frame
    cell_info, atom_O_name, array_atom_O = extract_type_atoms_fcoords_in_primitive_cell(
        cif_file, define_type
    )
    array_atom_O = make_supercell333(array_atom_O)

    #print(f"atom_O_name{atom_O_name}")
    Oinnode = O_filter_neighbor_atoms_by_dist(
        O_atom_numbers_in_a_node,
        array_node_center_in_frame,
        array_atom_O,
        distance_threshhold,
    )
    # each node center should have a range of other nonmetal atoms in the cluster like O or OH

    O_innode_indices = np.array(
        np.unique(Oinnode, axis=0), dtype=int
    )  # each line is a single metal cluster node in supercell

    # get pairs of metal center and O center
    # and then get paired metal cluster and O cluster in same node for adding dummy atoms
    O_clusters = np.empty((O_innode_indices.shape[0], O_atom_numbers_in_a_node, 3))
    O_node_center = np.empty(((0, 3)))
    for i in range(
        O_innode_indices.shape[0]
    ):  # for every node center (== O cluster center)
        O_node_center = np.vstack(
            (
                O_node_center,
                np.mean(array_atom_O[O_innode_indices[i, :]], axis=0),
            )
        )
        O_clusters[i] = array_atom_O[O_innode_indices[i, :]]
    return O_clusters, O_node_center




def reorder_O_clusters_based_on_node_order(metal_O_pair, O_clusters):
    reordered_O_clusters_array = np.empty(O_clusters.shape)
    for i in range(len(metal_O_pair)):
        O_cluster = O_clusters[metal_O_pair[i][1]]
        reordered_O_clusters_array[i] = O_cluster
        reordered_O_cluster_center = np.mean(O_cluster, axis=0)
        #print(f"reordered_O_center{reordered_O_cluster_center}")
    return reordered_O_clusters_array
