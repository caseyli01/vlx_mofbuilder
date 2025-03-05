import spglib
import re
import numpy as np
import spglib
import os
import datetime


def similar_group_name(group):
    # make first letter uppercase
    if ":" in group:
        group = group.split(":")[0]
    return re.sub(r"[^a-zA-Z0-9]", "", group)


# https://cci.lbl.gov/sginfo/hall_symbols.html
# 2d https://pscf.readthedocs.io/en/latest/groups.html
def fetch_group_number(group):
    numbers = []
    if group == "C12/m1":
        numbers = [71]
        return numbers
    if group == "Cmca":
        numbers = [304]  # ? a/e
        return numbers
    if group == "C2mm":
        numbers = [188]
        return numbers
    if group == "P121/n1":
        numbers = [89]
        return numbers

    if group == "Ccca:2":
        numbers = [325]  # ? a/e
        return numbers
    if group == "Cmma":
        numbers = [317]  # ? a/e
        return numbers
    if group == "C12/c1":
        numbers = [107]
        return numbers
    if group == "I12/a1":
        numbers = [107]
        return numbers
    if group == "I12/m1":
        numbers = [71]
        return numbers
    if group == "P121/c1":
        numbers = [89]
        return numbers
    # if group =="llw-z":
    #    numbers = [196]
    #    return numbers
    if group == "R-3:H":
        numbers = [148]
        return numbers
    if group == "R3m:H":
        numbers = [160]
        return numbers

    if group == "R3c:H":
        numbers = [161]
        return numbers

    if group == "R-3m:H":
        numbers = [166]
        return numbers
    if group == "R-3c:H":
        numbers = [167]
        return numbers

    if group[0] == "p":
        # print("2d group")
        return [-1]
    if group[0] == "c":
        # print("2d group")
        return [-1]

    for key in space_group_info_dict.keys():
        if group == key:
            numbers = [space_group_info_dict[key].hall_number]
            return numbers
        elif similar_group_name(group) == similar_group_name(key):
            # print('hall number:', space_group_info_dict[key])
            numbers.append(space_group_info_dict[key].hall_number)
    if len(numbers) > 1:
        print()
        print("more than one space group found", group, numbers)
        return [-1]
    elif len(numbers) == 0:
        print("*" * 10)
        print("no space group found", group)
        print("*" * 10)
        return [-1]
    return numbers


def extract_netinfo(net):
    # extract name, group, cell, nodes, and edge centers from net file
    name = net[1].strip("\n").split()[1]
    group = net[2].strip("\n").split()[1]
    cell = net[3].strip("\n").split()[1:]
    cell = list(map(float, cell))

    nodes = []
    edge_centers = []
    for line in net[4:]:
        line = line.strip("\n")
        if "NODE" in line:
            nodes.append(line.split()[1:])
        if "EDGE_CENTER" in line:
            edge_centers.append(line.split()[2:])
    # if len(nodes) <2:
    #    print('ditopic network')
    # elif len(nodes) == 2:
    #    print('multitopic network')
    if len(nodes) > 2:
        return False, None, None, None, None, None
    #    print('cannot handle this network')
    group_number = fetch_group_number(group)[
        0
    ]  # only one number should be in the returned list
    if group_number == -1:
        return False, None, None, None, None, None

    return (name, group, group_number, cell, nodes, edge_centers)


def process_nodes(nodes):
    # if len(nodes) ==1, then it is a ditopic network, return nodes
    # if len(nodes) ==2, then it is a multitopic network, Vnodes and ECnodes
    if len(nodes) == 1:
        V_con = int(nodes[0][1])
        EC_con = 0
        Vnodes = nodes[0][2:]
        ECnodes = []
        return V_con, EC_con, Vnodes, ECnodes
    elif len(nodes) == 2:
        if int(nodes[0][1]) < int(nodes[1][1]):
            V_con = int(nodes[1][1])
            EC_con = int(nodes[0][1])
            Vnodes = nodes[1][2:]
            ECnodes = nodes[0][2:]
        else:
            V_con = int(nodes[0][1])
            EC_con = int(nodes[1][1])
            Vnodes = nodes[0][2:]
            ECnodes = nodes[1][2:]
        return V_con, EC_con, Vnodes, ECnodes


# fetch_group_number('Fm-3m')[0] #523 as hall_number
def extract_unit_cell(cell_info):
    pi = np.pi
    aL, bL, cL, alpha, beta, gamma = cell_info
    aL, bL, cL, alpha, beta, gamma = list(map(float, (aL, bL, cL, alpha, beta, gamma)))
    ax = aL
    ay = 0.0
    az = 0.0
    bx = bL * np.cos(gamma * pi / 180.0)
    by = bL * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = cL * np.cos(beta * pi / 180.0)
    cy = (cL * bL * np.cos(alpha * pi / 180.0) - bx * cx) / by
    cz = (cL**2.0 - cx**2.0 - cy**2.0) ** 0.5
    unit_cell = np.asarray([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]).T
    return unit_cell


def apply_symmetry_operations(positions, space_group_number):
    """
    Apply symmetry operations to positions using a standardized primitive cell.
    """

    # Get symmetry operations for the space group
    dataset = spglib.get_symmetry_from_database(space_group_number)
    rotations = dataset["rotations"]
    translations = dataset["translations"]

    # Apply symmetry operations
    transformed_positions = set()

    for rot, trans in zip(rotations, translations):
        for pos in positions:
            new_pos = np.dot(rot, pos) + trans  # Apply rotation and translation
            new_pos = np.mod(new_pos, 1)  # Ensure periodic boundary conditions
            transformed_positions.add(tuple(new_pos))  # Store unique positions

    # Convert to numpy array and sort for consistency
    transformed_positions = np.array(sorted(transformed_positions))

    return transformed_positions


def get_primcell_points_ditopic(space_group_number, node_positions, edge_positions):
    # **🔹 Process nodes and edges separately**
    symmetrized_nodes = apply_symmetry_operations(node_positions, space_group_number)
    symmetrized_edges = apply_symmetry_operations(edge_positions, space_group_number)

    # **🔹 Print results in the required format**
    # print("Generated Symmetrized Positions:\n")
    all_lines = []
    # Print nodes
    for i, pos in enumerate(symmetrized_nodes, start=1):
        # print(f"V{i:<3}  V   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}")
        all_lines.append(f"V{i:<3}  V   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}\n")

    # Print edges
    for i, pos in enumerate(symmetrized_edges, start=1):
        # print(f"E{i:<3}  E   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}")
        all_lines.append(f"E{i:<3}  E   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}\n")
    return all_lines


def check_edge_connect_sametypenode(edge_positions, Vnodes, ECnodes):
    return True
    # if any edge is between two Vnodes or two ECnodes, then skip this network, return False
    for e in edge_positions:
        for vnode in Vnodes:
            side = 2 * e - vnode
            # if point side is in Vnodes, then skip this network, use isclose to compare float numbers
            if any(np.isclose(side, vnode, atol=1e-5).all() for vnode in Vnodes):
                print("skip this network", e, vnode)
                return False
        for ecnode in ECnodes:
            side = 2 * e - ecnode
            # if point side is in ECnodes, then skip this network, use isclose to compare float numbers
            if any(np.isclose(side, ecnode, atol=1e-5).all() for ecnode in ECnodes):
                print("skip this network", e, ecnode)
                return False
    return True


def get_primcell_multitopic(
    space_group_number, node_positions, ECnode_positions, edge_positions
):
    # **🔹 Process nodes and edges separately**
    symmetrized_nodes = apply_symmetry_operations(node_positions, space_group_number)
    symmetrized_nodes_EC = apply_symmetry_operations(
        ECnode_positions, space_group_number
    )
    symmetrized_edges = apply_symmetry_operations(edge_positions, space_group_number)
    if not check_edge_connect_sametypenode(
        symmetrized_edges, symmetrized_nodes, symmetrized_nodes_EC
    ):
        return False

    # **🔹 Print results in the required format**
    # print("Generated Symmetrized Positions:\n")
    all_lines = []
    # Print nodes
    for i, pos in enumerate(symmetrized_nodes, start=1):
        # print(f"V{i:<3}  V   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}")
        all_lines.append(f"V{i:<3}  V   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}\n")

    for i, pos in enumerate(symmetrized_nodes_EC, start=1):
        # print(f"V{i:<3}  V   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}")
        all_lines.append(f"EC{i:<3}  EC   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}\n")

    # Print edges
    for i, pos in enumerate(symmetrized_edges, start=1):
        # print(f"E{i:<3}  E   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}")
        all_lines.append(f"E{i:<3}  E   {pos[0]:.5f}  {pos[1]:.5f}  {pos[2]:.5f}\n")
    return all_lines


def write_cif_nobond(lines, params, cifname, infoline):
    a, b, c, alpha, beta, gamma = params
    os.makedirs("output_cifs", exist_ok=True)
    opath = os.path.join("output_cifs", cifname)
    print(opath)

    with open(opath, "w") as new_cif:
        new_cif.write("data_" + cifname[0:-4] + infoline + "\n")
        new_cif.write(
            "_audit_creation_date              "
            + datetime.datetime.today().strftime("%Y-%m-%d")
            + "\n"
        )
        new_cif.write("_audit_creation_method            'MOFbuilder_1.0'" + "\n")
        new_cif.write("_symmetry_space_group_name_H-M    'P1'" + "\n")
        new_cif.write("_symmetry_Int_Tables_number       1" + "\n")
        new_cif.write("_symmetry_cell_setting            triclinic" + "\n")
        new_cif.write("loop_" + "\n")
        new_cif.write("_symmetry_equiv_pos_as_xyz" + "\n")
        new_cif.write("  x,y,z" + "\n")
        if float(a) < 10:
            a = float(a) * 10
            b = float(b) * 10
            c = float(c) * 10
        new_cif.write("_cell_length_a                    " + str(a) + "\n")
        new_cif.write("_cell_length_b                    " + str(b) + "\n")
        new_cif.write("_cell_length_c                    " + str(c) + "\n")
        new_cif.write("_cell_angle_alpha                 " + str(alpha) + "\n")
        new_cif.write("_cell_angle_beta                  " + str(beta) + "\n")
        new_cif.write("_cell_angle_gamma                 " + str(gamma) + "\n")
        new_cif.write("loop_" + "\n")
        new_cif.write("_atom_site_label" + "\n")
        new_cif.write("_atom_site_type_symbol" + "\n")
        new_cif.write("_atom_site_fract_x" + "\n")
        new_cif.write("_atom_site_fract_y" + "\n")
        new_cif.write("_atom_site_fract_z" + "\n")

        new_cif.writelines(lines)

        new_cif.write("loop_" + "\n")


# check if the space group is in the space group info dict
# count = 0
# for key in space_group_info_dict.keys():
#    if group == key or similar_group_name(group) == similar_group_name(key):
#        print('hall number:', space_group_info_dict[key])
#        count += 1
# if count >1:
#    print('more than one space group found')
# elif count == 0:
#    print('no space group found')

# convert group name to hall number
# looking up the space group number from the space group name
# check the group name is in the space group info dict keys, or similar to the space group name
# extract all number and letter from the group name, use re to replace all non letter non number characters with ''
# then check if the group name is in the space group info dict keys

# make a dict of space group info
# key: international_short
# value: space group info
# space group info: (number, international_short, international_full, international, schoenflies, hall)
space_group_info_dict = {}
for i in range(1, 531):
    space_group_info_dict[spglib.get_spacegroup_type(i)["international_short"]] = (
        spglib.get_spacegroup_type(i)
    )

with open("/Users/chenxili/GitHub/vlx_mofbuilder/data/RCSRnets-2019-06-01.cgd") as f:
    lines = f.readlines()
# extract the space group info from the cgd file, start with CRYSTAL end with END
starts = []
ends = []
for i, line in enumerate(lines):
    if "CRYSTAL" in line or "crystal" in line:
        if "#" in line:
            continue
        starts.append(i)
    if "END" in line or "end" in line:
        if "#" in line:
            continue
        end = i
        ends.append(i)
# check if the number of starts and ends are the same
if len(starts) != len(ends):
    print("Error: the number of starts and ends are not the same")

# extract the space group info from the cgd file
crystal_infos = []
for i in range(len(starts)):
    crystal_info = []
    if ends[i] - starts[i] < 1:
        print(
            "Error: the crystal info is not complete",
            starts[i],
            ends[i],
            lines[starts[i] : ends[i] + 1],
        )
        continue

    for j in range(starts[i], ends[i] + 1):
        crystal_info.append(lines[j])
    crystal_infos.append(crystal_info)


for net in crystal_infos:
    if len(net) < 2:
        print(net)
        continue
    name, group, space_group_hallnumber, cell, nodes, edge_centers = extract_netinfo(
        net
    )

    if space_group_hallnumber == -1:
        continue
    if not name:
        continue
    if len(edge_centers) == 0:
        print("no edge centers")
        continue
    V_con, EC_con, Vnodes, ECnodes = process_nodes(nodes)
    if EC_con == 0:
        # print('ditopic network')
        node_positions = np.array(Vnodes).reshape((1, 3)).astype(float)
        edge_positions = np.array(edge_centers).astype(float)
        lines = get_primcell_points_ditopic(
            space_group_hallnumber, node_positions, edge_positions
        )
        infoline = f"_{group}  hall_number: {space_group_hallnumber}, V_con: {V_con}"
        write_cif_nobond(lines, cell, name + ".cif", infoline)

    else:
        # print('multitopic network')
        node_positions = np.array(Vnodes).reshape((1, 3)).astype(float)
        EC_node_positions = np.array(ECnodes).reshape((1, 3)).astype(float)
        edge_positions = np.array(edge_centers).astype(float)
        lines = get_primcell_multitopic(
            space_group_hallnumber, node_positions, EC_node_positions, edge_positions
        )
        if not lines:
            print("skip this network", name, group)
            continue
        infoline = f"_{group}  hall_number: {space_group_hallnumber}, V_con: {V_con}, EC_con: {EC_con}"
        write_cif_nobond(lines, cell, name + ".cif", infoline)
