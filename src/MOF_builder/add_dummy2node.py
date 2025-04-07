import numpy as np
import re
import os
import networkx as nx
import matplotlib.pyplot as plt
import datetime
from _superimpose import superimpose
import veloxchem as vlx

pi = np.pi

vertices = (
    "V",
    "Er",
    "Ti",
    "Ce",
    "S",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Ac",
    "Ag",
    "Al",
    "Am",
    "Au",
    "Ba",
    "Be",
    "Bi",
    "Bk",
    "Ca",
    "Cd",
    "Ce",
    "Cf",
    "Cm",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Dy",
    "Er",
    "Es",
    "Eu",
    "Fe",
    "Fm",
    "Ga",
    "Gd",
    "Hf",
    "Hg",
    "Ho",
    "In",
    "Ir",
    "K",
    "La",
    "Li",
    "Lr",
    "Lu",
    "Md",
    "Mg",
    "Mn",
    "Mo",
    "Na",
    "Nb",
    "Nd",
    "Ni",
    "No",
    "Np",
    "Os",
    "Pa",
    "Pb",
    "Pd",
    "Pm",
    "Pr",
    "Pt",
    "Pu",
    "Ra",
    "Rb",
    "Re",
    "Rh",
    "Ru",
    "Sc",
    "Sm",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Th",
    "Ti",
    "Tl",
    "Tm",
    "U",
    "V",
    "W",
    "Y",
    "Yb",
    "Zn",
    "Zr",
    "X",
)
pi = np.pi


def create_cif(name_label_coords, bonds, foldername, cifname):
    opath = os.path.join(foldername, cifname)

    with open(opath, "w") as out:
        out.write("data_" + cifname[0:-4] + "\n")
        out.write("_audit_creation_date              " +
                  datetime.datetime.today().strftime("%Y-%m-%d") + "\n")
        out.write("_audit_creation_method            'MOFbuilder'" + "\n")
        out.write("_symmetry_space_group_name_H-M    'P1'" + "\n")
        out.write("_symmetry_Int_Tables_number       1" + "\n")
        out.write("_symmetry_cell_setting            triclinic" + "\n")
        out.write("loop_" + "\n")
        out.write("_symmetry_equiv_pos_as_xyz" + "\n")
        out.write("  x,y,z" + "\n")
        out.write("_cell_length_a                    " + "50" + "\n")
        out.write("_cell_length_b                    " + "50" + "\n")
        out.write("_cell_length_c                    " + "50" + "\n")
        out.write("_cell_angle_alpha                 " + "90" + "\n")
        out.write("_cell_angle_beta                  " + "90" + "\n")
        out.write("_cell_angle_gamma                 " + "90" + "\n")
        out.write("loop_" + "\n")
        out.write("_atom_site_label" + "\n")
        out.write("_atom_site_type_symbol" + "\n")
        out.write("_atom_site_fract_x" + "\n")
        out.write("_atom_site_fract_y" + "\n")
        out.write("_atom_site_fract_z" + "\n")

        for l in name_label_coords:
            vec = list(map(float, l[2:5]))
            m = np.array(([0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]))
            cvec = np.dot(m, vec)

            cvec = np.mod(cvec, 1)
            extra = "   0.00000  Uiso   1.00       -0.000000"
            out.write("{:7} {:>4} {:>15} {:>15} {:>15}".format(
                l[0],
                l[1],
                "%.10f" % np.round(cvec[0], 10),
                "%.10f" % np.round(cvec[1], 10),
                "%.10f" % np.round(cvec[2], 10),
            ))
            out.write(extra)
            out.write("\n")

        out.write("loop_" + "\n")
        out.write("_geom_bond_atom_site_label_1" + "\n")
        out.write("_geom_bond_atom_site_label_2" + "\n")
        out.write("_geom_bond_distance" + "\n")
        out.write("_geom_bond_site_symmetry_2" + "\n")
        out.write("_ccdc_geom_bond_type" + "\n")

        for b in bonds:
            out.write("{:7} {:>7} {:>5} {:>7} {:>3}".format(
                b[0], b[1], "%.3f" % float(b[2]), b[3], b[4]))
            out.write("\n")


def isfloat(value):
    """
    determines if a value is a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def nn(string):
    return re.sub("[^a-zA-Z]", "", string)


def nl(string):
    return re.sub("[^0-9]", "", string)


def isvert(line):
    """
    identifies coordinates in CIFs
    """
    if len(line) >= 5:
        if (nn(line[0]) in vertices and line[1] in vertices
                and False not in map(isfloat, line[2:5])):
            return True
        else:
            return False


def isedge(line):
    """
    identifies bonding in cifs
    """
    if len(line) >= 5:
        if (nn(line[0]) in vertices and nn(line[1]) in vertices
                and isfloat(line[2]) and line[-1] in ("S", "D", "T", "A")):
            return True
        else:
            return False


def cif2G(cifpath):
    path = cifpath
    cifname = os.path.basename(cifpath)

    with open(path, "r") as cif:
        cif = cif.read()
        cif = list(filter(None, cif.split("\n")))

    G = nx.MultiGraph()

    edge_exist = False
    for line in cif:
        s = line.split()
        if "_cell_length_a" in line:
            aL = s[1]
        if "_cell_length_b" in line:
            bL = s[1]
        if "_cell_length_c" in line:
            cL = s[1]
        if "_cell_angle_alpha" in line:
            alpha = s[1]
        if "_cell_angle_beta" in line:
            beta = s[1]
        if "_cell_angle_gamma" in line:
            gamma = s[1]

    aL, bL, cL, alpha, beta, gamma = list(
        map(float, (aL, bL, cL, alpha, beta, gamma)))
    ax = aL
    ay = 0.0
    az = 0.0
    bx = bL * np.cos(gamma * pi / 180.0)
    by = bL * np.sin(gamma * pi / 180.0)
    bz = 0.0
    cx = cL * np.cos(beta * pi / 180.0)
    cy = (cL * bL * np.cos(alpha * pi / 180.0) - bx * cx) / by
    cz = (cL**2.0 - cx**2.0 - cy**2.0)**0.5
    unit_cell = np.asarray([[ax, ay, az], [bx, by, bz], [cx, cy, cz]]).T

    nc = 0  # @nc node_index
    ne = 0  # @ne edge_index

    types = []  # vertex type
    aae = []  # edge numbers?

    types_append = types.append
    aae_append = aae.append

    max_le = 1.0e6

    for line in cif:
        s = line.split()

        if isvert(s):
            ty = re.sub("[^a-zA-Z]", "", s[0])
            types_append(ty)
            nc += 1
            f_nvec = np.asarray(list(map(float, s[2:5])))
            c_nvec = np.dot(unit_cell, f_nvec)
            G.add_node(
                s[0],
                type=s[1],
                index=nc,
                ccoords=c_nvec,
                fcoords=f_nvec,
                cn=[],
                cifname=[],
            )

    S = [G.subgraph(c).copy() for c in nx.connected_components(G)
         ]  # a list of all the subgraphs representing connected components.
    # if len(S) > 1:
    # catenation = True #if there is more than one connected component
    # else:
    # catenation = False

    for net in [(s, unit_cell, cifname, aL, bL, cL, alpha, beta, gamma, max_le)
                for s in S]:
        SG = nx.MultiGraph()
        cns = []
        cns_append = cns.append

        count = 0
        for node in net[0].nodes(data=True):
            n, data = node
            cn = G.degree(
                n
            )  # Gets the degree (number of connections) of node n in the main graph G.
            ty = re.sub(
                "[0-9]", "", n
            )  # Removes numeric characters from the node identifier to get its type.
            # print(f"data{data}")
            cns_append((cn, ty))
            SG.add_node(n, **data)

            if count == 0:
                start = data[
                    "fcoords"]  # Sets the starting coordinates (start) to the fractional coordinates of the first node.
            count += 1

        count = 0
        e_types = []
        e_types_append = e_types.append

        for edge in net[0].edges(keys=True, data=True):
            count += 1
            e0, e1, key, data = edge
            key = tuple(
                [count] +
                [k for k in key[1:]
                 ])  # Adjusts the edge key by prepending the current count.
            data["index"] = count  # Sets the index attribute of the edge
            l = sorted(
                [re.sub("[^a-zA-Z]", "", e0),
                 re.sub("[^a-zA-Z]", "", e1)]
            )  # Sorts and strips non-alphabetic characters from node identifiers to determine the edge type.
            e_types_append((l[0], l[1]))

            SG.add_edge(e0, e1, key=key, type=(l[0], l[1]), **data)

        # yield (SG, start, unit_cell, set(cns), set(e_types), cifname, aL, bL, cL, alpha, beta, gamma, max_le, catenation)
        return (SG, unit_cell)


def lines_of_atoms(subgraph, subgraph_nodes):
    count = 1
    rows = []
    for sn in subgraph_nodes:
        label = subgraph.nodes[sn]["type"]
        coord = subgraph.nodes[sn]["ccoords"]
        name = sn  # label+str(count)
        count += 1
        rows.append([name, label, coord[0], coord[1], coord[2]])
    #sort rows by atom label
    if "D" in [nn(i) for i in subgraph_nodes]:
        rows = sorted(rows, key=lambda x: (x[1], x[0]))
    else:  #use reverse order for non dummy nodes
        rows = sorted(rows, key=lambda x: (x[1], x[0]), reverse=True)

    return rows


def dummynode_get_bonds_from_subgraph(subgraph):
    bonds = []
    for e in list(subgraph.edges()):
        atom1 = e[0]
        atom2 = e[1]
        length = 1  # subgraph.edges[e]['weight']
        sym = "."
        if nn(atom1) == "X" or nn(atom2[0]) == "X":
            bond_type = "A"
        else:
            bond_type = "S"
        bonds.append([atom1, atom2, length, sym, bond_type])

    return bonds


def get_diagonal_pair(array1):
    max_length = 0
    min_length = 100
    for i in range(1):
        for j in range(1, len(array1)):
            v = array1[i] - array1[j]
            length = np.linalg.norm(v)
            if length > max_length:
                max_length = length
                diag_pair = j
            if length < min_length:
                min_length = length
                near_pair = j
    return np.asarray((array1[0], array1[diag_pair], array1[near_pair]))


def fetch_template(metal):
    if metal == "Zr":  # Zr-D = 1A
        template_Zr_D = np.array((
            [0.70710678, 0.70710678, 0.0],
            [-0.70710678, 0.70710678, 0.0],
            [-0.70710678, -0.70710678, 0.0],
            [0.70710678, -0.70710678, 0.0],
            [0.0, 0.70710678, 0.70710678],
            [0.0, -0.70710678, 0.70710678],
            [0.0, -0.70710678, -0.70710678],
            [0.0, 0.70710678, -0.70710678],
        ))
        return template_Zr_D
    elif metal == "Hf":  # Hf-D = 1A
        template_Hf_D = np.array((
            [0.70710678, 0.70710678, 0.0],
            [-0.70710678, 0.70710678, 0.0],
            [-0.70710678, -0.70710678, 0.0],
            [0.70710678, -0.70710678, 0.0],
            [0.0, 0.70710678, 0.70710678],
            [0.0, -0.70710678, 0.70710678],
            [0.0, -0.70710678, -0.70710678],
            [0.0, 0.70710678, -0.70710678],
        ))
        return template_Hf_D
    elif metal == "Al":  # Al-D = 0.74A
        template_Al_D = np.array((
            [0.74, 0.0, 0.0],
            [0.0, 0.74, 0.0],
            [0.0, 0.0, 0.74],
            [-0.74, 0.0, 0.0],
            [0.0, -0.74, 0.0],
            [0.0, 0.0, -0.74],
        ))
        return template_Al_D
    elif metal == "Fe":  # Fe-D = 0.86A
        template_Fe_D = np.array((
            [0.86, 0.0, 0.0],
            [0.0, 0.86, 0.0],
            [0.0, 0.0, 0.86],
            [-0.86, 0.0, 0.0],
            [0.0, -0.86, 0.0],
            [0.0, 0.0, -0.86],
        ))
        return template_Fe_D
    elif metal == "Cr":  # Cr-D = 0.82A
        template_Cr_D = np.array((
            [0.82, 0.0, 0.0],
            [0.0, 0.82, 0.0],
            [0.0, 0.0, 0.82],
            [-0.82, 0.0, 0.0],
            [0.0, -0.82, 0.0],
            [0.0, 0.0, -0.82],
        ))
        return template_Cr_D


def order_ccoords(d_ccoords, template_Zr_D, target_metal_coords):
    d_ccoords = d_ccoords - target_metal_coords  # centering to origin

    # template_center is already at origin
    min_dist, rot, tran = superimpose(template_Zr_D, d_ccoords)
    ordered_ccoords = np.dot(template_Zr_D, rot) + target_metal_coords
    return ordered_ccoords


def add_dummy_atoms_nodecif(ciffile, metal):
    if metal in ["Zr", "Hf"]:
        metal_valence = 4
    elif metal in ["Al", "Fe", "Cr"]:
        metal_valence = 3

    dummy_ciffile = ciffile.removesuffix(".cif") + "_dummy.cif"
    template_metal_D = fetch_template(metal)
    sG, unit_cell = cif2G(ciffile)
    # options = {
    #            "font_size": 10,
    #            "node_size": 150,
    #            "node_color": "white",
    #            "edgecolors": "black",
    #            "linewidths": 1,
    #            "width": 1,
    #        }
    ##nx.draw_networkx(sG,**options)
    ind_max = max([int(nl(n)) for n in list(sG.nodes())])
    metal_nodes = [n for n in list(sG.nodes()) if nn(n) == metal]
    count = ind_max + 1
    for mn in metal_nodes:
        neighbor_nodes = list(nx.neighbors(sG, mn))
        Ocheck = all(nn(i) == "O" for i in neighbor_nodes)
        if len(neighbor_nodes) == 2 * metal_valence and Ocheck:
            # add dummy
            beginning_cc = sG.nodes[mn]["ccoords"]
            beginning_fc = sG.nodes[mn]["fcoords"]
            index = sG.nodes[mn]["index"]
            d_ccoords = []
            d_fcoords = []
            for nO in neighbor_nodes:
                sO = sG.nodes[nO]["ccoords"]
                cnorm_vec = (sO - beginning_cc) / np.linalg.norm(
                    sO - beginning_cc)  # 1 angstrom
                # f_vec = np.dot(np.linalg.inv(unit_cell),cnorm_vec)
                d_ccoord = beginning_cc + cnorm_vec
                # d_fcoord = beginning_fc+f_vec
                d_ccoords.append(d_ccoord)
                # d_fcoords.append(d_fcoord)
                sG.remove_edge(mn, nO)
            # order ccords based on template order
            ordered_ccoords = order_ccoords(d_ccoords, template_metal_D)
            ordered_fcoords = np.dot(np.linalg.inv(unit_cell),
                                     ordered_ccoords.T).T
            for row in range(len(d_ccoords)):
                sG.add_node(
                    "D" + str(count),
                    type="D",
                    index=index,
                    ccoords=ordered_ccoords[row],
                    fcoords=ordered_fcoords[row],
                    cn=[],
                    cifname=[],
                )
                sG.add_edge(mn, "D" + str(count))
                count += 1
    # nx.draw_networkx(sG,**options)

    # print(nx.number_of_nodes(sG))

    arr = np.empty((sG.number_of_nodes(), 4), dtype="O")
    row = 0
    for n in list(sG.nodes()):
        arr[row, 0] = n
        s = sG.nodes[n]["ccoords"]
        arr[row, 1] = s[0]
        arr[row, 2] = s[1]
        arr[row, 3] = s[2]
        row += 1

    sG_subparts = [
        c for c in sorted(nx.connected_components(sG), key=len, reverse=True)
    ]

    head = []
    tail = []
    for sub in sG_subparts:
        l = [nn(i) for i in sub]
        if "X" not in l:
            head.append(sorted(sub))
        else:
            tail.append(sorted(sub))

    sub_headlens = [len(i) for i in head]
    sub_taillens = [len(i) for i in tail]
    # sum(sub_headlens)+sum(sub_taillens),sub_headlens,sub_taillens

    dummy_count = 0
    hho_count = 0
    ho_count = 0
    o_count = 0
    ooc_count = 0

    for i in sub_headlens:
        if i == 3:
            hho_count += 1
        elif i == 2:
            ho_count += 1
        elif i == 1:
            o_count += 1
        else:
            dummy_count += 1
            dummy_res_len = i
    for j in sub_taillens:
        if j == 3:
            ooc_count += 1

    node_split_dict = {}
    node_split_dict["HHO_count"] = hho_count
    node_split_dict["HO_count"] = ho_count
    node_split_dict["O_count"] = o_count
    node_split_dict["METAL_count"] = dummy_count
    node_split_dict["dummy_res_len"] = dummy_res_len
    node_split_dict["OOC_count"] = ooc_count
    node_split_dict[
        "inres_count"] = hho_count + ho_count + o_count + dummy_count

    node_dict_name = dummy_ciffile.removesuffix(".cif") + "_dict"
    node_dict_path = node_dict_name
    with open(node_dict_path, "w") as f:
        for key in list(node_split_dict):
            f.write("{:20} {:<4}".format(key, node_split_dict[key]))
            f.write("\n")

    subpart_nodes = head + tail
    all_lines = []
    all_bonds = []

    for count_i in range(len(subpart_nodes)):
        subnodes = subpart_nodes[count_i]
        subgraph = nx.subgraph(sG, subnodes)
        all_lines += lines_of_atoms(subgraph, sorted(subnodes))
        all_bonds += dummynode_get_bonds_from_subgraph(subgraph)

    create_cif(
        all_lines,
        all_bonds,
        os.path.dirname(dummy_ciffile),
        os.path.basename(dummy_ciffile),
    )


# process node pdb file
def nodepdb(filename):
    xyzlines = ""
    inputfile = str(filename)
    with open(inputfile, "r") as fp:
        content = fp.readlines()
        # linesnumber = len(content)
    data = []
    for line in content:
        line = line.strip()
        if len(line) > 0:  # skip blank line
            if line[0:6] == "ATOM" or line[0:6] == "HETATM":
                value_atom = line[12:16].strip()  # atom_label
                value_x = float(line[30:38])  # x
                value_y = float(line[38:46])  # y
                value_z = float(line[46:54])  # z
                atom_type = line[67:80].strip()  # atom_note
                formatted_xyz_line = "{:6s}  {:8.3f} {:8.3f} {:8.3f}".format(
                    atom_type, value_x, value_y, value_z)
                xyzlines += formatted_xyz_line + "\n"
                data.append([value_atom, atom_type, value_x, value_y, value_z])
    head = str(len(data)) + "\n" + "\n"
    xyz_lines = head + xyzlines
    return xyz_lines, np.vstack(data)


# make node Graph based on connectivity matrix
def nodepdb2G(pdbfile, metal):
    xyzlines, node_data = nodepdb(pdbfile)
    G = nx.Graph()
    for n in node_data:
        G.add_node(
            n[0],
            ccoords=(np.asarray([float(n[2]),
                                 float(n[3]),
                                 float(n[4])])),
            type=n[1],
        )

    node_mol = vlx.Molecule.read_xyz_string(xyzlines)
    con_matrix = node_mol.get_connectivity_matrix()
    for i in range(len(node_data)):
        for j in range(i, len(node_data)):
            if con_matrix[i, j] == 1:
                if (nn(node_data[i][0]) == nn(node_data[j][0])
                        and nn(node_data[i][0])
                        == metal):  # skip same atom type bond metal-metal
                    continue
                # skip metal-X bond
                elif nn(node_data[i][0]) == metal and nn(
                        node_data[j][0]) == "X":
                    continue
                elif nn(node_data[j][0]) == metal and nn(
                        node_data[i][0]) == "X":
                    continue
                # skip metal-H bond
                elif nn(node_data[i][0]) == metal and nn(
                        node_data[j][0]) == "H":
                    continue
                elif nn(node_data[j][0]) == metal and nn(
                        node_data[i][0]) == "H":
                    continue
                else:
                    G.add_edge(node_data[i][0], node_data[j][0])
    return G


def create_pdb(filename, lines):
    newpdb = []
    newpdb.append("Dummy node \n" + str(filename) +
                  "   GENERATED BY MOF_builder\n")
    with open(str(filename) + ".pdb", "w") as fp:
        # Iterate over each line in the input file
        for i in range(len(lines)):
            # Split the line into individual values (assuming they are separated by spaces)
            values = lines[i]
            # Extract values based on their positions in the format string
            value1 = "ATOM"
            value2 = int(i + 1)
            value3 = values[0]  # label
            value4 = "MOL"  # residue
            value5 = 1  # residue number
            value6 = float(values[2])  # x
            value7 = float(values[3])  # y
            value8 = float(values[4])  # z
            value9 = "1.00"
            value10 = "0.00"
            value11 = values[1]  # note
            # Format the values using the specified format string
            formatted_line = "%-6s%5d%5s%4s%10d%8.3f%8.3f%8.3f%6s%6s%4s" % (
                value1,
                value2,
                value3,
                value4,
                value5,
                value6,
                value7,
                value8,
                value9,
                value10,
                value11,
            )
            newpdb.append(formatted_line + "\n")
        fp.writelines(newpdb)


def add_dummy_atoms_nodepdb(pdbfile, metal, nodeG):
    if metal in ["Zr", "Hf"]:
        metal_valence = 4
    elif metal in ["Al", "Fe", "Cr"]:
        metal_valence = 3

    dummy_pdbfile = pdbfile.removesuffix(".pdb") + "_dummy.pdb"

    # dummy_ciffile = 'test12zr_dummy.cif'
    template_metal_D = fetch_template(metal)
    sG = nodeG.copy()
    # options = {
    #            "font_size": 10,
    #            "node_size": 150,
    #            "node_color": "white",
    #            "edgecolors": "black",
    #            "linewidths": 1,
    #            "width": 1,
    #        }
    ##nx.draw_networkx(sG,**options)
    ind_max = max([int(nl(n)) for n in list(sG.nodes())])
    hydrogen_nodes = [n for n in list(sG.nodes()) if nn(n) == "H"]
    oxygen_nodes = [n for n in list(sG.nodes()) if nn(n) == "O"]
    metal_nodes = [n for n in list(sG.nodes()) if nn(n) == metal]
    print("metal_nodes", metal_nodes)

    count = ind_max + 1
    for mn in metal_nodes:
        neighbor_nodes = sG.adj[mn].copy()
        # print(neighbor_nodes, mn)
        Ocheck = all(nn(i) == "O" for i in neighbor_nodes)
        if len(neighbor_nodes) == 2 * metal_valence and Ocheck:
            # add dummy
            beginning_cc = sG.nodes[mn]["ccoords"]
            d_ccoords = []
            for nO in neighbor_nodes:
                sO = sG.nodes[nO]["ccoords"]
                cnorm_vec = (sO - beginning_cc) / np.linalg.norm(
                    sO - beginning_cc)  # 1 angstrom
                # f_vec = np.dot(np.linalg.inv(unit_cell),cnorm_vec)
                d_ccoord = beginning_cc + cnorm_vec
                # d_fcoord = beginning_fc+f_vec
                d_ccoords.append(d_ccoord)
                # d_fcoords.append(d_fcoord)
                sG.remove_edge(mn, nO)
            # order ccords based on template order
            ordered_ccoords = order_ccoords(d_ccoords, template_metal_D,
                                            beginning_cc)

            for row in range(len(d_ccoords)):
                sG.add_node("D" + str(count),
                            type="D",
                            ccoords=ordered_ccoords[row])
                sG.add_edge(mn, "D" + str(count))
                count += 1
    # nx.draw_networkx(sG,**options)
    for hn in hydrogen_nodes:
        neighbor_nodes = list(nx.neighbors(sG, hn))
        if len(neighbor_nodes) < 1:
            # find a nearest oxygen to add a connection
            min_dist = 1000
            for on in oxygen_nodes:
                dist = np.linalg.norm(sG.nodes[hn]["ccoords"] -
                                      sG.nodes[on]["ccoords"])
                if dist < min_dist:
                    min_dist = dist
                    nearest_o = on
            sG.add_edge(hn, nearest_o)

    # print(nx.number_of_nodes(sG))

    # arr = np.empty((sG.number_of_nodes(),4),dtype='O')
    # row=0
    # for n in list(sG.nodes()):
    #    arr[row,0] = n
    #    s = sG.nodes[n]['ccoords']
    #    arr[row,1] = s[0]
    #    arr[row,2] = s[1]
    #    arr[row,3] = s[2]
    #    row+=1
    #
    sG_subparts = [
        c for c in sorted(nx.connected_components(sG), key=len, reverse=True)
    ]
    # print(sG_subparts)  # debug
    head = []
    tail = []
    for sub in sG_subparts:
        l = [nn(i) for i in sub]
        if "X" not in l:
            head.append(sorted(sub))  # include D+metal
        else:
            tail.append(sorted(sub))

    sub_headlens = [len(i) for i in head]
    sub_taillens = [len(i) for i in tail]
    # sum(sub_headlens)+sum(sub_taillens),sub_headlens,sub_taillens

    dummy_count = 0
    hho_count = 0
    ho_count = 0
    o_count = 0
    ooc_count = 0
    dummy_res_len = 0

    for i in sub_headlens:
        if i == 3:
            hho_count += 1
        elif i == 2:
            ho_count += 1
        elif i == 1:
            o_count += 1
        else:
            dummy_count += 1
            dummy_res_len = i
    for j in sub_taillens:
        if j == 3:
            ooc_count += 1

    node_split_dict = {}
    node_split_dict["HHO_count"] = hho_count
    node_split_dict["HO_count"] = ho_count
    node_split_dict["O_count"] = o_count
    node_split_dict["METAL_count"] = dummy_count
    node_split_dict["dummy_res_len"] = dummy_res_len
    node_split_dict["OOC_count"] = ooc_count
    node_split_dict[
        "inres_count"] = hho_count + ho_count + o_count + dummy_count

    node_dict_name = dummy_pdbfile.removesuffix(".pdb") + "_dict"
    node_dict_path = node_dict_name
    with open(node_dict_path, "w") as f:
        for key in list(node_split_dict):
            f.write("{:20} {:<4}".format(key, node_split_dict[key]))
            f.write("\n")
    print(node_dict_path, "saved")
    subpart_nodes = head + tail
    all_lines = []
    all_bonds = []

    for count_i in range(len(subpart_nodes)):
        subnodes = subpart_nodes[count_i]
        subgraph = nx.subgraph(sG, subnodes)
        all_lines += lines_of_atoms(subgraph, sorted(subnodes))
        all_bonds += dummynode_get_bonds_from_subgraph(subgraph)

    # create_cif(all_lines,all_bonds,os.path.dirname(dummy_ciffile),os.path.basename(dummy_ciffile)) #if want dummy node cif file
    create_pdb(dummy_pdbfile.removesuffix(".pdb"), all_lines)
    print(dummy_pdbfile.removesuffix(".pdb") + ".pdb", "saved")
    return all_lines, dummy_pdbfile
