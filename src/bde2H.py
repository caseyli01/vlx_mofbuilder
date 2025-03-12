import veloxchem as vlx
import time
import numpy as np


def get_equiv(mol):
    id = vlx.AtomTypeIdentifier()
    id.ostream.mute()
    atom_type = id.generate_gaff_atomtypes(mol)
    id.identify_equivalences()
    equivalent_charges = id.equivalent_charges
    return atom_type, equivalent_charges


def analyze_equiv(molecule):
    atom_type, equiv = get_equiv(molecule)
    equiv_atoms_groups = []
    # Split the string by comma
    substrings = equiv.split(",")
    # Split each substring by "="
    for substr in substrings:
        unit = substr.split("=")
        equiv_atoms_groups.append(unit)
    # map str to int
    equiv_atoms_groups = [list(map(int, x)) for x in equiv_atoms_groups]
    for i in range(
        len(equiv_atoms_groups)
    ):  # this is to make the atom index start from 0
        equiv_atoms_groups[i] = [j - 1 for j in equiv_atoms_groups[i]]
    return atom_type, equiv_atoms_groups


def search_in_equiv_atoms_groups(equiv_atoms_groups, atom_idx):
    for group in equiv_atoms_groups:
        if atom_idx in group:
            return group
    return []


def add_H_connected_atom_info(labels, atoms_types, atom_idx, connetivity_matrix):
    # if the atom is not H, then return ''
    if labels[atom_idx] != "H":
        return ""
    else:
        # search it in connectivity matrix to get the connected atom index
        con_info = connetivity_matrix[atom_idx]
        # if the atom is H, then it should have only one connected atom, which value should be 1
        connected_atom_idx = np.where(con_info == 1)[0]
        if len(connected_atom_idx) != 1:
            print("!!!: H atom should have only one connected atom")
            return ""
        connected_atom = labels[connected_atom_idx[0]]
        connected_atom_type = atoms_types[connected_atom_idx[0]]
        return (
            str(connected_atom)
            + "_"
            + str(connected_atom_idx[0])
            + "_"
            + str(connected_atom_type)
        )


def atoms_analyzer(molecule, all=False, target_atom="H", show=False):
    if show:
        molecule.show(atom_indices=True)
    atom_type, equiv_atoms_groups = analyze_equiv(molecule)
    con_matrix = molecule.get_connectivity_matrix()
    labels = molecule.get_labels()
    atom_info_dict = {}
    for i in range(len(labels)):
        atom_info_dict[labels[i] + "_" + str(i)] = {
            "atom_type": atom_type[i],
            "equiv_group": search_in_equiv_atoms_groups(equiv_atoms_groups, i),
            "H_connected_atom": add_H_connected_atom_info(
                labels, atom_type, i, con_matrix
            ),
        }
    if all:
        return atom_info_dict
    else:
        target_atom_info = {}
        for key in atom_info_dict.keys():
            if target_atom == key.split("_")[0]:
                target_atom_info[key] = atom_info_dict[key]
        return target_atom_info


def fetch_unique_H(
    hydrogen_atoms_dict, use_equiv=False, only_sp3_carbon_hydrogen=False
):
    if not use_equiv:
        for key in hydrogen_atoms_dict.keys():
            hydrogen_atoms_dict[key]["equiv_group"] = []

    hydrogen_record = []
    unique_hydrogen_keys = []
    for key in hydrogen_atoms_dict.keys():
        if key.split("_")[0] == "H" and int(key.split("_")[1]) not in hydrogen_record:
            hydrogen_record.extend(hydrogen_atoms_dict[key]["equiv_group"])
            unique_hydrogen_keys.append(key)
    unique_hydrogen_indices = [int(x.split("_")[1]) for x in unique_hydrogen_keys]
    if not only_sp3_carbon_hydrogen:
        return unique_hydrogen_keys, unique_hydrogen_indices
    else:
        sp3_carbon_unique_hydrogen_indices = []
        sp3_carbon_unique_hydrogen_keys = []
        for key in unique_hydrogen_keys:
            if hydrogen_atoms_dict[key]["H_connected_atom"] != "":
                connected_atom = hydrogen_atoms_dict[key]["H_connected_atom"].split(
                    "_"
                )[0]
                connected_atom_type = hydrogen_atoms_dict[key][
                    "H_connected_atom"
                ].split("_")[2]
                if connected_atom == "C" and connected_atom_type == "c3":
                    sp3_carbon_unique_hydrogen_indices.append(int(key.split("_")[1]))
                    sp3_carbon_unique_hydrogen_keys.append(key)
        return sp3_carbon_unique_hydrogen_keys, sp3_carbon_unique_hydrogen_indices


def update_equiv_hydrogen_dissociationenergy(
    unique_hydrogen_dissociation_energies, unique_hydrogen_keys, hydrogen_atoms_dict
):
    au2kcal = 627.509
    au2kj = 2625.5
    for i in range(len(unique_hydrogen_keys)):
        key = unique_hydrogen_keys[i]
        energy_au = unique_hydrogen_dissociation_energies[i]
        equiv_group = hydrogen_atoms_dict[key]["equiv_group"]
        for j in equiv_group:
            hydrogen_atoms_dict["H_" + str(j)]["dissociation_energy_au"] = energy_au
            hydrogen_atoms_dict["H_" + str(j)]["dissociation_energy_kcal"] = (
                energy_au * au2kcal
            )
            hydrogen_atoms_dict["H_" + str(j)]["dissociation_energy_kj"] = (
                energy_au * au2kj
            )
    return hydrogen_atoms_dict


def print_hydrogen_dissociation_energy(hydrogen_atoms_dict, unit="kcal"):
    for key in hydrogen_atoms_dict.keys():
        if "dissociation_energy_au" in hydrogen_atoms_dict[key].keys():
            if unit == "kcal":
                print(
                    key,
                    round(hydrogen_atoms_dict[key]["dissociation_energy_kcal"], 1),
                    "kcal/mol",
                )
            elif unit == "kj":
                print(
                    key,
                    round(
                        hydrogen_atoms_dict[key]["dissociation_energy_kj"], 1, "kj/mol"
                    ),
                )
            elif unit == "au":
                print(key, hydrogen_atoms_dict[key]["dissociation_energy_au"])


def compute_whole_mol_scf_energy(
    molecule, functional, functional2, basis_set1, basis_set2
):
    print("Optimizing geometry of the molecule before removing hydrogens")

    basis = vlx.MolecularBasis.read(molecule, basis_set1, ostream=None)
    scf_drv = vlx.ScfRestrictedDriver()
    scf_drv.ostream.mute()
    scf_drv.xcfun = functional
    scf_drv.ri_coulomb = True
    scf_drv.diis_thresh = 100
    scf_drv.conv_thresh = 1.0e-4
    scf_drv.grid_level = 2
    scf_drv.level_shifting = 1.0
    scf_results = scf_drv.compute(molecule, basis)
    opt_drv = vlx.OptimizationDriver(scf_drv)
    scf_drv.level_shifting = 1.0
    opt_drv.conv_energy = 1e-04
    opt_drv.conv_drms = 1e-02
    opt_drv.conv_dmax = 2e-02
    opt_drv.conv_grms = 4e-03
    opt_drv.conv_gmax = 8e-03
    opt_results = opt_drv.compute(molecule, basis, scf_results)
    molecule = vlx.Molecule.read_xyz_string(opt_results["final_geometry"])
    basis = vlx.MolecularBasis.read(molecule, basis_set2)
    scf_drv = vlx.ScfRestrictedDriver()
    scf_drv.xcfun = functional2
    scf_drv.ri_coulomb = True
    scf_drv.diis_thresh = 100
    scf_drv.ostream.mute()
    scf_drv.conv_thresh = 1.0e-4
    scf_drv.grid_level = 2
    scf_drv.level_shifting = 1.0
    scf_results_big = scf_drv.compute(molecule, basis)
    return scf_results_big["scf_energy"]


def compute_hydrogen_radical_scf_energy(basis_set2, functional2):
    hydrogen = vlx.Molecule.read_str(""" H 0.0 0.0 0.0 """)
    hydrogen.set_multiplicity(2)
    basis = vlx.MolecularBasis.read(hydrogen, basis_set2)
    scf_drv = vlx.ScfUnrestrictedDriver()
    scf_drv.xcfun = functional2
    scf_drv.ostream.mute()
    scf_resultsH = scf_drv.compute(hydrogen, basis)
    return scf_resultsH["scf_energy"]


def remove_atom_by_idx(mol, atom_indices_to_remove):
    mol_string = mol.get_xyz_string()
    number_of_atoms = mol.number_of_atoms()
    mol_stringlist = mol_string.split("\n")
    # Identify the lines that start with atom and save the positions
    molecules = []
    for idx in atom_indices_to_remove:
        new_mol = mol_stringlist.copy()
        # remove the index+2 line, the first line is the number of atoms, the second line is the comment
        new_mol.pop(idx + 2)
        # Update the number of atoms
        new_mol[0] = str(number_of_atoms - 1)
        new_mol = "\n".join(new_mol)
        molecules.append(vlx.Molecule.read_xyz_string(new_mol))
    return molecules


def compute_mol_rad_scf_energy(mol, functional, functional2, basis_set2):
    step_start = time.time()
    basis_rad = vlx.MolecularBasis.read(mol, "def2-svp")
    mol.set_multiplicity(2)
    scf_drv_rad = vlx.ScfUnrestrictedDriver()
    scf_drv_rad.xcfun = functional
    scf_drv_rad.ri_coulomb = True
    scf_drv_rad.grid_level = 2
    scf_drv_rad.conv_thresh = 1.0e-3
    scf_drv_rad.level_shifting = 1.0
    scf_drv_rad.max_iter = 200
    scf_drv_rad.ostream.mute()
    scf_resultsmol = scf_drv_rad.compute(mol, basis_rad)
    opt_drv_rad = vlx.OptimizationDriver(scf_drv_rad)
    scf_drv_rad.level_shifting = 1.0
    opt_drv_rad.conv_energy = 1e-04
    opt_drv_rad.conv_drms = 1e-02
    opt_drv_rad.conv_dmax = 2e-02
    opt_drv_rad.conv_grms = 4e-03
    opt_drv_rad.conv_gmax = 8e-03
    opt_results_rad = opt_drv_rad.compute(mol, basis_rad, scf_resultsmol)
    mol = vlx.Molecule.read_xyz_string(opt_results_rad["final_geometry"])
    mol.set_multiplicity(2)
    basis = vlx.MolecularBasis.read(mol, basis_set2)
    scf_drv_rad = vlx.ScfUnrestrictedDriver()
    scf_drv_rad.xcfun = functional2
    scf_drv_rad.ri_coulomb = True
    scf_drv_rad.diis_thresh = 100
    scf_drv_rad.ostream.mute()
    scf_drv_rad.conv_thresh = 1.0e-4
    scf_drv_rad.grid_level = 2
    scf_drv_rad.max_iter = 200
    scf_drv_rad.level_shifting = 1.0
    scf_drv_rad.level_shifting_delta = 0.1
    scf_results_rad_big = scf_drv_rad.compute(mol, basis)
    step_end = time.time()
    print("-" * 50)
    # print('Computing energy of structure ' + str(i) + 'Done')
    print("time cost : " + str(time.time() - step_start))
    return scf_results_rad_big["scf_energy"]


# for testing
if __name__ == "__main__":
    molecule = vlx.Molecule.read_smiles("C")
    hydrogen_atoms_dict = atoms_analyzer(
        molecule, all=True, target_atom="H", show=False
    )
    unique_hydrogen_keys, unique_hydrogen_indices = fetch_unique_H(
        hydrogen_atoms_dict, use_equiv=True, only_sp3_carbon_hydrogen=True
    )

    basis_set1 = "def2-svp"
    basis_set2 = "def2-tzvp"
    functional = "PBE"
    functional2 = "PBE"
    whole_mol_scf_energy = compute_whole_mol_scf_energy(
        molecule, functional, functional2, basis_set1, basis_set2
    )
    hydrogen_rad_scf_energy = compute_hydrogen_radical_scf_energy(
        basis_set2, functional2
    )
    molecules = remove_atom_by_idx(molecule, unique_hydrogen_indices)
    unique_BDEs_au = []
    count = 1
    for mol_rad in molecules:
        print("Computing energy of structure :", count, " of ", len(molecules))
        mol_rad_scf_energy = compute_mol_rad_scf_energy(
            mol_rad, functional, functional2, basis_set2
        )
        bde_au = mol_rad_scf_energy - whole_mol_scf_energy + hydrogen_rad_scf_energy
        unique_BDEs_au.append(bde_au)
        count += 1

    # loop the unique_hydrogen_indices to remove the H atoms from the molecule and calulate the dissciation energy but save the energy for all equivalent H atoms
    # print the dissociation energy for each H atom
    hydrogen_atoms_dict = update_equiv_hydrogen_dissociationenergy(
        unique_BDEs_au, unique_hydrogen_keys, hydrogen_atoms_dict
    )
    print(len(unique_hydrogen_keys), "iterations")
    print_hydrogen_dissociation_energy(hydrogen_atoms_dict, unit="kcal")
