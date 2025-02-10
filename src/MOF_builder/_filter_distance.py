import numpy as np

def filter_neighbor_atoms_by_dist(
    neighbor_atoms_number, atom1_array, atom2_array, distance_threshhold
):
    node_list = np.empty((0, neighbor_atoms_number))
    for i in range(atom1_array.shape[0]):
        distance_metal = []
        distance_metal_sort = []
        for j in range(atom2_array.shape[0]):
            distance = np.linalg.norm(atom1_array[i, :] - atom2_array[j, :])
            distance_metal.append(distance)
            distance_metal_sort.append(distance)
        distance_metal_sort.sort()
        threshhold = distance_threshhold
        indices = [
            index for index, num in enumerate(distance_metal) if num < threshhold
        ]
        #print(indices,distance_metal_sort)
        #print(f"distancesort{distance_metal_sort[28:35]}")

        if (
            len(indices) == neighbor_atoms_number
        ):  # TODO: target node cluster with defined metal number
            node_list = np.vstack((node_list, indices))
    return node_list

def O_filter_neighbor_atoms_by_dist(
    neighbor_atoms_number, atom1_array, atom2_array, distance_threshhold
):
    node_list = np.empty((0, neighbor_atoms_number))
    for i in range(atom1_array.shape[0]):
        distance_metal = []
        distance_metal_sort = []
        for j in range(atom2_array.shape[0]):
            distance = np.linalg.norm(atom1_array[i, :] - atom2_array[j, :])
            distance_metal.append(distance)
            distance_metal_sort.append(distance)
        distance_metal_sort.sort()
        threshhold = distance_threshhold
        indices = [
            index for index, num in enumerate(distance_metal) if num < threshhold
        ]
        #print(indices,distance_metal_sort)
        #print(f"distancesort{distance_metal_sort[31:33]}")
        

        if (
            len(indices) == neighbor_atoms_number
        ):  # TODO: target node cluster with defined metal number
            node_list = np.vstack((node_list, indices))

        print(f"\n\nthreshhold {threshhold}")
    return node_list


def find_neighbor_atoms_at_dist(atom1, atom2_array, distance_threshhold):
    distance_neighbor = []
    for i in range(atom2_array.shape[0]):
        distance = np.linalg.norm(atom1 - atom2_array[i, :])
        distance_neighbor.append(distance)

    indices = [
        index
        for index, num in enumerate(distance_neighbor)
        if num == distance_threshhold
    ]
    # print(indices,distance_metal_sort)
    neighbor_atoms = atom2_array[indices]
    return neighbor_atoms