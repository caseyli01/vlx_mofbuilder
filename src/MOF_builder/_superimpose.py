import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
import itertools
from numpy.linalg import norm


def sort_by_distance(arr):
    # Calculate distances from the mass center to all other elements
    # com = np.mean(arr, axis=0)
    distances = [(np.linalg.norm(arr[i] - arr[0]), i) for i in range(len(arr))]
    # Sort distances in ascending order
    distances.sort(key=lambda x: x[0])
    return distances


def match_vectors(arr1, arr2, num):
    # Get sorted distances
    sorted_distances_arr1 = sort_by_distance(arr1)
    sorted_distances_arr2 = sort_by_distance(arr2)

    # Select the indices by distance matching in limited number

    indices_arr1 = [sorted_distances_arr1[j][1] for j in range(num)]
    indices_arr2 = [sorted_distances_arr2[j][1] for j in range(num)]

    # reorder the matching vectors# which can induce the smallest RMSD
    closest_vectors_arr1 = np.array([arr1[i] for i in indices_arr1])
    closest_vectors_arr2 = np.array([arr2[i] for i in indices_arr2])

    return closest_vectors_arr1, closest_vectors_arr2


def superimpose(arr1, arr2, min_rmsd=1e6):
    sup = SVDSuperimposer()
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    best_rot, best_tran = np.eye(3), np.zeros(3)

    if len(arr1) < 7:
        for perm in itertools.permutations(arr1):
            perm = np.asarray(perm)
            sup.set(arr2, perm)
            sup.run()
            rmsd = sup.get_rms()
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                best_rot, best_tran = sup.get_rotran()

    else:
        arr1, arr2 = match_vectors(arr1, arr2, 6)
        for perm in itertools.permutations(arr1):
            perm = np.asarray(perm)
            sup.set(arr2, perm)
            sup.run()
            rmsd = sup.get_rms()
            if rmsd < min_rmsd:
                min_rmsd = rmsd
                best_rot, best_tran = sup.get_rotran()

    return min_rmsd, best_rot, best_tran


def superimpose_rotateonly(arr1, arr2, min_rmsd=1e10):
    sup = SVDSuperimposer()
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    best_rot, best_tran = np.eye(3), np.zeros(3)
    if len(arr1) == len(arr2):
        if len(arr1) < 7:
            for perm in itertools.permutations(arr1):
                perm = np.asarray(perm)
                sup.set(arr2, perm)
                sup.run()

                rmsd = sup.get_rms()
                if rmsd < min_rmsd:
                    min_rmsd = rmsd
                    best_rot, best_tran = sup.get_rotran()
                    if np.allclose(np.dot(sup.get_rotran()[1], np.zeros(3)), 1e-1):
                        min_rmsd = sup.get_rms()
                        best_rot, best_tran = sup.get_rotran()
            return min_rmsd, best_rot, best_tran

    arr1, arr2 = match_vectors(arr1, arr2, min(6, len(arr1), len(arr2)))
    for perm in itertools.permutations(arr1):
        perm = np.asarray(perm)
        sup.set(arr2, perm)
        sup.run()
        rmsd = sup.get_rms()
        rmsd = sup.get_rms()
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_rot, best_tran = sup.get_rotran()
            if np.allclose(np.dot(sup.get_rotran()[1], np.zeros(3)), 1e-1):
                min_rmsd = rmsd
                best_rot, best_tran = sup.get_rotran()

    return min_rmsd, best_rot, best_tran
