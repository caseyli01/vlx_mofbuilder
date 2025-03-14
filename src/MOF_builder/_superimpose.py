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


def _superimpose(arr1, arr2, min_rmsd=1e6):
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


def _superimpose_rotateonly(arr1, arr2, min_rmsd=1e6):
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
                        break
            return min_rmsd, best_rot, best_tran

    arr1, arr2 = match_vectors(arr1, arr2, min(6, len(arr1), len(arr2)))
    for perm in itertools.permutations(arr1):
        perm = np.asarray(perm)
        sup.set(arr2, perm)
        sup.run()
        rmsd = sup.get_rms()
        if rmsd < min_rmsd:
            min_rmsd = rmsd
            best_rot, best_tran = sup.get_rotran()
            if np.allclose(np.dot(sup.get_rotran()[1], np.zeros(3)), 1e-1):
                break

    return min_rmsd, best_rot, best_tran


def superimpose(arr1, arr2, min_rmsd=1e6):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    best_rot, best_tran = np.eye(3), np.zeros(3)
    m_arr1, m_arr2 = match_vectors(arr1, arr2, min(6, len(arr1), len(arr2)))

    for perm in itertools.permutations(m_arr1):
        rmsd, _, _ = svd_superimpose(np.asarray(perm), m_arr2)
        if rmsd < min_rmsd:
            min_rmsd, best_rot, best_tran = svd_superimpose(np.asarray(perm), m_arr2)

    return min_rmsd, best_rot, best_tran


def svd_superimpose(coords, reference_coords):
    """
    Superimpose two sets of 3D points using Singular Value Decomposition (SVD).

    Parameters:
        reference_coords (numpy.ndarray): Nx3 array representing the fixed reference coordinates.
        coords (numpy.ndarray): Nx3 array representing the coordinates to be aligned.

    Returns:
        rmsd (float): Root Mean Square Deviation after superimposition.
        rot (numpy.ndarray): The optimal 3x3 rotation matrix.
        trans (numpy.ndarray): The optimal 1x3 translation vector.
    """
    if reference_coords.shape != coords.shape:
        raise ValueError("Both input arrays must have the same shape (Nx3).")

    # Compute centroids
    centroid_ref = np.mean(reference_coords, axis=0)
    centroid_coords = np.mean(coords, axis=0)

    # Center the points
    ref_centered = reference_coords - centroid_ref
    coords_centered = coords - centroid_coords

    # Compute the correlation matrix
    H = np.dot(coords_centered.T, ref_centered)

    # Compute SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    rot = np.dot(Vt.T, U.T).T

    # Ensure a proper rotation (prevent reflection)
    if np.linalg.det(rot) < 0:
        Vt[2] *= -1
        rot = np.dot(Vt.T, U.T).T

    # Compute translation vector
    trans = centroid_ref - np.dot(centroid_coords, rot)

    # Apply transformation
    transformed_coords = np.dot(coords, rot) + trans

    # Compute RMSD
    diff = transformed_coords - reference_coords
    rmsd = np.sqrt(np.sum(np.sum(diff**2)) / reference_coords.shape[0])

    return rmsd, rot, trans


def superimpose_rotateonly(arr1, arr2, min_rmsd=1e6):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    best_rot, best_tran = np.eye(3), np.zeros(3)

    m_arr1, m_arr2 = match_vectors(arr1, arr2, min(6, len(arr1), len(arr2)))
    for perm in itertools.permutations(m_arr1):
        rmsd, _, _ = svd_superimpose(np.asarray(perm), m_arr2)
        if rmsd < min_rmsd:
            min_rmsd, best_rot, best_tran = svd_superimpose(np.asarray(perm), m_arr2)
            if np.allclose(np.dot(best_tran, np.zeros(3)), 1e-1):
                min_rmsd, best_rot, best_tran = svd_superimpose(
                    np.asarray(perm), m_arr2
                )
                break

    return min_rmsd, best_rot, best_tran


if __name__ == "__main__":
    # test
    x = np.array(
        [
            [51.65, -1.90, 50.07],
            [50.40, -1.23, 50.65],
            [50.68, -0.04, 51.54],
            [50.22, -0.02, 52.85],
        ],
        "f",
    )

    y = np.array(
        [
            [51.30, -2.99, 46.54],
            [51.09, -1.88, 47.58],
            [52.36, -1.20, 48.03],
            [52.71, -1.18, 49.38],
        ],
        "f",
    )

    a, b, c = superimpose_rotateonly(x, y)
    d, e, f = _superimpose(x, y)
    print(a, b, c)
    print(d, e, f)
    comx = np.mean(x, axis=0)
    comy = np.mean(y, axis=0)
    print(comx - comy)
