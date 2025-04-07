import numpy as np


def make_supercell333(array_xyz):
    array_x1 = array_xyz + np.array([1, 0, 0])
    array_x2 = array_xyz + np.array([-1, 0, 0])
    array_y1 = array_xyz + np.array([0, 1, 0])
    array_y2 = array_xyz + np.array([0, -1, 0])
    array_x1_y1 = array_xyz + np.array([1, 1, 0])
    array_x1_y2 = array_xyz + np.array([1, -1, 0])
    array_x2_y1 = array_xyz + np.array([-1, 1, 0])
    array_x2_y2 = array_xyz + np.array([-1, -1, 0])
    array_3x3y1z_layer = np.vstack(
        (
            array_xyz,
            array_x1,
            array_x2,
            array_y1,
            array_y2,
            array_x1_y1,
            array_x1_y2,
            array_x2_y1,
            array_x2_y2,
        )
    )
    array_3x3y2z_layer = array_3x3y1z_layer + np.array([0, 0, 1])
    array_3x3y3z_layer = array_3x3y1z_layer + np.array([0, 0, -1])

    array_supercell333 = np.vstack(
        (array_3x3y1z_layer, array_3x3y2z_layer, array_3x3y3z_layer)
    )

    return array_supercell333


##Frame
def Carte_points_generator(xyz_num):
    x_num, y_num, z_num = xyz_num
    """this function is to generate a group of 3d points(unit=1) defined by user for further grouping points"""
    points = []
    for i in range(0, x_num + 1):
        for j in range(0, y_num + 1):
            for k in range(0, z_num + 1):
                points.append([i, j, k])
    points = np.array(points)
    return points
