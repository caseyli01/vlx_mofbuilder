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
        (
            array_3x3y1z_layer,
            array_3x3y2z_layer,
            array_3x3y3z_layer
        )
    )

    return array_supercell333

##Frame
def Carte_points_generator(xyz_num):
        x_num, y_num, z_num = xyz_num
        """this function is to generate a group of 3d points(unit=1) defined by user for further grouping points"""
        unit_dx, unit_dy, unit_dz = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # add x layer
        points = np.array([0, 0, 0])
        for i in range(0, x_num + 1):
            points = np.vstack((points, i * unit_dx))
        # add y layer
        points_x = points
        for i in range(0, y_num + 1):
            points = np.vstack((points, points_x + i * unit_dy))
        # add z layer
        points_xy = points
        for i in range(0, z_num + 1):
            points = np.vstack((points, points_xy + i * unit_dz))
        points = np.unique(points, axis=0)
        return points
