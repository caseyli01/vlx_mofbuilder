import numpy as np
def _make_supercell222(array_xyz):
    array_x = array_xyz + np.array([1, 0, 0])
    array_y = array_xyz + np.array([0, 1, 0])
    array_x_y = array_xyz + np.array([1, 1, 0])
    array_z = array_xyz + np.array([0, 0, 1])
    array_x_z = array_xyz + np.array([1, 0, 1])
    array_y_z = array_xyz + np.array([0, 1, 1])
    array_x_y_z = array_xyz + np.array([1, 1, 1])

    array_supercell222 = np.vstack(
        (
            array_xyz,
            array_x,
            array_y,
            array_z,
            array_x_y,
            array_x_z,
            array_y_z,
            array_x_y_z,
        )
    )

    return array_supercell222

def make_flex_supercell222(array_xyz):
    def sign(x):
        if np.sign(x)==0:
            return 1
        else:
            return -1*np.sign(x)
    x,y,z = np.mean(array_xyz,axis=0)
    x_sign = sign(x-0.5)
    y_sign = sign(y-0.5)
    z_sign = sign(z-0.5)
    #print(f"sign,xyz{x_sign,y_sign,z_sign,x,y,z}")
    array_x = array_xyz + np.array([x_sign, 0, 0])
    array_y = array_xyz + np.array([0, y_sign, 0])
    array_x_y = array_xyz + np.array([x_sign, y_sign, 0])
    array_z = array_xyz + np.array([0, 0, z_sign])
    array_x_z = array_xyz + np.array([x_sign, 0, z_sign])
    array_y_z = array_xyz + np.array([0, y_sign, z_sign])
    array_x_y_z = array_xyz + np.array([x_sign, y_sign, z_sign])

    array_supercell222 = np.vstack(
        (
            array_xyz,
            array_x,
            array_y,
            array_z,
            array_x_y,
            array_x_z,
            array_y_z,
            array_x_y_z,
        )
    )

    return array_supercell222


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

def _make_flex_supercell333(array_xyz):
    #def sign(x):
    #    if np.sign(x)==0:
    #        return 1
    #    else:
    #        return -1*np.sign(x)
    #x,y,z = np.mean(array_xyz,axis=0)
    #x_sign = sign(x-0.5)
    #y_sign = sign(y-0.5)
    #z_sign = sign(z-0.5)
    ##print(f"sign,xyz{x_sign,y_sign,z_sign,x,y,z}")
    #array_x1 = array_xyz + np.array([x_sign, 0, 0])
    #array_x2 = array_xyz + np.array([x_sign, 0, 0])
    #array_y = array_xyz + np.array([0, y_sign, 0])
    #array_x_y = array_xyz + np.array([x_sign, y_sign, 0])
    #array_z = array_xyz + np.array([0, 0, z_sign])
    #array_x_z = array_xyz + np.array([x_sign, 0, z_sign])
    #array_y_z = array_xyz + np.array([0, y_sign, z_sign])
    #array_x_y_z = array_xyz + np.array([x_sign, y_sign, z_sign])
#
    #array_supercell222 = np.vstack(
    #    (
    #        array_xyz,
    #        array_x,
    #        array_y,
    #        array_z,
    #        array_x_y,
    #        array_x_z,
    #        array_y_z,
    #        array_x_y_z,
    #    )
    #)



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


class __Frame:
    def __init__(self, x_num, y_num, z_num, tric_basis):
        self.x_num, self.y_num, self.z_num = x_num, y_num, z_num
        self.tric_basis = tric_basis

    def Carte_points_generator(x_num, y_num, z_num):
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

    def get_frame_supercell_points(self):
        supercell_points = __Frame.Carte_points_generator(
            self.x_num,
            self.y_num,
            self.z_num,
        )
        self.supercell_tric_points = np.round(np.dot(supercell_points, self.tric_basis),4)
        return self.supercell_tric_points