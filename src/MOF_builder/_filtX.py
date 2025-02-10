import numpy as np
import re

def filt_nodex_fvec(array):
    nodex_fvec=np.asarray([i for i in array if i[4]=='NODE' and re.sub('[0-9]','',i[2]) == 'X'])
    return nodex_fvec


def check_overlapX2(edgex_cvec,nodex_cvec):
    dist_arr=edgex_cvec-nodex_cvec
    for i in dist_arr:
        if np.linalg.norm(i) <2:
            return True
    return False


def filt_outside_edgex(Xs_fc,edge_center_fc,linker_topics):
    lcs_list = []
    lcs = []
    for i in range(len(Xs_fc)):
        lc = np.linalg.norm(Xs_fc[i]-edge_center_fc)
        lcs_list.append((i,lc))
        lcs.append(lc)
    lcs.sort(reverse=True)
    if len(lcs)>linker_topics:
        outside_edgex_indices=[i[0] for i in lcs_list if i[1]>lcs[linker_topics]]
        outside_edgex_ind_dist=[i for i in lcs_list if i[1]>lcs[linker_topics]]
        return outside_edgex_indices,outside_edgex_ind_dist
    else:
        outside_edgex_indices=[i[0] for i in lcs_list ]
        outside_edgex_ind_dist=[i for i in lcs_list ]
        return outside_edgex_indices,outside_edgex_ind_dist


def filt_close_edgex(Xs_fc,edge_center_fc,linker_topics):
    lcs_list = []
    lcs = []
    for i in range(len(Xs_fc)):
        lc = np.linalg.norm(Xs_fc[i]-edge_center_fc)
        lcs_list.append((i,lc))
        lcs.append(lc)
    lcs.sort()
    outside_edgex_indices=[i[0] for i in lcs_list if i[1]<lcs[linker_topics]]
    outside_edgex_ind_dist=[i for i in lcs_list if i[1]<lcs[linker_topics]]
    return outside_edgex_indices,outside_edgex_ind_dist

def get_rad_v1v2(v1,v2):
    cos_theta = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if cos_theta ==0:
        return 0
    else:
        rad = np.arccos(cos_theta)
        return rad

def filt_closest_x_angle(Xs_fc,edge_center_fc,node_center_fc):
    rds_list = []
    rads = []
    dists = []
    x_number = len(Xs_fc)
    half_x_number = int(0.5*x_number)
    for i in range(x_number):
        rad = get_rad_v1v2(Xs_fc[i]-edge_center_fc,node_center_fc-edge_center_fc)
        dist = np.linalg.norm(Xs_fc[i]-edge_center_fc)
        rds_list.append((i,rad,dist))
        rads.append(rad)
        dists.append(dist)
    rads.sort()
    dists.sort()
    x_idx=[i[0] for i in rds_list if (i[1]<rads[2] and i[2]<dists[half_x_number])]
    x_info=[i for i in rds_list if (i[1]<rads[2] and i[2]<dists[half_x_number])]
    if len(x_idx)==1:
        return x_idx,x_info
    elif len(x_idx)>1:
        min_d = min([j[2] for j in x_info])
        x_idx1=[i[0] for i in rds_list if  i[2]==min_d]
        x_info1=[i for i in rds_list if i[2]==min_d]
        return x_idx1,x_info1
    else:
        print("ERROR cannot find connected X")
        print(rds_list)