import numpy as np
import re



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
