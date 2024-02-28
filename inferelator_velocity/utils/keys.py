OBS_TIME_KEY = "program_{prog}_time"
OBSM_PCA_KEY = "program_{prog}_pca"

OBSP_DIST_KEY = "program_{prog}_distances"
UNS_GRAPH_SUBKEY = "program_{prog}_graph"

PROGRAM_KEY = "programs"

N_BINS = 10

NOISE2SELF_KEY = 'noise2self'
NOISE2SELF_DIST_KEY = 'noise2self_distance_graph'
NOISE2SELF_DENOISED_KEY = 'noise2self_denoised'

METRIC_SUBKEY = 'metric'
METRIC_GENE_SUBKEY = 'metric_genes'
METRIC_DIST_SUBKEY = '{metric}_distance'
MCV_LOSS_SUBKEY = 'molecular_cv_loss'
LEIDEN_CORR_SUBKEY = 'leiden_correlation'
PROGRAM_CLUST_SUBKEY = 'cluster_program_map'
N_COMP_SUBKEY = 'n_comps'
N_PROG_SUBKEY = 'n_programs'
PROG_NAMES_SUBKEY = 'program_names'

CENTROID_SUBKEY = 'centroids'
SHORTEST_PATH_SUBKEY = 'shortest_path'
CLOSEST_ASSIGNMENT_SUBKEY = 'closest_path_assignment'
ASSIGNMENT_NAME_SUBKEY = 'assignment_names'
ASSIGNMENT_CENTROID_SUBKEY = 'assignment_centroids'
ASSIGNMENT_PATH_SUBKEY = 'assignment_path'

CLUSTER_ORDER_SUBKEY = 'cluster_order'
CLUSTER_TIME_SUBKEY = 'cluster_times'

OBS_TIME_KEY_KEY = 'obs_time_key'
OBS_GROUP_KEY_KEY = 'obs_group_key'
OBSM_KEY_KEY = 'obsm_key'


def get_program_ids(adata):
    """
    Get program IDs from processed data object

    :param adata: _description_
    :type adata: _type_
    :return: _description_
    :rtype: _type_
    """

    if PROGRAM_KEY not in adata.uns:
        raise RuntimeError(
            "Unable to find programs data; "
            "run ifv.program_select() on this object"
        )

    return [
        p
        for p in adata.uns[PROGRAM_KEY][PROG_NAMES_SUBKEY]
        if p != '-1'
    ]
