# inferelator-velocity

[![PyPI version](https://badge.fury.io/py/inferelator-velocity.svg)](https://badge.fury.io/py/inferelator-velocity)
[![CI](https://github.com/flatironinstitute/inferelator-velocity/actions/workflows/python-package.yml/badge.svg)](https://github.com/flatironinstitute/inferelator-velocity/actions/workflows/python-package.yml/)
[![codecov](https://codecov.io/gh/flatironinstitute/inferelator-velocity/branch/main/graph/badge.svg)](https://codecov.io/gh/flatironinstitute/inferelator-velocity)

This is a package that calculates dynamic (time-dependent) latent parameters from 
single-cell expression data and associated experimental metadata or bulk RNA-seq data.
It is designed to create data that is compatible with the 
[inferelator](https://github.com/flatironinstitute/inferelator) or 
[supirfactor-dynamical](https://github.com/GreshamLab/supirfactor-dynamical) packages.

### Installation

Install this package using the standard python package manager `python -m pip install inferelator_velocity`.
It depends on standard python scientific computing packages (e.g. scipy, numpy, scikit-learn, pandas),
and on the AnnData data container package.

If you intend to use large sparse matrices (as is common for single-cell data), it is advisable to install
the intel math kernel library (e.g. with `conda install mkl`) and the python `sparse_dot_mkl` package with
`python -m pip install sparse_dot_mkl` to accelerate sparse matrix operations.

### Usage

#### Assigning genes to new time-dependent transcriptional programs

Load single-cell data into an [https://anndata.readthedocs.io/en/latest/](AnnData) object.
Call `program_select` on the raw, unprocessed integer count data, setting `n_programs` to
the expected number of distinct time-dependent transcriptional programs.

```
import anndata as ad
from inferelator_velocity import program_select

adata = ad.read(FILE_NAME)

program_select(
    adata,          # Anndata object
    layer='counts', # Layer with unprocessed integer count data
    n_programs=2,   # Number of transcriptional programs expected
    verbose=True    # Print additional status messages
)
```

This function will return the same anndata object with new attributes:

```
.var['leiden']: Leiden cluster ID
.var['programs']: Program ID
.uns['programs']: {
    'metric': Metric name,
    'leiden_correlation': Absolute value of spearman rho
        between PC1 of each leiden cluster,
    'metric_genes': Gene labels for distance matrix
    '{metric}_distance': Distance matrix for {metric},
    'cluster_program_map': Dict mapping gene clusters to gene programs,
    'program_PCs_variance_ratio': Variance explained by program PCs,
    'n_comps': Number of PCs selected by molecular crossvalidation,
    'molecular_cv_loss': Loss values for molecular crossvalidation
}
```

#### Assigining genes to existing time-dependent transcriptional programs

Call `assign_genes_to_programs` on an anndata object which `program_select` has already
been run on. This will assign any transcripts to the existing programs based on
mutual information. It is advisable to pass `default_program`, identifying the
transcriptional program to assign transcripts that have low mutual information with
all identified programs (these transcripts are often noise-driven and are best assigned
to whichever program best represents experimental wall clock time).

```
import anndata as ad
from inferelator_velocity import assign_genes_to_programs

adata = ad.read(FILE_NAME)

adata.var['programs'] = assign_genes_to_programs(
    adata,                      # Anndata object
    layer='counts',             # Layer with unprocessed integer count data
    default_program='0',        # 'Default' transcriptional program for low-MI transcripts
    default_threshold=0.1,      # Threshold for low-MI assignment in bits
    verbose=True                # Print additional status message
)
```

This function will return program labels for all transcripts without making
changes to the anndata object; they must be explicitly assigned to an attribute.

#### Assigning time values to individual observations

Call `program_times` on an anndata object which `program_select` has already
been run on. This will embed observations into a low-dimensional space, different
for each transcriptional program, find user-defined anchoring points with real-world
time values, and project cells onto that real-world time trajectory.

```
import anndata as ad
from inferelator_velocity import program_times

adata = ad.read(FILE_NAME)

# Dict that maps programs to experimental or inferred cell groups
# which are stored in a column of the `adata.obs` attribute 

time_metadata = {
    '0': 'Experiment_Obs_Column',
    '1': 'Cell_Cycle_Obs_Column'
}

# Dict that orders cell groups and defines the average time value
# for each group. Each entry is of the format
# {'CLUSTER_ID': ('NEXT_CLUSTER_ID', time_at_first_centroid, time_at_next_centroid)}
# and the overall trajectory may be linear or circular

time_order = {
    '0': {
        '1': ('2', 0, 20),
        '2': ('3', 20, 40),
        '3': ('4', 40, 60)
    },
    '1': {
        'M-G1': ('G1', 7, 22.5),
        'G1': ('S', 22.5, 39.5),
        'S': ('G2', 39.5, 56.5),
        'G2': ('M', 56.5, 77.5), 
        'M': ('M-G1', 77.5, 95)
    }
}

# Optional dict to identify programs where times should wrap
# because the trajectory is circular (like the cell cycle)

time_wrapping = {
    '0': None,
    '1': 88.0
}

program_times(
    adata,                      # Anndata object
    time_metadata,              # Group metadata columns in obs
    time_order,                 # Group ordering and anchoring times
    layer='counts',             # Layer with unprocessed integer count data
    wrap_time=time_wrapping,    # Program wrap times for circular trajectories
    verbose=True                # Print additional status message
)
```

This function will return the same anndata object with each transcriptional
program put into anndata attributes:

```
.obs['program_0_time']: Assigned time value
.obsm['program_0_pca']: Low-dimensional projection values
```

#### Embedding k-nearest neighbors graph

Call `global_graph` on an anndata object. The data provided to this function
should be standardized. The noise2self algorithm will select `k` and `n_pcs`
for the k-NN graph.

```
import anndata as ad
import scanpy as sc
from inferelator_velocity import global_graph

adata = ad.read(FILE_NAME)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

global_graph(
    adata,          # Anndata object
    layer="X",      # Layer with standardized float count data
    verbose=True    # Print additional status message
)
```

This function will return the same anndata object with a k-nn graph
added to attributes.

```
.obsp['noise2self_distance_graph']: k-NN graph
.uns['noise2self']: {
    'npcs': Number of principal components used to build distance graph,
    'neighbors': Number of neighbors (k) used to build distance graph
}
```

#### Estimating RNA velocity

Call `calc_velocity` on an anndata object. The data provided to this function
should be standardized to depth but not otherwise transformed, so that the velocity
units are interpretable. It may or may not be helpful to denoise count data prior
to calling this function. This requires a k-NN graph and calculated per-observation
time values.

```
import anndata as ad
import scanpy as sc
from inferelator_velocity import calc_velocity

adata = ad.read(FILE_NAME)
sc.pp.normalize_total(adata)

adata.layers['velocity'] = calc_velocity(
    adata.X,                                # Standardized float count data
    adata.obs['program_0_time'].values,     # Assigned time values
    adata.obsp['noise2self'],               # k-NN graph
    wrap_time=None                          # Wrap times for circular trajectories
)
```

This function will return RNA rate of change for all transcripts without making
changes to the anndata object; they must be explicitly assigned to an attribute.

#### Bounded estimate of RNA decay

Call `calc_decay_sliding_windows` on an anndata object. This requires times from
`program_times` and velocities from `calc_velocity`.

```
import anndata as ad
import numpy as np
from inferelator_velocity import calc_decay_sliding_windows

adata = ad.read(FILE_NAME)

_decay_bound = calc_decay_sliding_windows(
    adata.X,                            # Standardized float count data
    adata.layers['velocity'],           # Velocity data
    adata.obs['program_0_time'].values, # Assigned time values
    centers=np.arange(0, 60),           # Centers for sliding window
    width=1.                            # Width of each window
)

adata.varm['decay_rate'] = np.array(_decay_bound[0]).T
adata.varm['decay_rate_standard_error'] = np.array(_decay_bound[1]).T
```

This function will return decay rate, standard error of decay rate, estimate
of maximum transcription, and the centers for each sliding window.
They must be explicitly assigned to an attribute.
