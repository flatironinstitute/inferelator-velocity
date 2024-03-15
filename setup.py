import os
from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "pandas>=1.0",
    "anndata>=0.8",
    "scanpy",
    "scipy",
    "joblib",
    "scikit-learn",
    "tqdm",
    "sparse_dot_mkl",
    "leidenalg",
    "numba"
]

tests_require = [
    "coverage",
    "pytest"
]

version = "1.1.1"

# Description from README.md
long_description = "\n\n".join(
    [open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "README.md"
        ),
        "r"
    ).read()]
)

setup(
    name="inferelator_velocity",
    version=version,
    description="Inferelator-Velocity Calcualtes Dynamic Latent Parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flatironinstitute/inferelator-velocity",
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(include=[
        "inferelator_velocity",
        "inferelator_velocity.*"
    ]),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="pytest",
)
