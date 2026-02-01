from setuptools import setup, find_packages

setup(
    name="sincps",
    version="1.0.0",
    description="Semantic-aware Implicit Neural Compression for Physics Simulations",
    author="ByteBoost Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "h5py>=3.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.3.0"],
    },
)
