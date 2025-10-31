"""ALIEGNN: Atomistic Line Equivariant Graph Neural Network - Minimal Setup"""

import setuptools

setuptools.setup(
    name="aliegnn",
    version="2025.10.30",
    description="ALIEGNN: Atomistic Line Equivariant Graph Neural Network",
    install_requires=[
        # Core scientific computing
        "numpy>=1.24.4",
        "scipy>=1.10.1",
        
        # NOTE: PyTorch, torchvision, and DGL must be installed manually
        # with the correct CUDA version BEFORE running pip install -e .
        # 
        # Installation commands:
        #   pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
        #       --index-url https://download.pytorch.org/whl/cu118
        #   conda install -c dglteam/label/th24_cu118 dgl
        #
        # Uncomment below only if using CPU-only versions:
        # "torch>=2.4.1",
        # "torchvision>=0.19.1",
        # "dgl>=2.4.0",
        
        # ALIEGNN dependencies
        "jarvis-tools>=2025.5.30",
        "e3nn>=0.5.8",
        
        # ML and data processing
        "scikit-learn>=1.3.2",
        "pandas>=2.0.3",
        "pytorch-ignite>=0.5.2",
        "pydantic>=1.10.23,<2.0",  
        
        # Visualization and utilities
        "matplotlib>=3.7.5",
        "tqdm>=4.66.5",
        "pyparsing>=3.1.4",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
