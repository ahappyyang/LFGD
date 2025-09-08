# A robust low-pass filtering graph diffusion clustering framework for hyperspectral images

Aitao Yang; Min Li; Yao Ding; Yaoming Cai; Jie Feng; Yujie He; Yuanchao Su

___________

The code in this toolbox implements the ["A robust low-pass filtering graph diffusion clustering framework for hyperspectral images"]( https://ieeexplore.ieee.org/document/10746460). 



Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

A. Yang, M. Li, Y. Ding, X. Xiao and Y. He, "An Efficient and Lightweight Spectral-Spatial Feature Graph Contrastive Learning Framework for Hyperspectral Image Clustering," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-14, 2024, Art no. 5537714, doi: 10.1109/TGRS.2024.3493096.

      @ARTICLE{YANG2025113782,
        author={Aitao Yang and Min Li and Yao Ding and Yaoming Cai and Jie Feng and Yujie He and Yuanchao Su},
        journal={Knowledge-Based Systems}, 
        title={A robust low-pass filtering graph diffusion clustering framework for hyperspectral images}, 
        year={2025},
        volume={324},
        issn = {0950-7051},
        number={},
        pages={113782},
        doi={https://doi.org/10.1016/j.knosys.2025.113782}}

    
System-specific notes
---------------------
The codes of networks were tested using PyTorch 1.12.1 version (CUDA 10.1) in Python 3.7 on Ubuntu system.

How to use it?
---------------------
Directly run **MAIN_SA.py** functions with different network parameter settings to produce the results. Please note that due to the randomness of the parameter initialization, the experimental results might have slightly different from those reported in the paper.

For the datasets:
Add your dataset path to function “load_dataset” in function.py



