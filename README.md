# A robust low-pass filtering graph diffusion clustering framework for hyperspectral images

Aitao Yang; Min Li; Yao Ding; Yaoming Cai; Jie Feng; Yujie He; Yuanchao Su

___________

The code in this toolbox implements the ["A robust low-pass filtering graph diffusion clustering framework for hyperspectral images"]( https://www.sciencedirect.com/science/article/abs/pii/S0950705125008287). 



Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

Yang, Aitao, et al. "A robust low-pass filtering graph diffusion clustering framework for hyperspectral images." Knowledge-Based Systems (2025): 113782.

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



