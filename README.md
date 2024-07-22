# COALA - COlorectal CAncer Liver metastasis Assessment 
COALA is an AI-powered tool for automated segmentation of colorectal liver metastases (CRLM) in contrast-enhanced CT scans. This repository contains code for inference and evaulation. Model weights can be downloaded here: https://www.dropbox.com/scl/fo/cb7zc5ihhenclwvi6pwzc/ADpAzY20WsaHje-zo11drQU?rlkey=m7p532mh8ke8nmw3eyfj96ex1&st=o8355kgn&dl=0. 

## Key Features

1. **Automated CRLM Segmentation**: Utilizes deep learning to accurately identify and segment CRLM in CT scans.
2. **Enhanced Consistency**: Reduces inter-observer variability, leading to more reliable assessments.
3. **Research Tool**: Aids in quantitative analysis of CRLM for clinical research and treatment planning.

## Installation and Setup

COALA does not require a GPU. We very strongly recommend you install COALA in a virtual environment.
Python 2 is deprecated and not supported. Please make sure you are using Python 3.
For more information about COALA, please read the following paper:

TODO

Please also cite this paper if you are using COALA for your research!


1. Install COALA:
```
 git clone https://github.com/JackieBereska/COALA.git
 cd COALA
 pip install -e .
```
Adjust all paths by globally searching for 'your_path'. Place your CT scan files (in .nii.gz format) in the prediction_input folder. Place the weights in the appropriate folder.

2. Run COALA:
```
python main_crlm.py
```
Results will be saved in the prediction_output folder.

3. Evaluate COALA:
```
python CRLM/helper/eval.py
```

## Contact
For questions or issues, please open an issue in this repository or contact j.i.bereska@gmail.com.
