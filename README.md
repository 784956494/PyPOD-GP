# PyPOD-GP
In repository, we implement a prototype Pytorch-based implementation of the (localized) POD-GP method that enables GPU utilization during both the learnining and inferece stages. We provide an example running script in ```run_pod.py``` for reference.

## Setup
To use our code, first create a Python 3.10 environment and install FEniCS, Dolfin, and mshr following the instructions from their official website: 
```
https://fenics.readthedocs.io/en/latest/installation.html
```
Then install the requirements with the command:
```
pip install -r requirements.txt
```
## Learning(Training)
To use our example script for training, create a ```PyPOD_GP``` object by adjusting the following arguments in command line:
```
optional arguments:
  -h, --help            show this help message and exit
  --x X                 space dimension for x-axis in mesh
  --Y Y                 space dimension for y-axis in mesh
  --z Z                 space dimension for z-axis in mesh
  --x-dim X_DIM         number of cells in x-axis
  --y-dim Y_DIM         number of cells in y-axis
  --z-dim Z_DIM         number of cells in z-axis
  --time-steps TIME_STEPS
                        number of cells in z-axis
  --cuda CUDA           which cuda device to use (-1 for cpu training)
  --told TOL            padding for float point comparisons
  --num-modes NUM_MODES the number of modes to use, could be integer or list
  --Nu NU               the number of functional units
  --surfaces SURFACES   subdomain of interests for G matrix
  --sampling-interval SAMPLING_INTERVAL
                        the last time step
  --degree DEGREE       the degree of polynomials to integrate
  --steps STEPS         the number of steps to take to solve the ODE
  --save SAVE           whether or not to save training results
  --save-dir SAVE_DIR   paths to the direcotry to save training results
  --task TASK           which task to perform, can be any of [train, predict, both]
  --save-format SAVE_FORMAT 
                        which format to save/read the trained constants, can be any of [txt, csv]
```
Then modify the script with paths to the corresponding dataset, floor plan file, and power density file for each floor plan. Then run the following command for training and saving the results:
```
python run_pod.py --[ARGUMENTS] --save 1 --save-dir [PATHS] --task train --save-format [FORMAT]
```

## Inference
To predict the temperature when there are already trained constants saved, simple run the following command to get the inference results after modifying the script to include the paths to the modes data:
```
python run_pod.py --[ARGUMENTS] --task predict --save-dir [PATHS TO CONSTANTS] --save-format [FORMAT]
```
Note that the script assume that the modes and constants are saved in a particular format, namely ```save-dir/[C/G/Ps_matrix]_[Nu].[format]```.

## Complete Training and Prediction Procedure
To run the entire training and inference program, run the following command:
```
python run_pod.py --[ARGUMENTS] --save-dir [PATHS TO SAVE RESULTS] --task both
```

## Test CPU Data
we provide a test dataset for the purpose of demonstration. To run the dataset, first download the CPU temperature data from 
```
https://drive.google.com/drive/folders/1-id6igacZXYnFBT5M6_n7BC4vvWY6NVF?usp=sharing
```
The data was collected on a AMD ATHLON II X4 610e CPU chip. After downloading the dataset, edit the ```run_pod_cpu.py``` file for the corresponding paths of the files. The parameters in the ```config.py``` was already configured for this dataset. To run the code, simply run the command:
```
python run_pod_cpu.py
```

## Citations
If you find this repository useful and use PyPOD-GP in your research, use cite the following:
```
Lin Jiang, Anthony Dowling, Yu Liu, Ming-C. Cheng, Ensemble learning model for effective thermal simulation of multi-core CPUs, Integration, Elsevier, Volume 97, 2024

Lin Jiang, Anthony Dowling, Ming-Cheng Cheng, Yu Liu, PODTherm-GP: A Physics-based Data-Driven Approach for Effective Architecture-Level Thermal Simulation of Multi-Core CPUs, IEEE Transactions on Computers, 72(10), 2023.10

Alessandro Pulimeno, Graham Coates-Farley, Martin Veresko, Lin Jiang, Ming-Cheng Cheng, Yu Liu, Daqing Hou, Physics-driven Proper Orthogonal Decomposition: A Simulation Methodology for Partial Differential Equations, MethodsX, Elsevier, 10 (2023), 2023.4

Lin Jiang, Yu Liu, Ming-Cheng Cheng, Fast Accurate Full-Chip Dynamic Thermal Simulation with Fine Resolution Enabled by a Learning Method, IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, Vol. 42, Issue 8, 2022.12
```
