# SQT1-L-Carnitine

# In-silico modelling of SQT1 and the effects of L-Carnitine.

This model is part of the Journal X (2023) paper: 'Beneficial normalization of cardiac repolarization by carnitine in transgenic SQT1 rabbit models' by Ilona Bodi, Konstantin Michaelides, Lea Mettke, Tibor Hornyik, Stefan Meier, Saranda Nimani, Stefanie Perez-Feliz, Ibrahim el-Battrawy, Manfred Zehender, Michael Brunner, Jordi Heijman, Katja E. Odening.
Doi: X

:file_folder: The [MMT](https://github.com/HeijmanLab/SQT1-L-Carnitine/tree/main/MMT) folder contains the adapted O'Hara Rudy human ventricular cardiomyocyte model (ORd), wherein Loewe et al. (2014) I<sub>Kr</sub> SQT1 and WT formulations were implemented.

:file_folder: The [Data](https://github.com/HeijmanLab/SQT1-L-Carnitine/tree/main/Data) folder contains all the experimental data needed for the simulations. 

:file_folder: The [Figures](https://github.com/HeijmanLab/SQT1-L-Carnitine/tree/main/Figures) folder is a results folders where some of the figures will be stored. 

:computer: :snake: The Python script to create the simulations and figures used in the paper can be found in [SQT1_script](https://github.com/HeijmanLab/SQT1-L-Carnitine/blob/main/SQT1_script.py).

:computer: :snake: The functions used for the above-mentioned simulations can be found in [SQT1_functions](https://github.com/HeijmanLab/SQT1-L-Carnitine/blob/main/SQT1_functions.py).


## Virtual environment (Instructions for pip):

Follow the below mentioned steps to re-create te virtual environment with all the correct package versions for this project.

:exclamation: **Before creating a virtual environment please make sure you fully installed Python 3 and myokit (v. 1.35.2) already. Please follow these steps carefully: http://myokit.org/install.** :exclamation:


***1. Clone the repo:***

`git clone git@github.com:HeijmanLab/SQT1-L-Carnitine.git`

***2. Create virtual environment:***

This re-creates a virtual environment with all the correct packages that were used to create the model and run the simulations. 

- Set the directory:

`cd SQT1-L-Carnitine`

- Create the virtual environment:

`python -m venv SQT1_env`

- Activate the environment:

`On Windows: SQT1_env\Scripts\activate`

`On macOS/Linux: source SQT1_env/bin/activate`

- Install packages from requirements.txt:

`pip install -r requirements.txt`

***3. Setup and launch spyder (or any other IDE of your liking) from your anaconda prompt:***

`spyder`

