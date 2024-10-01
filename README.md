AutoCP
==============================

Implementation of AutoCP - Lightweight Neural Architecture Search for Cerebral Palsy Detection.

### Libraries

The code is based on Python >= 3.10 and PyTorch (2.3.0). Run the following command to install all the required packages 
from setup.py:
```
pip install .
```

### Dataset 

The npy files for the CP dataset have to be stored in the following folder:
```
data
├── npy_files
│   └── processed
│       ├── cp_19   # put the 19-point npy files here
│       └── cp_29   # put the 29-point npy files here
```
Additionally, you need to specify your node in the config_node.yaml where you can guide the algorithm.
This enables you to run the code on other clusters and put the right paths into place.


# Run configuration
The user is provided with the following modes:

- "nas" Mode: Executes code related to neural architecture search (NAS) using AutoCP.
- "buld" Mode: Executes a retraining of a defined architecture inside the config file and executes the test.

To run a model search, check the config file and change the following parameters in the config_node:

```
work_dir    -> path where you want to store the logs    --> /Users/yourUser/AutoCP/logs
root_folder -> this should be something like           -->  /Users/yourUser/AutoCP/data/npy_files
```

The modes can either be activated or deactivated by setting the flags to True or False - refer to the different 
config run files for further information.

Make sure you have the chosen config file ready, which can be given as ``--config /path/to/your/config``.
Furthermore, you have to provide the path for the datasets via ``--node /path/to/your/node/config``.
The standard config files can be found in the config folder, where you can also specify your node paths.

## Run NAS 
To run a NAS model search, check the config file and define a Search Space and other parameters.

Afterward, execute:
```
python main.py -m nas
```

## Run Build
To build an architecture from the obtained weights from the NAS, change the config retrain_... parameters.

Afterward, execute:
```
python main.py -m build
```

## Results

The results reported in our study are stored in the `./logs` folder.
There are also predefined configs stored there, which can be used :).

The best architecture reported in the paper is **argmax_1003**.

## Citation and Contact

If you have any questions, feel free to email `felix.e.f.tempel@ntnu.no`.

Please cite our paper if you use this code in your research. :)
```
@misc{tempelLightweightNeuralArchitecture2024,
	title = {Lightweight {Neural} {Architecture} {Search} for {Cerebral} {Palsy} {Detection}},
	url = {http://arxiv.org/abs/2409.20060},
	publisher = {arXiv},
	author = {Tempel, Felix and Ihlen, Espen Alexander F. and Strümke, Inga},
	year = {2024},
}
```
