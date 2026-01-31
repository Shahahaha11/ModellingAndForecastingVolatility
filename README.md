FILE TREE

THESIS/
├── data/
├── models/
├── src/
├── models/
│   ├── A_DataPrep.ipynb
│   ├── B_ComputeIV.ipynb
│   ├── C_Model_T.ipynb
│   ├── D_StrategyMSG.ipynb
├── src/
│   ├── LSTM_lib.py
│   ├── __pycache__
│   ├── bsm_IV.py
│   ├── dataPrep.py
│   ├── interpol.py
│   ├── interpolate.py
│   ├── metrics.py
│   ├── mfiv.py
│   ├── mfivDaily.py
│   ├── msGarch.py

FILES IN "src" SUB FOLDER :
The model uses helper functions defined in files inside src.

FILES IN "models" SUB FOLDER :
- DataPrep : Contains sample data preparation. (Replaced by class src.dataPrep)
- ComputeIV : Contains sample IV computations. (Replaced by class src.interpol)
- Model_T : The main model which uses IV measures forecasted at time t using options expiring in 5 hours from now. 

Note : Two other folders "outputs" and "data" can be found on the shared drive. 
Place the two sub-folders in the parent folder where THESIS is located for optimal notebook run.
