FILE TREE

THESIS/
│   README.md
│   analysis
│   data
│   models       # Main Sub Folder
│   outputs
│   requirements.txt
│   src
│   
├── analysis/
│   
├── data/
│   
├── models/
│   A_DataPrep.ipynb 
│   B_ComputeIV.ipynb
│   C_Model_T.ipynb
│   D_StrategyMSG.ipynb
│   
├── outputs/
│   
├── src/
│   LSTM_lib.py
│   bsm_IV.py
│   dataPrep.py
│   interpol.py
│   interpolate.py
│   metrics.py
│   mfiv.py
│   mfivDaily.py
│   msGarch.py
│   
│   LSTM_lib.cpython-313.pyc
│   __init__.cpython-313.pyc
│   dataPrep.cpython-313.pyc
│   functionsMS.cpython-313.pyc
│   interpol.cpython-313.pyc
│   interpolate.cpython-313.pyc
│   metrics.cpython-313.pyc
│   mfiv.cpython-313.pyc
│   mfivDaily.cpython-313.pyc
│   msGarch.cpython-313.pyc


FILES IN "models" SUB FOLDER :

- Model_T : The main model which uses IV measures forecasted at time t using options expiring in 5 hours from now. 