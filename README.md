OBJECTIVE :
The research aims to develop a tool for forecasting variance of BTC_USD using regime switching models and exogenous Implied Volatility inputs.
We hypothesise that market expectations (through IV) do give some explanatory power in forecasting variance for the next hour.  

DATA :
The dataset consists of BTC–USD market and options data organised by calendar year (2022-2025) and further partitioned into quarterly subsets (Q1–Q4). Each year is processed independently, and all modelling, forecasting, and evaluation are performed at the quarterly level to preserve temporal structure and avoid cross-period leakage. High-frequency observations are aligned within each quarter, and all results are reported and compared on a quarter-by-quarter basis across years.

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
- StrategyMSG : This file compares a simple options trading strategy using variance from base model versus a IV corrected Base model forecast.

Note : Two other folders "outputs" and "data" can be found on the shared drive. 
Place the two sub-folders in the parent folder where THESIS is located for optimal notebook run.
