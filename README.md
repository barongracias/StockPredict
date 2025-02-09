# Stock Predict Project
This project was to learn best practises of training, deploying and monitoring a machine learning model for stock prediction.

### Initial set up
Open this project and activate virtual environment using

```
source stock_venv/bin/activate
```

### Getting the data
The `download_data.py` file contains the loading process using the Yahoo Finance API and will save the data as parquet files to the data directory.

Currently the file is sourcing Mastercard stock data for 730 days from 2022-01-01, feel free to adjust this to your preference.

Execute `download_data.py` via terminal i.e.

```
python download_data.py
```

### Baseline model
Use the `baseline_model.py` file to create an initial model to compare against. This will transform the data to create features and targets using the `preprocess.py` file. This baseline model simply uses the previous price to predict the next price and extracts the mean absolute error.

Execute `baseline_model.py` via terminal i.e.

```
python baseline_model.py
```

### Train model
Use the `train_model.py` file to train the data using a various machine learning models. The model performance metrics will be saved as plots in the results directory. Here you will find plots for residuals, actual vs predicted, and learning curve.

#### Execution arguments
**Model selection:**
The training model can be passed here, with the default set to lasso. The model options are `lasso` for Lasso, `light` for LGBMRegressor, `boost` for XGBRegressor, and `forest` for RandomForestRegressor.

For example:
```
python train_model.py --model forest
```

**Hyperparameter tuning:**
The model hyperparamter can be tuned using grid search. The default is set to False, simply call the tune argument to set to True. This will find the optimal hyperparamters and save the new model and plots.

For example:
```
python train_model.py --tune
```

**Sample size:**
A different sample size can be set for the training process as an integer. The default is set to None.

For example:
```
python train_model.py --sample 5
```

**Trials:**
The number of trials can be passed as an integer, with a default set to 10.

For example:
```
python train_model.py --trials 5
```

**Combination:**
All of the arguments above can be used simultaneously.

For example:
```
python train_model.py --model forest --tune --sample 5 --trials 5
```

### Deployment and Monitoring
In development.