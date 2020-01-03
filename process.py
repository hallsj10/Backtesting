# Empirical Analysis of Forward Returns
## Steve Hall
### January 2020
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

# Load data
raw_px = pd.read_excel('C:\devl\LeadingIndicators\Data_tests_v3.xlsx', 
                         sheet_name='Clean', index_col='Dates')

desc = raw_px.describe().transpose()
min_rows = int(desc['count'].min())
clean_px = raw_px.iloc[-min_rows:]
# TODO: Cleaner broadcasting
# clean_px['TwosTens'] = clean_px["USGG10YR"] - clean_px["USGG2YR"] 

def pct_returns(df, end_dt, beg_dt):
    """Calculates percentage returns of all series in a data frame"""
    return (df.shift(periods=end_dt) / df.shift(periods=beg_dt)) - 1

# RESPONSE VARIABLES
response_vars = {'SPX','USGG10YR'}    
response = clean_px[response_vars]
weeks = 8
forecast_pd = weeks*5  # Number of days in forecast period.  
forecast_rets = pct_returns(response, -forecast_pd, 0) # Shift back for fwd returns.
select_Y = forecast_rets
#select_Y = frcst_rets.rank(axis=1, ascending = False) # categorical rank

# PREDICTOR VARIABLES
# One-month pct return.
x1 = pct_returns(clean_px, 0, 21) # 21 days in a month.
x2 = x1.add_prefix('1M_')
# Three-month pct return.
x3 = pct_returns(clean_px, 0, 21*3)
x4 = x3.add_prefix('3M_')
# Merge predictors
select_X = x2.merge(x4,left_index=True, right_index=True)
# Compute relative returns
col_names = x4.columns.to_list()
x_ratios = x4.copy()
for c, col in enumerate(col_names):
    x_ratios['SPX' + '-' + col]  = x4['3M_SPX'] - x4[col]

# SELECT DATASET
select_dataset = select_Y.merge(select_X,left_index=True, right_index=True)
select_dataset = select_dataset.dropna()
stats = select_dataset.describe()

# Split the time series into train and test series
X = select_dataset.filter(regex='^3M', axis=1)
Y = select_dataset['SPX']

split = len(X) - 500 # days in OOS Test
X_train = X.iloc[:split]
X_test = X.iloc[split+5:]
Y_train = Y.iloc[:split]
Y_test = Y.iloc[split+5:].to_frame()
resids = Y_test.copy()
results = dict()

# KNN Regression
num_neighbors = [50, 75, 100]

for i, wgts in enumerate(['uniform', 'distance']):
    for j, k in enumerate(num_neighbors):
        knn = neighbors.KNeighborsRegressor(k, weights=wgts)
        prediction = knn.fit(X_train, Y_train).predict(X_test)
        Y_test[wgts + '_' + str(k)] = prediction
        resids[wgts + '_' + str(k)] = Y_test['SPX'] - prediction
        results[wgts + '_' + str(k)] = np.sqrt(
                mean_squared_error(Y_test['SPX'],prediction))
        
plt.scatter(Y_test['uniform_50'], Y_test['SPX'])
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.axis('tight')
plt.title("KNN (k = 50, weights = 'uniform')")
plt.show()

select_model = neighbors.KNeighborsRegressor(50, weights='uniform')
final_X = select_X.filter(regex='^3M', axis=1)
final_X = final_X.iloc[-forecast_pd:]
final_X['y_pred'] = select_model.fit(X, Y).predict(final_X)

