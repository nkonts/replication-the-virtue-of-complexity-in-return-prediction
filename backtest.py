import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, precision_score, recall_score, accuracy_score

class Backtest():
    def __init__(self, z: int=1000, T: int=12, use_type=np.float32):
        self.z = z
        self.T = T
        self.use_type = np.float32
        self.prediction = None
        self.performance = None

    def predict(self, X: np.array, y: np.array):
        """Trains a ridge regression on T training samples and creates a 1-step ahead prediction.
        'Timing Strategy' is the return of the forecast weighted by its forecast value. Both 'Timing Strategy' and 'Market Return'
        are 1-step ahead and refer to the next month. E.g. for Index 2020-01-01, those values refer to the return in 2020-02-01.

        Args:
            X (np.array): independent variables
            y (np.array): dependent variable
        """
        backtest = []
        T_max, self.P = X.shape
        self.c = self.P / self.T

        index = list(range(self.T, T_max))
        for t in index:
            S_train = X[t-self.T:t].astype(self.use_type)
            R_train = y[t-self.T:t].astype(self.use_type)

            S_test = X[t:t+1].astype(self.use_type)
            R_test = y[t:t+1].astype(self.use_type)

            # Ridge.alpha is adjusted by T to get the same results as in the paper. 
            beta = Ridge(alpha=(self.z*self.T), solver="svd", fit_intercept=False).fit(S_train, R_train).coef_
            forecast = S_test @ beta
            # Keep in mind that R_test is 1-step ahead, thus it satisfies beta'*S_t*R_t+1
            timing_strategy = forecast * R_test

            backtest.append({
                "index": R_test.index[0],
                "beta_norm": np.sqrt((beta**2).sum()),
                "forecast": forecast[0],
                "timing_strategy": timing_strategy[0],
                "market_return": R_test[0]
            })
        # The last value for market_return is NaN since it is predicting the next month
        self.backtest = pd.DataFrame(backtest).set_index("index")
        self.prediction = self.backtest["forecast"]
        return self


    def calc_performance(self, time_factor:int = 12):
        """Calculates various performance metrics for the backtest.

        Args:
            time_factor (int, optional): Factor to annualize the data e.g. 12 for a monthly frequency. Defaults to 12.
        """
        data = self.backtest.dropna()
        # Calculate Alpha & Beta of the timing strategy
        market_reg = LinearRegression().fit(data[["market_return"]], data["timing_strategy"])
        beta = market_reg.coef_[0]
        alpha = market_reg.intercept_

        # Annualize returns
        sqrt_time_factor = np.sqrt(time_factor)
        mean = data["timing_strategy"].mean()*time_factor
        std = data["timing_strategy"].std()*sqrt_time_factor
        mean_market = data["market_return"].mean()*time_factor

        self.performance = {
            "beta_norm_mean" : data["beta_norm"].mean(),
            "Market Sharpe Ratio" : (data["market_return"].mean()*time_factor) / (data["market_return"].std()*sqrt_time_factor),
            "Expected Return" : mean,
            "Volatility" : std,
            "R2" : r2_score(data["market_return"], data["forecast"]),
            "SR" : mean/std,
            # Adjust IR by beta to get a better scaling since mean converges to zero for higher c
            "IR" : (mean - mean_market*beta)/std, 
            "Alpha" : alpha, 
            # Does sign(forecast) match sign(market_return)?
            "Precision": precision_score(data["market_return"] > 0, data["forecast"] > 0),
            "Recall": recall_score(data["market_return"] > 0, data["forecast"] > 0),
            "Accuracy": accuracy_score(data["market_return"] > 0, data["forecast"] > 0),
        }
        return self.performance

    
