"""Develop a model to forecast the returns for stocks."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMAResults
import os


class SarimaxModel():
    """The model to forecast the returns from stocks."""

    def calculate_time_diff(self, df):
        """Calculate the time interval between consecutive data points."""
        df['date'] = pd.to_datetime(df['date'], format="%Y/%m/%d")
        df['time_diff'] = 0
        df['time_diff'] = df['date'].diff().dt.days
        return df

    def modify_gapped_series(self, df):
        """Truncate or interpolate the time series having missing entries"""
        max_index = df['time_diff'].argmax()
        max_gap = df['time_diff'].max()
        if max_gap < 28:
            df = df.set_index('date')
            df = df.resample('7D').mean().interpolate()
        else:
            if max_index > len(df.time_diff) * 0.5:
                df = df.iloc[:max_index]
            else:
                df = df.iloc[max_index:]
        return df

    def select_features(self, df):
        """Remove features having high correlation with each other and low correlation with target."""
        features = df.iloc[:, 5:15]
        Y = df["target"]
        uncorrelated_factors = self.trimm_correlated(features, 0.9)

        p_corr = uncorrelated_factors.corrwith(df["target"])
        abs_corr = abs(p_corr)
        relevant_features = abs_corr[abs_corr > 0.1]
        X = uncorrelated_factors[relevant_features.index]
        return X, Y

    def trimm_correlated(self, df_in, threshold):
        """Remove highly correlated features."""
        df_corr = df_in.corr(method='pearson', min_periods=1)
        df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
        un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
        df_out = df_in[un_corr_idx]
        return df_out


    def split_data(self, X, Y):
        """Split data into train and test data sets."""
        tss = TimeSeriesSplit(n_splits=11)
        for train_index, test_index in tss.split(X, Y):
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        return X_train, Y_train, X_test, Y_test


    def model_data(self, X_train, Y_train, X_test, Y_test):
        """Find the best model and fit that SARIMAX model on the data."""
        arima_model = auto_arima(Y_train, exogenous=X_train, seasonal=True, suppress_warnings=True, trace=True)
        model = SARIMAX(endog=Y_train, exog=X_train, order=arima_model.order, seasonal_order=arima_model.seasonal_order)
        model = model.fit(endog=Y_train, exog=X_train, disp=0, maxiter=200)
        result = model.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
        #Calculate RMSE for the predictions
        error = mean_squared_error(result, Y_test, squared=False)
        relative_error = error / Y_test
        mean_relative_error = np.abs(relative_error).mean()

        return result, error, relative_error, mean_relative_error, X_test, Y_test, model

    def forecast_and_save(self, data, df_stock, path):
        """Forecast the returns for stocks based on identifier and save the model."""
        result = {}
        error = {}
        relative_error = {}
        mean_relative_error = {}
        test_input = {}
        test_labels = {}
        model = {}

        #Forecast the returns for one stock at a time
        for identity in data.identifier.unique():
            df_stock[identity] = df_stock[identity].reset_index()
            #Ignore if we have less than 40 datapoints
            if len(df_stock[identity]) < 40:
                continue
            else:
                df_stock[identity] = self.calculate_time_diff(df_stock[identity])
                #Check if the data is continous
                if (df_stock[identity]['time_diff'][1:] == df_stock[identity]['time_diff'][1]).all():
                    X, Y = self.select_features(df_stock[identity])
                    X_train, Y_train, X_test, Y_test = self.split_data(X, Y)
                    result[identity], error[identity], relative_error[identity], mean_relative_error[identity], test_input[
                        identity], test_labels[identity], model[identity] = self.model_data(X_train, Y_train, X_test, Y_test)
                else:
                    df_stock[identity] = self.modify_gapped_series(df_stock[identity])
                    X, Y = self.select_features(df_stock[identity])
                    X_train, Y_train, X_test, Y_test = self.split_data(X, Y)
                    result[identity], error[identity], relative_error[identity], mean_relative_error[identity], test_input[
                        identity], test_labels[identity], model[identity] = self.model_data(X_train, Y_train, X_test, Y_test)
            file = identity
            model[identity].save(path + '/' + file + '.pkl')

        return model, test_input, test_labels

    def load_model(self, path, data):
        """Load the saved models."""
        loaded = {}
        for identity in data.identifier.unique():
            try:
                loaded[identity] = ARIMAResults.load(path + '/' + identity + '.pkl')
            except OSError:
                continue
        return loaded


if __name__ == '__main__':

    data = pd.read_csv("/Users/madwarajhatwar/Downloads/data.csv")
    #Split the dataframe based on stock ids
    df_stock = dict(iter(data.groupby('identifier')))
    path = '/Users/madwarajhatwar/Downloads/Sarimax_models'
    os.mkdir(path)

    Model = SarimaxModel()
    saved_model, X_test, Y_test = Model.forecast_and_save(data, df_stock, path)
    loaded_models = Model.load_model(path, data)

    #Example plot prediction for 1 stock
    Y_pred = loaded_models['BMYEQL8ZXN07'].predict(start = 96, end = 103, exog = X_test['BMYEQL8ZXN07'])

    plt.plot(Y_test['BMYEQL8ZXN07'], label='ground_truth')
    plt.plot(Y_pred, label='prediction')
    plt.title('Returns for a particular stock')
    plt.xlabel('Time step (in weeks)')
    plt.ylabel('Returns')
    plt.legend()
    plt.show()