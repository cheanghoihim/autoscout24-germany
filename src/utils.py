import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


class Outlier_Analysis:
    def __init__(self, df):
        self.df = df

    def mod_z_score(self, x):
        if x:
            return (x-self.median)/(1.486*self.mean_ad)
        else:
            return (x-self.median)/(1.253314*self.mad)

    def z_score(self, x):
        return (x-self.mean)/self.std

    def outlier_detection(self, col, threshold, mod=False):
        series = self.df[col].copy()
        self.median = np.median(series)
        self.mean = np.mean(series)
        self.mad = np.sum(abs(series - self.median))/len(series)
        self.mean_ad = np.sum(abs(series - self.mean))/len(series)
        self.std = np.std(series)
        series_z_mod = series.apply(self.mod_z_score)
        series_z = series.apply(self.z_score)
        series_z_scipy = scipy.stats.zscore(series)
        assert all(series_z == series_z_scipy), 'wrong z score calculation'
        if mod:
            outlier = series[(series_z_mod > threshold) |
                             (series_z_mod < -threshold)]
            series[(series_z_mod > threshold) | (
                series_z_mod < -threshold)] = 0

        else:
            outlier = series[(series_z > threshold) | (series_z < -threshold)]
            series[(series_z > threshold) | (series_z < -threshold)] = 0
        return series, outlier, self.std

    def outlier_treatment(self, outlier_series_index):
        return self.df.drop(outlier_series_index, inplace=True)

    def outlier_gridsearch(self, columns):
        fig, axes = plt.subplots(nrows=len(columns), ncols=1)
        for i in range(len(columns)):
            z_thresh = 3
            while z_thresh < 100:
                series, outlier_series, std = self.outlier_detection(
                    columns[i], z_thresh, True)
                norm_high = series.max()
                outlier_min = outlier_series.min()
                if (outlier_min - norm_high) > 3 * std:
                    series.plot(ax=axes[i], title=columns[i])
                    outlier_series.plot(ax=axes[i])
                    self.outlier_treatment(outlier_series.index)
                    break
                z_thresh += 1
            else:
                print('No outlier for {}'.format(columns[i]))
        return self.df


class Missing_Value_Analysis:
    def __init__(self, df):
        self.df = df
        self.df_preprocess = pd.DataFrame()
        self.label_category = {}

    def label_encoding(self):
        for col in self.df.columns:
            if self.df[col].isnull().sum():
                self.label_category[col] = LabelEncoder().fit(
                    self.df[col].dropna())
                impute_ordinal = self.label_category[col].transform(
                    self.df[col].dropna())
                self.df_preprocess[col] = self.df[col]
                self.df_preprocess.loc[self.df_preprocess[col].notnull(), col] = np.squeeze(
                    impute_ordinal)
            elif self.df[col].dtype.name == 'object':
                self.label_category[col] = LabelEncoder().fit(self.df[col])
                self.df_preprocess[col] = self.label_category[col].transform(
                    self.df[col])
            else:
                self.df_preprocess[col] = self.df[col]

    def KNN_imputer(self, n):
        self.label_encoding()
        imputer = KNNImputer(n_neighbors=n)
        df_impute = pd.DataFrame(imputer.fit_transform(
            self.df_preprocess).astype('int'), columns=self.df_preprocess.columns)
        for col in self.df.columns:
            if self.df[col].dtype.name == 'object':
                df_impute[col] = self.label_category[col].inverse_transform(
                    df_impute[col])
        return df_impute

class ETS_Model:
    def __init__(self, df, col):
        self.df = df
        self.col = col

    def plot_output(self):
        self.df.plot(label='Original data')
        self.fit.fittedvalues.plot(label='Statsmodels fit - ETS Model')
    
    def fit_model(self):
        model = ETSModel(self.df[self.col], seasonal_periods=3, error='mul', trend='add', seasonal = 'mul', 
                     damped_trend=True, initial_level=self.df[self.col].mean())
        self.fit = model.fit(maxiter=10000)
        self.plot_output()
    
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def evaluate_model(self):
        expected_ets = self.df[self.col]
        predicted_ets = self.fit.fittedvalues
        mape_ets = self.mean_absolute_percentage_error(expected_ets, predicted_ets)
        print('Mean Absolute Percentage Error: {}%'.format(round(mape_ets,2)))
    
    def predict(self, window):
        print(self.fit.forecast(window))