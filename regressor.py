from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.decomposition import KernelPCA


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([('scaler', StandardScaler()),
                             ('anova', SelectKBest(f_regression, k=3000)),
                             ('KernelPCA',KernelPCA(n_components=20)),
                             ("RF", RandomForestRegressor(n_estimators=200))])
        
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)