import numpy as np
from scipy import signal

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120

def get_enso_mean(tas):
    return tas.loc[:, en_lat_bottom:en_lat_top, en_lon_left:en_lon_right].mean(dim=('lat','lon'))

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, temperatures_xray, n_burn_in, n_lookahead):
        pass

    def transform(self, temperatures_xray, n_burn_in, n_lookahead, skf_is):
        """Use world temps as features."""        
        # Set all temps on world map as features
        valid_range = range(n_burn_in, temperatures_xray['time'].shape[0] - n_lookahead)
        time_steps, lats, lons = temperatures_xray['tas'].values.shape
        X = temperatures_xray['tas'].values.reshape((time_steps,lats*lons))
        
        X.reshape((-1, 12, lats*lons))
        count_matrix = np.ones(X.shape)
        monthly_mean = (X.cumsum(axis=0) / count_matrix.cumsum(axis=0))
        monthly_mean.reshape(-1, lats*lons)

        X = X - monthly_mean

        Y = X[valid_range,:]
        for i in range(1, 12):
            Y = np.hstack((Y, X[[v-i for v in valid_range],:]))
                
        return Y