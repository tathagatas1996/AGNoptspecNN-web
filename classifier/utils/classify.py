from importlib.resources import files
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
import tensorflow.keras as keras 

class ClassificationModel:
    def __init__(self):
        """
        ==========
        Using three models here:   Dimensions:      (None, nx ,  1) 
        model_v1_ninterp500  : The model which uses (None, 500,  1) as input. 
        model_v1_ninterp1000 : The model which uses (None, 1000, 1) as input.
        model_v1_ninterp2000 : The model which used (None, 2000, 1) as input.
        ==========
        """
        model_path1 = files("classifier.models").joinpath("model_v1_ninterp500.keras" )
        model_path2 = files("classifier.models").joinpath("model_v1_ninterp1000.keras")
        model_path3 = files("classifier.models").joinpath("model_v1_ninterp2000.keras")

        self.model1 = tf.keras.models.load_model(str(model_path1))
        self.model2 = tf.keras.models.load_model(str(model_path2))
        self.model3 = tf.keras.models.load_model(str(model_path3))

        # ===== The input dimension for classification: spectra will be interpolated into  ===== #
        # ===== three spectra 
        self.nx1 = 500  # 
        self.nx2 = 1000 #
        self.nx3 = 2000 #


    def reshape_spec(self, wave, X, N):
        """
        ============
        wave : Wavelength 
        X    : Flux
        N    : Number of interpolated spectral points
        ============
        returns: reshaped spectra. 
        """
        flux_func1 = interp1d(wave, X)
        wave_st  = np.linspace(min(wave), max(wave), N)
        flux_std = flux_func1(wave_st)
        flux_std = flux_std/np.mean(flux_std)
        return (wave_st, flux_std)

    def vote_array(self, arr):
        """
        =============
        Take an array and check for the element which has the maximum occurance.
        =============
        """
        unique, counts = np.unique(arr, return_counts=True)
        if np.any(counts >= len(arr)/2):    
            return unique[np.argmax(counts)]
        else:
            return np.random.choice(arr) 

    def classify_the_spectrum(self, Spectrum):
        """
        The single spectrum is takes as input for classification.
        Classify a single spectrum here. The spectrum should be passed as an argument.
        """
        wave = Spectrum[:,0]
        X = Spectrum[:,1]

        wave_1, X_1 = self.reshape_spec(wave, X, self.nx1)
        wave_2, X_2 = self.reshape_spec(wave, X, self.nx2)
        wave_3, X_3 = self.reshape_spec(wave, X, self.nx3)

        X_for_prediction1 = X_1.reshape(1,-1,1)
        X_for_prediction2 = X_2.reshape(1,-1,1)
        X_for_prediction3 = X_3.reshape(1,-1,1)

        ypred1 = np.argmax((self.model1.predict(X_for_prediction1, verbose=0).T))
        ypred2 = np.argmax((self.model2.predict(X_for_prediction2, verbose=0).T))
        ypred3 = np.argmax((self.model3.predict(X_for_prediction3, verbose=0).T))

        prediction_model  = np.array([1,2,3])
        model_label       = np.array(["Model-1", "Model-2", "Model-3"])  
        y_prediction_list = np.array([ypred1, ypred2, ypred3])
        
        ypred = self.vote_array(y_prediction_list)

        return (ypred, prediction_model, model_label, y_prediction_list)