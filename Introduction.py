import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from  classifier.utils.classify import *
from classifier import __version__

st.title(f"AGN Spectral Classifier Version {__version__}")
st.header("App Description")
st.write("This app uses 1-D convolutional neural networks for classifying the data. The training has been performed using simulated optical data.")

st.header("Training data")
st.write(r"""
         - The training data has: AGN-continuum, galaxy templates (Mannucci et al. 2001 [MNRAS, 326, 745](https://ui.adsabs.harvard.edu/abs/2001MNRAS.326..745M/abstract)), [iron line template](http://servo.aob.rs/FeII_AGN/link3.html), and Gaussian emission lines.
         - The training takes place in three steps. We interpolate the data to have $N_{\rm input}$=500, 1000, and 2000 points. The neural network trains on these three sets and gives three models where the input layer is of size $N_{\rm input}$."
""")

st.header("Limitations and Further improvements")
st.subheader("Limitations")
st.write("""
         - There are chances of misclassification for some spectra as the training sample is based on simulated data. In this version, we try to limit misclassification by using three trained models.
         - The model classifies the spectra into three classes only i.e. Type-1, Type-1.9, and Type-2. Future models will extend the classification to all other types of AGNs supermassive black hole transients and galaxies. 
""")

st.subheader("Future Improvements")
st.write("The app will eventually be improved in the later versions with:")

st.write("""
         - There are chances of misclassification for some spectra as the training sample is based on simulated data. In this version, we try to limit misclassification by using three trained models.
         - The model classifies the spectra into three classes only, i.e., Type-1, Type-1.9, and Type-2. Future models will extend the classification to all other types of AGNs, supermassive black hole transients, and galaxies.
""")