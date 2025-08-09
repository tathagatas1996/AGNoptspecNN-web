import sys
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classifier.utils.classify import *
from classifier import __version__

st.title(f"AGN Spectral Classifier Version {__version__}")

uploaded_file = st.file_uploader(r"Upload text spectrum file for form: Column 1-wavelength ($$\lambda$$), Column-2 flux", type=["txt"])

if uploaded_file is not None:
    spectrum = np.loadtxt(uploaded_file)

    if spectrum.ndim == 1 or spectrum.shape[1] < 2:
        st.error("Error: File must contain at two columns (e.g., wavelength and flux).")
    else:
        st.success("File loaded successfully!")

    a0 = ClassificationModel()
    final_type, model, model_label, type_list = a0.classify_the_spectrum(spectrum)

    type_no = np.array([0,1,2])
    type_labels = np.array(["Type 1", "Type 1.9", "Type 2"], dtype="str")

    if final_type == 0:
        message="Type 1 AGN"
    elif final_type == 1:
        message="Type 1.9 AGN"
    elif final_type == 2:
        message="Type 2 AGN"

    st.success(f"This is a {message} spectrum.")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(spectrum[:,0], spectrum[:,1]/np.mean(spectrum[:,1]))
    axs[0].set_xlabel("Wavelength [Angstrom]")
    axs[0].set_ylabel("Flux [mean normalized]")
    axs[0].set_title("Spectral Plot")

    axs[1].plot(model, type_list, "--P", markersize=10)
    axs[1].set_xticks(model)
    axs[1].set_xticklabels(model_label)
    axs[1].set_yticks(type_no)
    axs[1].set_yticklabels(type_labels)
    axs[1].set_ylim(-0.1,2.1)
    axs[1].set_title("Output from each model")
    st.pyplot(fig)