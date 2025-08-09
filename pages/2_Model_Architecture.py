import io
import sys
import streamlit as st
from pathlib import Path
from contextlib import redirect_stdout

sys.path.append(str(Path(__file__).resolve().parent.parent))

from classifier.utils.classify import *
from classifier import __version__


st.title(f"AGN Spectral Classifier Version {__version__}")
st.write("Here is the model architecture of the Neural Network.")

def display_model_architecture(model, i):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.summary()    
    model_summary_string = buffer.getvalue()
    st.subheader(f"Model-{i} Summary:")
    st.code(model_summary_string, language='text')

a0 = ClassificationModel()

display_model_architecture(a0.model1, 1)
display_model_architecture(a0.model2, 2)
display_model_architecture(a0.model3, 3)