import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_artifact(path='model_artifact.joblib'):
    return joblib.load(path)

artifact = load_artifact()
model = artifact['estimator']
feature_names = artifact['feature_names']

st.set_page_config(page_title='Wine Quality Predictor', layout='centered')
st.title('🍷 Boutique Winery — Wine Quality Predictor')
st.write(
    'Enter the chemical attributes of a wine sample to predict whether it meets '
    'premium standards (quality ≥ 7).'
)

# Main page
st.header("Input wine attributes")
inputs = []
cols = st.columns(2)

for i, feat in enumerate(feature_names):
    with cols[i % 2]:
        val = st.number_input(label=feat, value=0.0, format="%.4f")
        inputs.append(val)

# button
if st.button('Predict Wine Quality'):
    X = np.array(inputs).reshape(1, -1)
    proba = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])

    label = "✅ Good Quality Wine" if pred == 1 else "❌ Not Good Quality Wine"

    # results
    with st.container():
        st.markdown("### 🧾 Prediction Result")
        if pred == 1:
            st.success(label)
        else:
            st.error(label)

        st.metric(label="Confidence Score", value=f"{proba:.2%}")
        st.progress(int(proba * 100))
        st.caption(
            "The confidence score is the model's estimated probability "
            "that the wine is of good quality."
        )
