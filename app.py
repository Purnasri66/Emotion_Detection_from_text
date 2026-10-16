
import re
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(
    page_title="Emotion Detection from Text",
    page_icon="💬",
    layout="centered"
)

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "emotion_detection_pipeline.joblib"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Please upload "
            "models/emotion_detection_pipeline.joblib "
            "to your GitHub repository."
        )
    return joblib.load(MODEL_PATH)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.title("💬 Emotion Detection from Text")

st.write(
    "Enter a sentence and the machine learning model "
    "will predict the emotion expressed in the text."
)

st.caption(
    "Supported emotion labels depend on the classes "
    "used to train your model."
)

user_text = st.text_area(
    "Enter your text",
    placeholder="Example: I am very happy today!",
    height=150
)

if st.button("Predict Emotion", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            model = load_model()
            text = clean_text(user_text)

            prediction = model.predict([text])[0]

            st.success(
                f"Predicted Emotion: {str(prediction).title()}"
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([text])[0]
                classes = model.classes_

                st.subheader("Prediction Scores")

                for emotion, score in sorted(
                    zip(classes, probabilities),
                    key=lambda item: item[1],
                    reverse=True
                ):
                    st.progress(
                        float(score),
                        text=f"{emotion}: {score:.1%}"
                    )

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Prediction failed: {error}")
