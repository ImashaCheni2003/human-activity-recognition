
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "linear_svm_model.joblib"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "UCI HAR Dataset"

ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- HEADER ----------------
st.title("🏃 Human Activity Recognition")
st.markdown(
    """
    **Machine Learning powered activity classification**

    Explore how a Linear Support Vector Machine (SVM) classifies
    smartphone sensor measurements into six human activities.
    """
)

st.divider()

# ---------------- MODEL PERFORMANCE ----------------
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test Accuracy", "96.71%")

with col2:
    st.metric("Macro F1 Score", "96.74%")

with col3:
    st.metric("Activities Recognized", "6")

st.caption(
    "Performance measured on the official UCI HAR test set "
    "(2,947 records)."
)

# ---------------- NAVIGATION ----------------
overview_tab, predict_tab, evaluation_tab, data_tab = st.tabs(
    ["🏠 Overview", "🔍 Try a Prediction", "📈 Evaluation", "📂 Dataset"]
)

with overview_tab:
    st.subheader("About this project")
    st.write(
        """
        This project uses smartphone accelerometer and gyroscope
        features to recognize human physical activities.

        The Linear SVM model was selected after comparing it with
        Logistic Regression and Random Forest on the same official
        training and test split.
        """
    )

    st.markdown("**Activities supported**")
    st.write(", ".join(ACTIVITIES.values()))

    st.info(
        "The model uses the dataset's 561 engineered sensor features. "
        "It does not directly process raw smartphone signals."
    )

with predict_tab:
    st.subheader("Predict an activity from sensor features")

    st.write(
        "Upload a CSV containing exactly 561 feature columns, "
        "in the same order as the original UCI HAR feature dataset."
    )

    uploaded_file = st.file_uploader(
        "Upload sensor feature CSV",
        type=["csv"],
        key="prediction_csv",
    )

    if uploaded_file is not None:
        try:
            input_data = pd.read_csv(uploaded_file)

            if input_data.shape[1] != 561:
                st.error(
                    f"Expected 561 feature columns, but received "
                    f"{input_data.shape[1]}."
                )
            elif input_data.empty:
                st.error("The uploaded CSV contains no records.")
            elif not all(
                pd.api.types.is_numeric_dtype(dtype)
                for dtype in input_data.dtypes
            ):
                st.error("All 561 feature columns must be numeric.")
            elif input_data.isna().any().any():
                st.error("The CSV contains missing values.")
            else:
                predictions = model.predict(input_data)

                st.success("Prediction completed!")

                output = pd.DataFrame({
                    "Record": range(1, len(predictions) + 1),
                    "Predicted Activity": [
                        ACTIVITIES[int(value)]
                        for value in predictions
                    ],
                })

                st.dataframe(output, use_container_width=True)

                st.download_button(
                    "Download predictions",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="activity_predictions.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Could not process the uploaded file: {exc}")

with evaluation_tab:
    st.subheader("Model evaluation")

    comparison_path = RESULTS_DIR / "model_comparison.csv"
    matrix_path = RESULTS_DIR / "linear_svm_confusion_matrix.png"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

        st.write("Comparison of the three tested models")
        st.dataframe(
            comparison.style.format({
                "Accuracy": "{:.2%}",
                "Macro F1 Score": "{:.2%}",
            }),
            use_container_width=True,
        )

    if matrix_path.exists():
        st.write("Linear SVM confusion matrix")
        st.image(
            str(matrix_path),
            caption="Actual versus predicted activities",
            use_container_width=True,
        )

    st.warning(
        "SITTING had the lowest recall in our evaluation. "
        "Some sitting records were classified as standing."
    )

with data_tab:
    st.subheader("Dataset information")

    st.markdown(
        """
        - **Dataset:** UCI Human Activity Recognition Using Smartphones
        - **Training records:** 7,352
        - **Testing records:** 2,947
        - **Features per record:** 561
        - **Activity classes:** 6
        - **Final model:** Linear Support Vector Machine
        """
    )

    st.markdown(
        "[View the original UCI dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)"
    )

st.divider()
st.caption("Human Activity Recognition | Machine Learning Portfolio Project")
