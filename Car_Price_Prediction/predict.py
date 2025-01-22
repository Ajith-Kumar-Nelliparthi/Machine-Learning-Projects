import json
import joblib
import streamlit as st
import pandas as pd

with open("columns_attributes.json", "r") as f_in:
    col_attrs = json.load(f_in)
with open("brand_models.json", "r") as f_in:
    options_mapping = json.load(f_in)
pipeline = joblib.load("pipeline.joblib")

st.title("Car Price Prediction")

input_data = {}
for col_name, col_info in col_attrs.items():
    attr_type = col_info[0]

    if attr_type == "category":
        options = col_info[1]
        if col_name == 'model':
            options = options_mapping[input_data['brand']]
        input_data[col_name] = st.selectbox(
            f"Select {col_name} (Category)", options
        )

    elif attr_type == "numeric":
        num_type = col_info[1]
        min_value = col_info[2]
        max_value = col_info[3]
        if num_type == "int64":
            input_data[col_name] = st.number_input(
                f"Enter {col_name}", step=1, format="%d",
                min_value=min_value,
                max_value=max_value,
                value=min_value
            )
        elif num_type == "float64":
            input_data[col_name] = st.number_input(
                f"Enter {col_name}", step=0.1, format="%.1f",
                min_value=min_value,
                max_value=max_value,
                value=min_value
            )

if st.button("Predict"):
    prediction = pipeline.predict(pd.DataFrame([input_data]))[0]
    st.markdown(
        f"<h2 style='text-align: center; color: blue;'>Estimated Price: ${prediction:,.2f}</h2>",
        unsafe_allow_html=True
    )
