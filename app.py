import streamlit as st
import pandas as pd
import pickle


model = pickle.load(open("svm_model.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📊 Credit Default Prediction App")
st.subheader("Upload credit client data to check if they may default next month.")

uploaded_file = st.file_uploader("📁 Upload your CSV file here", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("📄 Raw Uploaded Data:")
        st.dataframe(data.head())

        data = data.drop(columns=["ID", "default.payment.next.month"], errors="ignore")

        if data.shape[1] != scaler.mean_.shape[0]:
            st.error("⚠️ Number of features does not match model input. Please check your file.")
        else:
            scaled_data = scaler.transform(data)
            pca_data = pca.transform(scaled_data)

            predictions = model.predict(pca_data)

            data["Prediction"] = predictions
            data["Prediction"] = data["Prediction"].map({0: "✅ Will Pay", 1: "❌ Default"})

            st.success("✅ Predictions completed!")
            st.write("🧾 Final Results:")
            st.dataframe(data)

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
