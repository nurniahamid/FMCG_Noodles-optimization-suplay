# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = joblib.load('model_rf.pkl')  # Pastikan model disimpan dengan nama ini

st.set_page_config(page_title="Prediksi Tonase Produk Gudang", layout="centered")
st.title("📦 Prediksi Berat Produk ke Gudang")
st.write("Masukkan data gudang untuk memprediksi berat produk yang dikirim (dalam ton)")

# Form input
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        storage_issue = st.number_input("Storage Issue Reported (3 Bulan Terakhir)", min_value=0, max_value=50, value=10)
        warehouse_age = st.number_input("Usia Gudang (tahun)", min_value=0, max_value=50, value=15)
        breakdown = st.number_input("Jumlah Kerusakan Gudang (3 Bulan Terakhir)", min_value=0, max_value=20, value=3)

    with col2:
        temp_reg_mach = st.selectbox("Ada Mesin Pengatur Suhu?", options={0: 'Tidak', 1: 'Ya'})
        location_type = st.selectbox("Tipe Lokasi Gudang", options={0: 'Rural', 1: 'Urban'})
        certificate = st.selectbox("Jenis Sertifikat Gudang", options={0: 'A+', 1: 'A', 2: 'B', 3: 'C', 4: 'Unknown'})

    submitted = st.form_submit_button("Prediksi")

# Prediksi satuan
if submitted:
    input_data = pd.DataFrame({
        'storage_issue_reported_l3m': [storage_issue],
        'warehouse_age': [warehouse_age],
        'wh_breakdown_l3m': [breakdown],
        'temp_reg_mach': [int(temp_reg_mach)],
        'Location_type': [int(location_type)],
        'approved_wh_govt_certificate': [int(certificate)]
    })

    pred = model.predict(input_data)[0]
    st.success(f"🎯 Prediksi Berat Produk: {pred:,.0f} ton")

    st.markdown("---")
    st.caption("Model: Random Forest Regressor | Akurasi R²: 99.3%")

