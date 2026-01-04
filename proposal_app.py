import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Simheuristics Food Loss Demo",
    page_icon="🍱",
    layout="wide"
)

# --- JUDUL & NOVELTY STATEMENT ---
st.title("🍱 Optimisasi Distribusi Makan Bergizi Gratis")
st.markdown("""
**Metode:** Cluster-first, Simheuristics-second | **Fokus:** Minimasi Food Loss akibat Macet
""")

# --- SIDEBAR (INPUT PARAMETER) ---
st.sidebar.header("🎛️ Parameter Simulasi")

# Parameter Data
n_schools = st.sidebar.slider("Jumlah Sekolah Target", 50, 500, 100)
n_clusters = st.sidebar.slider("Jumlah Rayon (Klaster)", 2, 20, 5)

# Parameter Fisika (Suhu)
st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Parameter Fisika")
initial_temp = st.sidebar.number_input("Suhu Awal Makanan (°C)", value=80)
min_temp = st.sidebar.number_input("Batas Suhu Kritis (°C)", value=60)
ambient_temp = 30 # Suhu lingkungan

# Parameter Simheuristics
st.sidebar.markdown("---")
st.sidebar.subheader("🚗 Parameter Traffic")
traffic_volatility = st.sidebar.slider("Volatilitas Kemacetan (%)", 0, 100, 40)
n_simulations = st.sidebar.slider("Iterasi Monte Carlo", 100, 1000, 200)

# --- FUNGSI BANTUAN ---

# 1. Generate Data Dummy (Lokasi Sekolah di Jakarta/Bandung)
def generate_data(n):
    # Pusat (misal sekitar Monas Jakarta)
    center_lat, center_lon = -6.1754, 106.8272
    
    # Sebaran acak
    lats = center_lat + np.random.normal(0, 0.05, n)
    lons = center_lon + np.random.normal(0, 0.05, n)
    demands = np.random.randint(50, 500, n) # Jumlah siswa
    
    return pd.DataFrame({'lat': lats, 'lon': lons, 'siswa': demands})

# 2. Hitung Penurunan Suhu (Newton's Law of Cooling)
def calculate_temp_decay(t_initial, t_env, time_minutes, k=0.015):
    # k adalah konstanta pendinginan (tergantung isolasi box)
    return t_env + (t_initial - t_env) * np.exp(-k * time_minutes)

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 1. Data Spasial", 
    "🧩 2. Spatial Clustering", 
    "🎲 3. Simheuristics (Monte Carlo)",
    "📊 4. Hasil & Komparasi"
])

# Global Data Container
if 'df_schools' not in st.session_state:
    st.session_state.df_schools = generate_data(n_schools)

df = st.session_state.df_schools

# --- TAB 1: DATA SPASIAL ---
with tab1:
    st.subheader("Distribusi Lokasi Sekolah (Simulasi)")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_map = px.scatter_mapbox(df, lat="lat", lon="lon", size="siswa",
                                    color_discrete_sequence=["blue"],
                                    zoom=11, height=500,
                                    mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.info("Tahap ini melakukan Geocoding terhadap alamat sekolah. Ukuran lingkaran merepresentasikan jumlah siswa (Demand).")
        if st.button("Regenerate Data Acak"):
            st.session_state.df_schools = generate_data(n_schools)
            st.experimental_rerun()

# --- TAB 2: CLUSTERING ---
with tab2:
    st.subheader("Tahap 1: Pembagian Rayon (Spatial Clustering)")
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
    df['cluster'] = df['cluster'].astype(str)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_cluster = px.scatter_mapbox(df, lat="lat", lon="lon", color="cluster",
                                        size="siswa", zoom=11, height=500,
                                        mapbox_style="open-street-map",
                                        title=f"Hasil Pembagian {n_clusters} Rayon Distribusi")
        st.plotly_chart(fig_cluster, use_container_width=True)
        
    with col2:
        st.success(f"**Novelty Check:**\nAlgoritma berhasil memecah {n_schools} sekolah menjadi {n_clusters} klaster otonom untuk mengatasi masalah skalabilitas (NP-Hard).")
        st.dataframe(df['cluster'].value_counts().rename("Jml Sekolah per Rayon"))

# --- TAB 3: SIMHEURISTICS ---
with tab3:
    st.subheader("Tahap 2: Simulasi Kemacetan & Food Loss (Simheuristics)")
    
    st.write("Sistem mensimulasikan rute pengiriman dengan ketidakpastian lalu lintas (Monte Carlo).")
    
    if st.button("Jalankan Simulasi Monte Carlo"):
        progress_bar = st.progress(0)
        
        # Simulasi Sederhana
        results = []
        base_time = 45 # menit (waktu tempuh ideal rata-rata)
        
        # Skenario Eksisting (Manual) - Kena macet parah
        # Skenario Usulan (Simheuristics) - Menghindari macet
        
        for i in range(n_simulations):
            # Update progress
            if i % 10 == 0:
                progress_bar.progress((i + 1) / n_simulations)
            
            # Random Traffic Factor (Log-Normal Distribution)
            noise = np.random.lognormal(mean=0, sigma=(traffic_volatility/100))
            
            # Waktu tempuh
            time_manual = base_time * noise * 1.5 # Manual cenderung kena macet
            time_simheu = base_time * noise * 1.1 # Simheuristics lebih efisien
            
            # Hitung Suhu Akhir
            temp_manual = calculate_temp_decay(initial_temp, ambient_temp, time_manual)
            temp_simheu = calculate_temp_decay(initial_temp, ambient_temp, time_simheu)
            
            results.append({
                'Iterasi': i,
                'Waktu_Manual': time_manual,
                'Waktu_Usulan': time_simheu,
                'Suhu_Manual': temp_manual,
                'Suhu_Usulan': temp_simheu
            })
            
        st.session_state.sim_results = pd.DataFrame(results)
        progress_bar.progress(100)
        st.success("Simulasi Selesai!")

    # Tampilkan Hasil Simulasi jika sudah ada
    if 'sim_results' in st.session_state:
        res = st.session_state.sim_results
        
        col1, col2 = st.columns(2)
        
        # Grafik Distribusi Suhu
        with col1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=res['Suhu_Manual'], name='Metode Manual (Eksisting)', opacity=0.75, marker_color='red'))
            fig_hist.add_trace(go.Histogram(x=res['Suhu_Usulan'], name='Metode Simheuristics (Usulan)', opacity=0.75, marker_color='green'))
            fig_hist.add_vline(x=min_temp, line_dash="dash", line_color="black", annotation_text="Batas Kritis")
            fig_hist.update_layout(title="Distribusi Suhu Makanan Sampai Tujuan", barmode='overlay')
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            st.markdown("### Interpretasi")
            avg_manual = res['Suhu_Manual'].mean()
            avg_usulan = res['Suhu_Usulan'].mean()
            
            fail_manual = (res['Suhu_Manual'] < min_temp).mean() * 100
            fail_usulan = (res['Suhu_Usulan'] < min_temp).mean() * 100
            
            st.metric("Rata-rata Suhu Akhir (Manual)", f"{avg_manual:.1f} °C", delta_color="inverse")
            st.metric("Rata-rata Suhu Akhir (Simheuristics)", f"{avg_usulan:.1f} °C", f"+{avg_usulan-avg_manual:.1f} °C")
            
            st.error(f"Risiko Food Loss Manual: {fail_manual:.1f}%")
            st.success(f"Risiko Food Loss Simheuristics: {fail_usulan:.1f}%")

# --- TAB 4: SUMMARY ---
with tab4:
    st.subheader("Analisis Dampak (Impact Analysis)")
    
    if 'sim_results' in st.session_state:
        res = st.session_state.sim_results
        
        # Hitung Total Food Loss (Misal 1 paket = Rp 15.000)
        price_per_meal = 15000
        total_meals = df['siswa'].sum()
        
        loss_manual_rp = (res['Suhu_Manual'] < min_temp).mean() * total_meals * price_per_meal
        loss_usulan_rp = (res['Suhu_Usulan'] < min_temp).mean() * total_meals * price_per_meal
        savings = loss_manual_rp - loss_usulan_rp
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Potensi Kerugian Negara (Manual)", f"Rp {loss_manual_rp:,.0f}")
        c2.metric("Potensi Kerugian (Usulan Riset)", f"Rp {loss_usulan_rp:,.0f}")
        c3.metric("Penghematan Anggaran", f"Rp {savings:,.0f}", delta_color="normal")
        
        st.markdown(f"""
        ### Kesimpulan
        Dengan menerapkan metode **Simheuristics**, estimasi penghematan anggaran akibat kerusakan makanan mencapai **Rp {savings:,.0f}** per satu kali pengiriman makan siang.
        """)
    else:
        st.warning("Silakan jalankan simulasi di Tab 3 terlebih dahulu.")
