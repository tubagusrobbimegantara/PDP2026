import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import io

# --- 1. KONFIGURASI HALAMAN & TEMA ---
st.set_page_config(
    page_title="MBG-Route - Smart Logistics",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (UNTUK MEMPERCANTIK) ---
st.markdown("""
<style>
    /* Import Font Keren */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Header Gradient */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin: 0;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 5px 0 0 0;
    }

    /* Metric Cards Custom */
    div.metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        text-align: center;
    }
    div.metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: #2a5298;
    }
    .metric-title {
        color: #757575;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #1e3c72;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-top: 3px solid #1e3c72;
        color: #1e3c72;
        font-weight: bold;
    }

    /* Tombol Download */
    .stDownloadButton button {
        background-color: #2ECC71 !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(46, 204, 113, 0.3);
    }
    .stDownloadButton button:hover {
        background-color: #27ae60 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. KONFIGURASI VISUAL SEABORN ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1

# --- 4. FUNGSI LOGIKA (BACKEND) ---

def solve_routing_simple(kitchen_coord, school_coords):
    """Greedy Nearest Neighbor Algorithm"""
    route_indices = []
    current_pos = kitchen_coord.reshape(1, -1)
    remaining_schools = school_coords.copy()
    visited_mask = np.zeros(len(school_coords), dtype=bool)
    
    for _ in range(len(school_coords)):
        dists = cdist(current_pos, remaining_schools, metric='euclidean')
        dists[0, visited_mask] = np.inf
        nearest_idx = np.argmin(dists)
        
        route_indices.append(nearest_idx)
        visited_mask[nearest_idx] = True
        current_pos = remaining_schools[nearest_idx].reshape(1, -1)
        
    return route_indices

# --- 5. UI UTAMA (FRONTEND) ---

# Header Section
st.markdown("""
<div class="main-header">
    <h1>🍱 MBG-Route: Sistem Distribusi Cerdas</h1>
    <p>Optimalisasi Rute Program Makan Bergizi Gratis Berbasis Spasial & VRP</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/delivery--v1.png", width=80)
    st.markdown("### ⚙️ Panel Kontrol")
    st.markdown("---")
    n_clusters = st.slider("Jumlah Dapur/Rayon", 2, 20, 5, help="Sistem akan membagi wilayah menjadi sekian klaster.")
    st.info("""
    **Cara Kerja:**
    1. Upload Excel Data Sekolah.
    2. Sistem menghitung titik tengah (Dapur).
    3. Sistem membuat rute terpendek.
    4. Download Kebijakan.
    """)
    st.markdown("---")
    st.caption("© 2025 MBG-Route Research Team")

# Global State
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = None

# Tabs
tab1, tab2, tab3 = st.tabs(["📂 1. Input Data", "🗺️ 2. Analisis Peta", "📋 3. Output Kebijakan"])

# --- TAB 1: INPUT ---
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        st.markdown("### 📥 Unggah Data")
        st.write("Silakan unduh template di bawah, isi data sekolah, lalu unggah kembali.")
        
        # Template
        dummy_data = pd.DataFrame({
            'Nama_Sekolah': ['SDN 1 Merdeka', 'SDN 2 Juara', 'SMP 1 Bangsa', 'SDN 3 Cerdas'],
            'latitude': [-6.9175, -6.9200, -6.9150, -6.9180],
            'longitude': [107.6191, 107.6200, 107.6180, 107.6210],
            'Jumlah_Siswa': [200, 150, 300, 100]
        })
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            dummy_data.to_excel(writer, index=False, sheet_name='Data')
            
        st.download_button("⬇️ Download Template Excel", data=buffer.getvalue(), file_name="template_MBG-Route.xlsx")
        
        st.markdown("---")
        uploaded_file = st.file_uploader("", type=['xlsx'], help="Pastikan format sesuai template")

    with col_right:
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.session_state.data_uploaded = df
                st.success(f"✅ Data berhasil dimuat: {len(df)} Sekolah")
                
                # Preview Data dengan Style
                st.dataframe(
                    df.head(), 
                    use_container_width=True,
                    column_config={
                        "latitude": st.column_config.NumberColumn(format="%.5f"),
                        "longitude": st.column_config.NumberColumn(format="%.5f"),
                    }
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.info("👈 Menunggu file Excel diunggah...")
            # Placeholder Image
            st.markdown("""
            <div style="text-align: center; color: #ccc; padding: 50px; border: 2px dashed #ccc; border-radius: 10px;">
                <h3>Area Preview Data</h3>
                <p>Data akan muncul di sini setelah diunggah</p>
            </div>
            """, unsafe_allow_html=True)

# --- LOGIKA PROSES ---
if st.session_state.data_uploaded is not None:
    df = st.session_state.data_uploaded.copy()
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster_ID'] = kmeans.fit_predict(df[['latitude', 'longitude']])
    kitchen_centers = kmeans.cluster_centers_
    
    # Routing
    route_results = []
    for c_id in range(n_clusters):
        cluster_schools = df[df['Cluster_ID'] == c_id].copy().reset_index(drop=True)
        kitchen_loc = kitchen_centers[c_id]
        
        route_order = solve_routing_simple(kitchen_loc, cluster_schools[['latitude', 'longitude']].values)
        
        for seq_num, idx in enumerate(route_order):
            school = cluster_schools.iloc[idx]
            route_results.append({
                'ID_Dapur': c_id + 1,
                'Lokasi_Dapur_Lat': kitchen_loc[0],
                'Lokasi_Dapur_Lon': kitchen_loc[1],
                'Urutan_Pengiriman': seq_num + 1,
                'Nama_Sekolah': school['Nama_Sekolah'],
                'Lat_Sekolah': school['latitude'],
                'Lon_Sekolah': school['longitude'],
                'Jumlah_Siswa': school.get('Jumlah_Siswa', 0)
            })
    df_routes = pd.DataFrame(route_results)

    # --- TAB 2: PETA ---
    with tab2:
        st.markdown("### 🗺️ Visualisasi Rute & Klaster")
        
        col_map, col_legend = st.columns([3, 1])
        
        with col_map:
            with st.spinner("Membuat Peta Interaktif..."):
                fig, ax = plt.subplots(figsize=(14, 9))
                # Custom Palette
                palette = sns.color_palette("bright", n_clusters)
                
                # Plot Background Grid lebih halus
                ax.grid(True, linestyle='--', alpha=0.3)
                
                for c_id in range(n_clusters):
                    dapur_data = df_routes[df_routes['ID_Dapur'] == c_id + 1].sort_values('Urutan_Pengiriman')
                    k_lat, k_lon = dapur_data.iloc[0]['Lokasi_Dapur_Lat'], dapur_data.iloc[0]['Lokasi_Dapur_Lon']
                    color = palette[c_id]
                    
                    # Rute Lines
                    # Dapur -> Sekolah 1
                    ax.plot([k_lon, dapur_data.iloc[0]['Lon_Sekolah']], 
                            [k_lat, dapur_data.iloc[0]['Lat_Sekolah']], 
                            c=color, linestyle='--', alpha=0.4)
                    # Sekolah -> Sekolah
                    ax.plot(dapur_data['Lon_Sekolah'], dapur_data['Lat_Sekolah'], c=color, linewidth=2, alpha=0.8, label=f'Rute Dapur {c_id+1}')
                    
                    # Titik Sekolah
                    ax.scatter(dapur_data['Lon_Sekolah'], dapur_data['Lat_Sekolah'], s=80, c=[color], edgecolors='white', alpha=0.9)
                    
                    # Titik Dapur (Penting)
                    ax.scatter(k_lon, k_lat, s=400, marker='*', c=[color], edgecolors='black', zorder=10)
                    ax.text(k_lon, k_lat+0.0005, f"Dapur {c_id+1}", fontsize=10, fontweight='bold', ha='center',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

                ax.set_xlabel("longitude")
                ax.set_ylabel("latitude")
                ax.set_title("Peta Operasional Distribusi", fontsize=16, pad=20)
                sns.despine()
                st.pyplot(fig)
        
        with col_legend:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #1e3c72;">
                <h4>Legenda Peta</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>⭐ <b>Bintang Besar:</b> Lokasi Dapur Sementara (Pusat Rayon)</li>
                    <li>● <b>Lingkaran:</b> Sekolah Tujuan</li>
                    <li>━ <b>Garis Warna:</b> Rute Kendaraan</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Mini Summary
            total_siswa = df_routes['Jumlah_Siswa'].sum()
            st.metric("Total Siswa Dilayani", f"{total_siswa:,}")
            st.metric("Rata-rata Sekolah/Dapur", f"{len(df)/n_clusters:.1f}")

    # --- TAB 3: OUTPUT ---
    with tab3:
        st.markdown("### 📋 Dasbor Kebijakan Distribusi")
        
        # 1. Download Section
        col_dl_desc, col_dl_btn = st.columns([3, 1])
        with col_dl_desc:
            st.write("Unduh file Excel ini untuk diberikan kepada manajer operasional dan supir logistik.")
        with col_dl_btn:
            buffer_out = io.BytesIO()
            with pd.ExcelWriter(buffer_out, engine='xlsxwriter') as writer:
                df_routes.to_excel(writer, index=False, sheet_name='Kebijakan')
            st.download_button("⬇️ Download Excel Kebijakan", data=buffer_out.getvalue(), file_name="Kebijakan_Distribusi_Final.xlsx")
        
        st.markdown("---")
        
        # 2. Tabel Data Rinci
        with st.expander("🔍 Lihat Detail Tabel Jadwal Pengiriman", expanded=True):
            st.dataframe(
                df_routes[['ID_Dapur', 'Urutan_Pengiriman', 'Nama_Sekolah', 'Jumlah_Siswa']],
                use_container_width=True,
                hide_index=True
            )

        # 3. KARTU BEBAN KERJA (Custom HTML/CSS)
        st.markdown("#### 📊 Beban Kerja per Dapur (Workload)")
        
        beban_kerja = df_routes.groupby('ID_Dapur').agg({'Jumlah_Siswa':'sum', 'Nama_Sekolah':'count'}).reset_index()
        
        # Grid Layout untuk Kartu
        cols = st.columns(4) # 4 Kartu per baris
        
        for index, row in beban_kerja.iterrows():
            with cols[index % 4]:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-icon">🍳</div>
                    <div class="metric-title">Dapur {row['ID_Dapur']}</div>
                    <div class="metric-value">{row['Jumlah_Siswa']:,}</div>
                    <div style="font-size: 0.8rem; color: #888;">
                        Melayani <b>{row['Nama_Sekolah']}</b> Sekolah
                    </div>
                </div>
                <br>
                """, unsafe_allow_html=True)
