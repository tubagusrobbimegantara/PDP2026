import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import io

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Optimisasi Distribusi Makan Bergizi",
    page_icon="🚚",
    layout="wide"
)

# --- KONFIGURASI VISUAL ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)

# --- FUNGSI ALGORITMA ---

def solve_routing_simple(kitchen_coord, school_coords):
    """
    Algoritma Greedy Nearest Neighbor untuk menentukan urutan kunjungan.
    Dari Dapur -> Sekolah Terdekat -> Sekolah Terdekat berikutnya -> ...
    """
    route_indices = []
    current_pos = kitchen_coord.reshape(1, -1)
    
    # Copy koordinat sekolah agar tidak merusak data asli
    remaining_schools = school_coords.copy()
    original_indices = np.arange(len(school_coords)) # Melacak indeks asli
    
    # Masking untuk sekolah yang sudah dikunjungi
    visited_mask = np.zeros(len(school_coords), dtype=bool)
    
    for _ in range(len(school_coords)):
        # Hitung jarak dari posisi sekarang ke semua sekolah
        dists = cdist(current_pos, remaining_schools, metric='euclidean')
        
        # Set jarak ke infinity untuk sekolah yang sudah dikunjungi
        dists[0, visited_mask] = np.inf
        
        # Cari sekolah terdekat yang belum dikunjungi
        nearest_idx = np.argmin(dists)
        
        # Simpan rute
        route_indices.append(nearest_idx)
        visited_mask[nearest_idx] = True
        
        # Update posisi sekarang menjadi sekolah yang baru dikunjungi
        current_pos = remaining_schools[nearest_idx].reshape(1, -1)
        
    return route_indices

# --- UI UTAMA ---

st.title("🚚 Sistem Penentuan Rute")
st.markdown("""
**Input:** Data Excel Sekolah (Lat, Lon) | **Proses:** Clustering & VRP Routing | **Output:** Kebijakan Distribusi
""")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Pengaturan Operasional")
n_clusters = st.sidebar.slider("Jumlah Dapur / Klaster", 2, 20, 5)
st.sidebar.info("Aplikasi akan menentukan lokasi Dapur secara otomatis di titik tengah (Centroid) klaster sekolah.")

# --- TABS ---
tab_input, tab_process, tab_output = st.tabs([
    "📂 1. Input Data", 
    "🗺️ 2. Peta Rute & Klaster", 
    "📋 3. Output Kebijakan (Excel)"
])

# --- GLOBAL VAR ---
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = None

# --- TAB 1: INPUT ---
with tab_input:
    st.subheader("Unggah Data Sekolah")
    
    # Template Download
    dummy_data = pd.DataFrame({
        'Nama_Sekolah': ['SDN 1 Bandung', 'SDN 2 Bandung', 'SMP 1 Bandung', 'SDN 3 Bandung'],
        'Latitude': [-6.9175, -6.9200, -6.9150, -6.9180],
        'Longitude': [107.6191, 107.6200, 107.6180, 107.6210],
        'Jumlah_Siswa': [200, 150, 300, 100]
    })
    
    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        st.write("Belum punya format?")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            dummy_data.to_excel(writer, index=False, sheet_name='Sheet1')
        
        st.download_button(
            label="⬇️ Download Template Excel",
            data=buffer.getvalue(),
            file_name="template_sekolah.xlsx",
            mime="application/vnd.ms-excel"
        )

    uploaded_file = st.file_uploader("Upload file Excel Anda", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            # Validasi kolom dasar
            required_cols = ['Latitude', 'Longitude', 'Nama_Sekolah']
            if not all(col in df.columns for col in required_cols):
                st.error(f"File Excel wajib memiliki kolom: {required_cols}")
            else:
                st.session_state.data_uploaded = df
                st.success(f"Berhasil memuat {len(df)} data sekolah.")
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

# --- PROSES UTAMA (JIKA DATA ADA) ---
if st.session_state.data_uploaded is not None:
    df = st.session_state.data_uploaded.copy()
    
    # 1. CLUSTERING (Menentukan Dapur)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster_ID'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
    
    # Ambil koordinat dapur (Centroids)
    kitchen_centers = kmeans.cluster_centers_
    
    # 2. ROUTING (Menentukan Urutan)
    # Kita akan membuat DataFrame baru untuk hasil rute
    route_results = []
    
    for c_id in range(n_clusters):
        # Filter sekolah di klaster ini
        cluster_schools = df[df['Cluster_ID'] == c_id].copy().reset_index(drop=True)
        kitchen_loc = kitchen_centers[c_id]
        
        # Hitung rute (urutan indeks)
        school_coords = cluster_schools[['Latitude', 'Longitude']].values
        route_order = solve_routing_simple(kitchen_loc, school_coords)
        
        # Masukkan ke hasil dengan urutan
        for seq_num, idx in enumerate(route_order):
            school = cluster_schools.iloc[idx]
            route_results.append({
                'ID_Dapur': c_id + 1,
                'Lokasi_Dapur_Lat': kitchen_loc[0],
                'Lokasi_Dapur_Lon': kitchen_loc[1],
                'Urutan_Pengiriman': seq_num + 1,
                'Nama_Sekolah': school['Nama_Sekolah'],
                'Lat_Sekolah': school['Latitude'],
                'Lon_Sekolah': school['Longitude'],
                'Jumlah_Siswa': school.get('Jumlah_Siswa', 0) # Handle jika kolom tidak ada
            })
            
    df_routes = pd.DataFrame(route_results)

    # --- TAB 2: VISUALISASI PETA ---
    with tab_process:
        st.subheader("Peta Rute Distribusi & Lokasi Dapur")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Warna untuk setiap klaster
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            for c_id in range(n_clusters):
                # Data Rute per Dapur
                dapur_data = df_routes[df_routes['ID_Dapur'] == c_id + 1].sort_values('Urutan_Pengiriman')
                
                kitchen_lat = dapur_data.iloc[0]['Lokasi_Dapur_Lat']
                kitchen_lon = dapur_data.iloc[0]['Lokasi_Dapur_Lon']
                color = colors[c_id]
                
                # 1. Plot Titik Dapur (Bintang Besar)
                ax.scatter(kitchen_lon, kitchen_lat, s=300, marker='*', c=[color], edgecolors='black', zorder=10, label=f'Dapur {c_id+1}')
                ax.text(kitchen_lon, kitchen_lat, f" Dapur {c_id+1}", fontsize=9, fontweight='bold')
                
                # 2. Plot Titik Sekolah
                ax.scatter(dapur_data['Lon_Sekolah'], dapur_data['Lat_Sekolah'], s=50, c=[color], alpha=0.7)
                
                # 3. Plot Garis Rute (LineString)
                # Dari Dapur ke Sekolah Pertama
                ax.plot([kitchen_lon, dapur_data.iloc[0]['Lon_Sekolah']], 
                        [kitchen_lat, dapur_data.iloc[0]['Lat_Sekolah']], 
                        c=color, linestyle='--', alpha=0.5)
                
                # Antar Sekolah (Sesuai Urutan)
                ax.plot(dapur_data['Lon_Sekolah'], dapur_data['Lat_Sekolah'], c=color, alpha=0.6, linewidth=1.5)
                
                # Kembali ke Dapur (Opsional, biasanya mobil balik)
                ax.plot([dapur_data.iloc[-1]['Lon_Sekolah'], kitchen_lon], 
                        [dapur_data.iloc[-1]['Lat_Sekolah'], kitchen_lat], 
                        c=color, linestyle=':', alpha=0.3)

            ax.set_title("Visualisasi Rute VRP (Dapur -> Sekolah)", fontsize=15)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            st.pyplot(fig)
            
        with col2:
            st.info("Keterangan Peta:")
            st.markdown("""
            * **Bintang (⭐):** Lokasi Dapur Sementara (Pusat Klaster).
            * **Titik Bulat:** Lokasi Sekolah.
            * **Garis:** Rute perjalanan kendaraan.
            """)
            st.metric("Total Dapur", n_clusters)
            st.metric("Total Sekolah", len(df))

    # --- TAB 3: OUTPUT KEBIJAKAN ---
    with tab_output:
        st.subheader("📋 Kebijakan Operasional Distribusi")
        
        st.write("Tabel ini menunjukkan Dapur mana yang bertanggung jawab atas sekolah mana, serta urutan pengirimannya untuk efisiensi waktu.")
        
        # Tampilkan Dataframe Rapi
        display_cols = ['ID_Dapur', 'Urutan_Pengiriman', 'Nama_Sekolah', 'Jumlah_Siswa', 'Lat_Sekolah', 'Lon_Sekolah']
        st.dataframe(df_routes[display_cols], height=400)
        
        # Download Button
        buffer_out = io.BytesIO()
        with pd.ExcelWriter(buffer_out, engine='xlsxwriter') as writer:
            df_routes.to_excel(writer, index=False, sheet_name='Rute_Kebijakan')
            
        st.download_button(
            label="⬇️ Download Kebijakan (Excel)",
            data=buffer_out.getvalue(),
            file_name="Kebijakan_Rute_VRP.xlsx",
            mime="application/vnd.ms-excel"
        )
        
        # Statistik per Dapur
        st.markdown("### Beban Kerja per Dapur")
        beban_kerja = df_routes.groupby('ID_Dapur')['Jumlah_Siswa'].sum().reset_index()
        beban_kerja['Jumlah_Sekolah'] = df_routes.groupby('ID_Dapur')['Nama_Sekolah'].count().values
        
        col_metrics = st.columns(n_clusters)
        for idx, row in beban_kerja.iterrows():
            with col_metrics[idx % 3]: # Agar tidak terlalu lebar jika cluster banyak
                st.metric(f"Dapur {row['ID_Dapur']}", f"{row['Jumlah_Siswa']:,} Siswa", f"{row['Jumlah_Sekolah']} Sekolah")

else:
    # Pesan jika belum ada data
    with tab_process:
        st.info("Silakan unggah data Excel di Tab 1 terlebih dahulu.")
    with tab_output:
        st.info("Silakan unggah data Excel di Tab 1 terlebih dahulu.")
