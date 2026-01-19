import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# APP CONFIG & HELPER FUNCTIONS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Thermal-to-CAPEX (EarthSavvy Add-on)",
    page_icon="🔥",
    layout="wide"
)

# Constants for Physics
STEFAN_BOLTZMANN = 5.67e-8  # W/m2/K4

def flatten_earthsavvy_json(file_content, filename):
    """
    Parses EarthSavvy JSON format:
    { "site_key": { "name": "...", "data": [[ts, val1, val2], ...] } }
    Returns a DataFrame.
    """
    try:
        data = json.loads(file_content)
        records = []
        
        # Handle dict structure
        if isinstance(data, dict):
            for site_key, site_content in data.items():
                site_name = site_content.get('name', site_key)
                raw_rows = site_content.get('data', [])
                
                for row in raw_rows:
                    # Basic canonical record
                    record = {
                        'source_file': filename,
                        'site_id': site_key,
                        'site_name': site_name,
                        'raw_data': row  # Store full list to extract columns later
                    }
                    
                    # Attempt to parse timestamp from first column
                    if row and len(row) > 0:
                        try:
                            # Try standard EarthSavvy format "YYYY-MM-DD HH:MM"
                            ts_str = str(row[0])
                            ts = pd.to_datetime(ts_str, errors='coerce')
                            if pd.notnull(ts):
                                record['timestamp'] = ts
                        except:
                            record['timestamp'] = None
                    
                    records.append(record)
                    
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Error parsing {filename}: {e}")
        return pd.DataFrame()

def estimate_heat_loss(row, temp_col_idx, area_col_idx, params):
    """
    Physics engine for heat loss estimation.
    """
    try:
        # Extract data from raw list
        data_row = row['raw_data']
        
        # Get Surface Temp (T_s)
        if temp_col_idx is not None and len(data_row) > temp_col_idx:
            t_surf_input = float(data_row[temp_col_idx])
        else:
            return None # Cannot calc without temp
            
        # Get Area (A)
        if area_col_idx is not None and len(data_row) > area_col_idx:
            area = float(data_row[area_col_idx])
        else:
            area = params['default_area']

        # Unit conversion
        t_surf_c = t_surf_input
        t_amb_c = params['t_ambient']
        
        # Temperature in Kelvin
        tk_surf = t_surf_c + 273.15
        tk_amb = t_amb_c + 273.15
        
        # 1. Convection: q_conv = h * (T_surf - T_amb)
        # Ensure T_surf > T_amb for loss, otherwise 0 (or gain, but focused on loss here)
        delta_t = max(0, t_surf_c - t_amb_c)
        q_conv_flux = params['h_conv'] * delta_t # W/m2
        
        # 2. Radiation: q_rad = eps * sigma * (T_surf^4 - T_amb^4)
        q_rad_flux = params['emissivity'] * STEFAN_BOLTZMANN * (tk_surf**4 - tk_amb**4)
        q_rad_flux = max(0, q_rad_flux)
        
        total_flux = q_conv_flux + q_rad_flux # W/m2
        total_power = total_flux * area # W (Joules/sec)
        
        # Annualisation
        # Power (kW) * Hours/Year * Duty Cycle
        power_kw = total_power / 1000.0
        annual_kwh = power_kw * 8760 * params['duty_cycle']
        
        # Cost and Carbon
        annual_cost = annual_kwh * params['tariff_gbp_per_kwh']
        annual_carbon = annual_kwh * params['carbon_factor']
        
        return {
            'temp_c': t_surf_c,
            'area_m2': area,
            'flux_w_m2': total_flux,
            'power_kw': power_kw,
            'annual_kwh': annual_kwh,
            'annual_cost_gbp': annual_cost,
            'annual_tco2e': annual_carbon / 1000.0, # kg to tonnes
            'classification': classify_hotspot(delta_t, area)
        }
    except Exception as e:
        return None

def classify_hotspot(delta_t, area):
    """
    Simple heuristic rule-based classification.
    """
    if delta_t > 50:
        return "Likely Process Hotspot (High T)"
    elif delta_t > 10 and area > 50:
        return "Likely Fabric Loss (Large Area)"
    elif delta_t > 10:
        return "Minor Leak / Thermal Bridge"
    else:
        return "Low/Background Signal"

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_json(df):
    return df.to_json(orient='records', date_format='iso')

# -----------------------------------------------------------------------------
# MAIN APP UI
# -----------------------------------------------------------------------------

st.title("🏭 Thermal-to-CAPEX Estimate Tool")
st.markdown("""
**EarthSavvy Add-on** | *Screening-Grade Energy & Cost Estimation*

This tool transforms raw thermal or detection time-series data into actionable CAPEX business cases.
It calculates heat loss using physics-based approximations (Convection + Radiation) and estimates annual financial and environmental impact.
""")

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.header("🌍 Global Parameters")
st.sidebar.markdown("Define baseline assumptions for the physics model.")

with st.sidebar.expander("Physics Constants", expanded=True):
    p_t_amb = st.number_input("Ambient Temp (°C)", value=10.0, step=1.0, help="Baseline air temperature.")
    p_emis = st.number_input("Surface Emissivity (0-1)", value=0.9, step=0.05, help="0.9 for brick/concrete, 0.1 for polished metal.")
    p_h_conv = st.number_input("Conv. Coeff (h) [W/m²K]", value=10.0, step=1.0, help="10 for calm air, 25 for windy.")

with st.sidebar.expander("Economic & Operational", expanded=True):
    p_tariff = st.number_input("Energy Tariff (£/kWh)", value=0.15, format="%.3f")
    p_carbon = st.number_input("Carbon Factor (kgCO2e/kWh)", value=0.193, format="%.3f", help="UK Grid Average")
    p_duty = st.slider("Duty Cycle (Scaling)", 0.1, 1.0, 1.0, help="Fraction of year the heat loss occurs (1.0 = 24/7/365).")

p_default_area = st.sidebar.number_input("Default Area (m²)", value=10.0, help="Used if no area column is mapped.")

# Store params in dict
model_params = {
    't_ambient': p_t_amb,
    'emissivity': p_emis,
    'h_conv': p_h_conv,
    'tariff_gbp_per_kwh': p_tariff,
    'carbon_factor': p_carbon,
    'duty_cycle': p_duty,
    'default_area': p_default_area
}

# --- TABS ---
tab_upload, tab_map, tab_analysis, tab_export = st.tabs(["📂 Upload & Data", "⚙️ Map Columns", "📊 Results & CAPEX", "💾 Export"])

# =============================================================================
# TAB 1: UPLOAD & DETECT
# =============================================================================
with tab_upload:
    st.subheader("Import EarthSavvy Data")
    
    # 1. Scan Local Directory
    st.write("### 1. Local File Scan")
    local_files = [f for f in os.listdir('.') if f.endswith(('.json', '.csv'))]
    selected_local_files = st.multiselect("Select files found in folder:", local_files)
    
    # 2. Upload via UI
    st.write("### 2. Upload Files")
    uploaded_files = st.file_uploader("Drop JSON/CSV files here", accept_multiple_files=True, type=['json', 'csv'])
    
    all_records = []
    
    # Process Local Files
    for fname in selected_local_files:
        try:
            with open(fname, 'r') as f:
                content = f.read()
                if fname.endswith('.json'):
                    df_local = flatten_earthsavvy_json(content, fname)
                    if not df_local.empty:
                        all_records.append(df_local)
        except Exception as e:
            st.warning(f"Could not read local file {fname}: {e}")

    # Process Uploaded Files
    if uploaded_files:
        for uf in uploaded_files:
            content = uf.read().decode('utf-8')
            if uf.name.endswith('.json'):
                df_up = flatten_earthsavvy_json(content, uf.name)
                if not df_up.empty:
                    all_records.append(df_up)
            elif uf.name.endswith('.csv'):
                try:
                    df_csv = pd.read_csv(io.StringIO(content))
                    # Basic canonicalisation for CSV
                    df_csv['source_file'] = uf.name
                    df_csv['site_id'] = df_csv.get('site_id', 'Unknown')
                    # Wrap rows into 'raw_data' to match JSON structure logic
                    # This is a simplification; in a real app, we'd handle CSVs more robustly
                    df_csv['raw_data'] = df_csv.values.tolist()
                    all_records.append(df_csv)
                except Exception as e:
                    st.error(f"Error reading CSV {uf.name}: {e}")

    # Combine
    if all_records:
        master_df = pd.concat(all_records, ignore_index=True)
        st.session_state['master_df'] = master_df
        st.success(f"Loaded {len(master_df)} records from {len(set(master_df['source_file']))} files.")
        
        st.write("##### Raw Data Preview")
        # FIXED: Create a display copy and cast list columns to string to avoid PyArrow error
        display_df = master_df.head().copy()
        if 'raw_data' in display_df.columns:
            display_df['raw_data'] = display_df['raw_data'].astype(str)
        st.dataframe(display_df)
    else:
        st.info("Please upload files or select local files to proceed.")
        st.stop()

# =============================================================================
# TAB 2: MAP COLUMNS
# =============================================================================
with tab_map:
    st.subheader("Map Data Columns to Physics Model")
    st.markdown("""
    The tool needs to know which columns in your data represent **Temperature** and **Area**. 
    EarthSavvy JSONs often have data arrays like `["Timestamp", Value1, Value2]`.
    """)
    
    if 'master_df' in st.session_state:
        df = st.session_state['master_df']
        
        # Analyze the first row's data structure to help user choose
        if not df.empty and 'raw_data' in df.columns:
            sample_row = df.iloc[0]['raw_data']
            st.code(f"Sample Data Row Structure: {sample_row}")
        else:
            st.warning("Dataframe appears empty or missing 'raw_data' column.")
            sample_row = []
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Temperature Input**")
            temp_idx = st.number_input("Index of Temperature Column (0-based)", min_value=0, value=1, step=1, help="Usually index 1 in EarthSavvy exports.")
            
        with col2:
            st.markdown("**Area Input (Optional)**")
            use_area_col = st.checkbox("Read Area from data column?", value=False)
            area_idx = None
            if use_area_col:
                area_idx = st.number_input("Index of Area Column (0-based)", min_value=0, value=2, step=1)
            else:
                st.info(f"Using fixed default area: {p_default_area} m² (change in sidebar)")
                
        if st.button("Run Engineering Analysis"):
            # Apply Calculations
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            total_rows = len(df)
            
            for i, row in df.iterrows():
                res = estimate_heat_loss(row, temp_idx, area_idx, model_params)
                if res:
                    # Merge original metadata with calculation results
                    full_row = row.to_dict()
                    # Remove raw_data to keep it clean (and prevent Arrow errors later)
                    if 'raw_data' in full_row:
                        del full_row['raw_data']
                    full_row.update(res)
                    results.append(full_row)
                
                if i % 100 == 0:
                    progress_bar.progress(min(i / total_rows, 1.0))
            
            progress_bar.progress(1.0)
            
            if results:
                st.session_state['results_df'] = pd.DataFrame(results)
                st.success("Analysis Complete! Go to 'Results' tab.")
            else:
                st.error("Analysis failed. Check your column indices.")

# =============================================================================
# TAB 3: RESULTS
# =============================================================================
with tab_analysis:
    if 'results_df' in st.session_state:
        res_df = st.session_state['results_df']
        
        st.subheader("🏆 Portfolio Summary")
        
        # KPI Cards
        kpi1, kpi2, kpi3 = st.columns(3)
        total_cost = res_df['annual_cost_gbp'].sum()
        total_kwh = res_df['annual_kwh'].sum()
        total_co2 = res_df['annual_tco2e'].sum()
        
        kpi1.metric("Total Avoidable Cost", f"£{total_cost:,.0f}", help="Based on estimated heat loss and tariff")
        kpi2.metric("Total Energy Loss", f"{total_kwh:,.0f} kWh")
        kpi3.metric("Carbon Footprint", f"{total_co2:,.1f} tCO2e")
        
        st.markdown("---")
        
        # Ranking Table
        st.write("### 🏢 Site/Asset Ranking")
        st.markdown("Aggregated by Site Name. Click headers to sort.")
        
        ranking = res_df.groupby('site_name').agg({
            'annual_cost_gbp': 'sum',
            'annual_kwh': 'sum',
            'annual_tco2e': 'sum',
            'area_m2': 'sum'
        }).reset_index().sort_values('annual_cost_gbp', ascending=False)
        
        st.dataframe(ranking.style.format({
            'annual_cost_gbp': "£{:,.0f}",
            'annual_kwh': "{:,.0f}",
            'annual_tco2e': "{:,.2f}",
            'area_m2': "{:,.1f}"
        }), use_container_width=True)
        
        # Drill Down
        st.markdown("---")
        st.write("### 🔍 Detailed Analysis")
        
        col_sel, col_chart = st.columns([1, 2])
        
        with col_sel:
            selected_site = st.selectbox("Select Site to Inspect", res_df['site_name'].unique())
            site_data = res_df[res_df['site_name'] == selected_site]
            
            st.write(f"**Records:** {len(site_data)}")
            st.write(f"**Avg Temp:** {site_data['temp_c'].mean():.1f} °C")
            st.write(f"**Est. Cost:** £{site_data['annual_cost_gbp'].sum():,.0f}")
            
            # Classification breakdown
            st.write("**Hotspot Types:**")
            counts = site_data['classification'].value_counts()
            st.dataframe(counts)

        with col_chart:
            # Time Series Plot
            if site_data['timestamp'].notnull().any():
                fig = px.line(site_data.sort_values('timestamp'), x='timestamp', y='annual_cost_gbp', 
                              title=f"Estimated Cost Run Rate over Time - {selected_site}",
                              labels={'annual_cost_gbp': 'Annualised Cost Rate (£/yr)', 'timestamp': 'Date'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timestamp data available for time-series plotting.")
                # Distribution plot instead
                fig = px.histogram(site_data, x='temp_c', title=f"Temperature Distribution - {selected_site}")
                st.plotly_chart(fig, use_container_width=True)

        # Uncertainty Analysis (Monte Carlo light)
        with st.expander("🎲 Uncertainty & Sensitivity"):
            st.write(f"Assuming +/- 20% variance on Heat Transfer Coefficient (h={model_params['h_conv']})")
            low_bound = total_cost * 0.8
            high_bound = total_cost * 1.2
            st.write(f"**Cost Range:** £{low_bound:,.0f} - £{high_bound:,.0f}")
            
    else:
        st.info("Please map columns and run analysis in the 'Map Columns' tab first.")

# =============================================================================
# TAB 4: EXPORT
# =============================================================================
with tab_export:
    st.subheader("Generate Reports")
    
    if 'results_df' in st.session_state:
        res_df = st.session_state['results_df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. Download Data")
            csv = convert_df_to_csv(res_df)
            json_dat = convert_df_to_json(res_df)
            
            st.download_button(
                label="Download Results (CSV)",
                data=csv,
                file_name='earthsavvy_capex_estimates.csv',
                mime='text/csv',
            )
            
            st.download_button(
                label="Download Results (JSON)",
                data=json_dat,
                file_name='earthsavvy_capex_estimates.json',
                mime='application/json',
            )
            
        with col2:
            st.markdown("#### 2. HTML Summary Report")
            if st.button("Generate HTML Report"):
                # Simple string template
                total_cost = res_df['annual_cost_gbp'].sum()
                report_html = f"""
                <html>
                <head><title>EarthSavvy CAPEX Report</title></head>
                <body style="font-family: sans-serif; padding: 20px;">
                    <h1 style="color: #2E86C1;">Thermal-to-CAPEX Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                    <hr>
                    <h2>Executive Summary</h2>
                    <p>Total Identified Annual Saving Potential: <strong>£{total_cost:,.2f}</strong></p>
                    <p>Total Carbon Reduction Potential: <strong>{res_df['annual_tco2e'].sum():,.2f} tCO2e</strong></p>
                    <h3>Top Opportunities</h3>
                    <ul>
                """
                
                # Add top 5 sites
                ranking = res_df.groupby('site_name')['annual_cost_gbp'].sum().sort_values(ascending=False).head(5)
                for site, cost in ranking.items():
                    report_html += f"<li><strong>{site}</strong>: £{cost:,.2f}</li>"
                
                report_html += """
                    </ul>
                    <p><em>Disclaimer: These are screening-grade estimates based on surface temperature anomalies. Verify with direct contact measurement before capital investment.</em></p>
                </body>
                </html>
                """
                
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="capex_report.html">Click here to download HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
    else:
        st.info("No results to export yet.")

# Footer
st.markdown("---")
st.caption("Thermal-to-CAPEX Tool v1.0 | Offline Mode | No external API dependencies")
