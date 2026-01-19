import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

# --- APP CONFIG ---
st.set_page_config(page_title="Thermal-to-CAPEX (EarthSavvy Add-on)", layout="wide")

"""
# Thermal-to-CAPEX (EarthSavvy Add-on)
### How it works:
1. **Upload & Detect**: Load EarthSavvy JSON exports or CSV data. The tool canonicalises various formats into a unified time-series/metric structure.
2. **Heat Loss Model**: Uses a physics-informed approach (Convection + Radiation) to estimate heat flux ($W/m^2$) from surface temperature anomalies.
3. **Financials & Carbon**: Extrapolates instantaneous loss to annual kWh, £, and $tCO_2e$ using degree-day scaling or seasonal multipliers.
4. **Classification**: Categorises hotspots based on thermal variance and magnitude to identify "Fabric Loss" vs "Process Hotspots".
"""

# --- UTILS & PARSING ---

def parse_earth_savvy_json(data, filename):
    """
    Attempts to parse EarthSavvy JSON formats.
    Handles: nested dicts with 'data' arrays, list of records, or flat dicts.
    """
    records = []
    try:
        # Format 1: { "site_name": { "name": "...", "data": [[ts, val1, val2], ...] } }
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict) and "data" in val:
                    site_name = val.get("name", key)
                    for entry in val["data"]:
                        # Entry usually [timestamp, val1, val2...]
                        ts = entry[0]
                        for i, metric_val in enumerate(entry[1:]):
                            records.append({
                                "timestamp": ts,
                                "location_id": site_name,
                                "metric_name": f"Metric_{i}",
                                "metric_value": metric_val,
                                "source_file": filename
                            })
                # Format 2: Flat dictionary of metrics
                elif not isinstance(val, (dict, list)):
                    records.append({
                        "timestamp": None,
                        "location_id": "Unknown",
                        "metric_name": key,
                        "metric_value": val,
                        "source_file": filename
                    })
        # Format 3: List of records
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    records.append({**item, "source_file": filename})
    except Exception as e:
        st.error(f"Error parsing {filename}: {e}")
    
    df = pd.DataFrame(records)
    if not df.empty and "timestamp" in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def estimate_heat_loss(anomaly_k, area, params):
    """
    Simplified Heat Loss Estimation (Screening Grade)
    Q = Q_conv + Q_rad
    Q_conv = h * A * dT
    Q_rad = epsilon * sigma * A * (T_s^4 - T_a^4) 
    Since we often only have 'anomaly' (dT), we linearise radiation for screening.
    """
    h = params['h_conv']
    epsilon = params['emissivity']
    sigma = 5.67e-8
    T_ambient = params['ambient_temp_c'] + 273.15
    T_surface = T_ambient + anomaly_k
    
    # Convective component
    q_conv = h * area * anomaly_k
    
    # Radiative component
    q_rad = epsilon * sigma * area * (T_surface**4 - T_ambient**4)
    
    total_w = q_conv + q_rad
    return total_w

# --- UI TABS ---

tab_upload, tab_settings, tab_results, tab_export = st.tabs([
    "📂 Upload & Detect", "⚙️ Analysis Settings", "📊 Results", "💾 Export"
])

if 'canonical_df' not in st.session_state:
    st.session_state.canonical_df = pd.DataFrame()

with tab_upload:
    st.header("Data Intake")
    uploaded_files = st.file_uploader("Upload EarthSavvy JSON or CSV files", accept_multiple_files=True)
    
    all_dfs = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.json'):
                content = json.load(uploaded_file)
                all_dfs.append(parse_earth_savvy_json(content, uploaded_file.name))
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                df['source_file'] = uploaded_file.name
                all_dfs.append(df)
    
    if all_dfs:
        st.session_state.canonical_df = pd.concat(all_dfs, ignore_index=True)
        st.success(f"Loaded {len(st.session_state.canonical_df)} records.")
        st.dataframe(st.session_state.canonical_df.head(), use_container_width=True)
    else:
        st.info("Awaiting file upload...")

with tab_settings:
    st.header("Modelling Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Physics Defaults")
        h_conv = st.slider("Convective Heat Transfer Coeff ($W/m^2K$)", 2.0, 25.0, 5.0, help="5.0 is typical for still air/low wind.")
        emissivity = st.slider("Surface Emissivity", 0.5, 1.0, 0.9)
        ambient_temp = st.number_input("Average Ambient Temp (°C)", value=10.0)
        default_area = st.number_input("Default Building Area ($m^2$)", value=500.0)
    
    with col2:
        st.subheader("Financials & Scaling")
        tariff = st.number_input("Energy Tariff (£/kWh)", value=0.15)
        carbon_factor = st.number_input("Carbon Factor ($kgCO_2e/kWh$)", value=0.21)
        scaling_factor = st.number_input("Annual Scaling Factor (Hours/Year)", value=4000, help="Adjust for seasonality/operation hours.")
        uncertainty_range = st.slider("Uncertainty Bound (%)", 5, 50, 20)

    params = {
        'h_conv': h_conv,
        'emissivity': emissivity,
        'ambient_temp_c': ambient_temp,
        'tariff': tariff,
        'carbon_factor': carbon_factor,
        'scaling': scaling_factor,
        'uncertainty': uncertainty_range / 100.0
    }

with tab_results:
    if st.session_state.canonical_df.empty:
        st.warning("Please upload data first.")
    else:
        df = st.session_state.canonical_df.copy()
        
        # Calculate Loss
        # We assume 'metric_value' is the temperature anomaly in Kelvin/Celsius for this screening
        df['heat_loss_w'] = df['metric_value'].apply(lambda x: estimate_heat_loss(x, default_area, params))
        
        # Aggregate by location
        summary = df.groupby('location_id').agg({
            'heat_loss_w': ['mean', 'max', 'std'],
            'metric_value': 'mean'
        }).reset_index()
        summary.columns = ['Location', 'Avg_Loss_W', 'Peak_Loss_W', 'Loss_Std', 'Avg_Anomaly_K']
        
        # Derived metrics
        summary['Annual_kWh'] = (summary['Avg_Loss_W'] / 1000) * params['scaling']
        summary['Annual_Cost_£'] = summary['Annual_kWh'] * params['tariff']
        summary['Annual_tCO2e'] = (summary['Annual_kWh'] * params['carbon_factor']) / 1000
        
        # Uncertainty Bounds
        summary['Cost_Low'] = summary['Annual_Cost_£'] * (1 - params['uncertainty'])
        summary['Cost_High'] = summary['Annual_Cost_£'] * (1 + params['uncertainty'])
        
        # Hotspot Classification
        def classify(row):
            if row['Avg_Anomaly_K'] > 15: return "Likely Process Hotspot"
            if row['Loss_Std'] / (row['Avg_Loss_W'] + 1e-6) > 0.5: return "Mixed/Uncertain"
            return "Likely Fabric Loss"
        
        summary['Classification'] = summary.apply(classify, axis=1)
        
        st.subheader("Ranked Avoidable Costs")
        st.dataframe(summary.sort_values(by='Annual_Cost_£', ascending=False), use_container_width=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_bar = px.bar(summary, x='Location', y='Annual_Cost_£', color='Classification', title="Annual Avoidable Cost by Location")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_c2:
            if 'timestamp' in df.columns and not df['timestamp'].isnull().all():
                selected_loc = st.selectbox("Drill-down: Select Location", summary['Location'].unique())
                ts_data = df[df['location_id'] == selected_loc].sort_values('timestamp')
                fig_ts = px.line(ts_data, x='timestamp', y='heat_loss_w', title=f"Thermal Loss Profile: {selected_loc}")
                st.plotly_chart(fig_ts, use_container_width=True)

with tab_export:
    st.header("Export Assets")
    if not st.session_state.canonical_df.empty:
        # CSV Export
        csv = summary.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Results", data=csv, file_name="thermal_capex_results.csv", mime="text/csv")
        
        # JSON Export
        json_res = summary.to_json(orient='records')
        st.download_button("Download JSON Results", data=json_res, file_name="thermal_capex_results.json", mime="application/json")
        
        # Simple HTML Report
        report_html = f"""
        <html><body>
        <h1>Thermal-to-CAPEX Screening Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <h2>Summary Table</h2>
        {summary.to_html()}
        <p><i>Assumptions: Tariff £{params['tariff']}, Scaling {params['scaling']} hrs/yr.</i></p>
        </body></html>
        """
        st.download_button("Download HTML Report", data=report_html, file_name="site_report.html", mime="text/html")
    else:
        st.info("No data available to export.")

# --- FOOTER ---
st.markdown("---")
st.caption("Thermal-to-CAPEX EarthSavvy Add-on | Expert Mode | Offline Ready")
