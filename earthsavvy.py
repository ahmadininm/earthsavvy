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
# APP CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Thermal-to-CAPEX | EarthSavvy & AME",
    page_icon="🔥",
    layout="wide"
)

# -----------------------------------------------------------------------------
# CONSTANTS & PHYSICS
# -----------------------------------------------------------------------------
STEFAN_BOLTZMANN = 5.67e-8  # W/m2/K4

def load_image(image_name):
    """Helper to verify image exists before displaying to prevent errors."""
    if os.path.exists(image_name):
        return image_name
    return None

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
                    record = {
                        'source_file': filename,
                        'site_id': site_key,
                        'site_name': site_name,
                        'raw_data': row  # Store full list to extract columns later
                    }
                    
                    # Attempt to parse timestamp from first column
                    if row and len(row) > 0:
                        try:
                            ts_str = str(row[0])
                            # Handles ISO format standard in EarthSavvy
                            ts = pd.to_datetime(ts_str, errors='coerce')
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
        data_row = row['raw_data']
        
        # --- 1. GET TEMPERATURE ---
        if temp_col_idx is not None and len(data_row) > temp_col_idx:
            val = data_row[temp_col_idx]
            # Handle potential None or non-numeric
            t_surf_input = float(val) if val is not None else 0.0
        else:
            return None 
            
        # --- 2. GET AREA ---
        # Logic: If area column is mapped, check if it has valid data (>0).
        # If it is 0.0 or missing, FALLBACK to default_area.
        area = 0.0
        if area_col_idx is not None and len(data_row) > area_col_idx:
            try:
                area = float(data_row[area_col_idx])
            except:
                area = 0.0
        
        # Fallback check
        if area <= 0.001:
            area = params['default_area']

        # --- 3. PHYSICS CALC ---
        t_surf_c = t_surf_input
        t_amb_c = params['t_ambient']
        
        # Temperature in Kelvin
        tk_surf = t_surf_c + 273.15
        tk_amb = t_amb_c + 273.15
        
        # A. Convection: q_conv = h * A * (T_surf - T_amb)
        delta_t = t_surf_c - t_amb_c
        
        # If Surface is COLDER than ambient, assume 0 loss (ignore cooling loads for this specific CAPEX tool)
        if delta_t < 0:
            delta_t = 0
            
        q_conv_flux = params['h_conv'] * delta_t # W/m2
        
        # B. Radiation: q_rad = eps * sigma * (T_surf^4 - T_amb^4)
        if tk_surf > tk_amb:
            q_rad_flux = params['emissivity'] * STEFAN_BOLTZMANN * (tk_surf**4 - tk_amb**4)
        else:
            q_rad_flux = 0
            
        total_flux = q_conv_flux + q_rad_flux # W/m2
        total_power = total_flux * area # W (Joules/sec)
        
        # --- 4. FINANCIALS ---
        power_kw = total_power / 1000.0
        # Annual KWh = kW * 8760 * duty_cycle
        annual_kwh = power_kw * 8760 * params['duty_cycle']
        
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
    """Heuristic rule-based classification."""
    if delta_t > 50:
        return "Critical: Process/Oven Leak"
    elif delta_t > 20 and area > 20:
        return "Major: Fabric Insulation Fail"
    elif delta_t > 10:
        return "Moderate: Thermal Bridge"
    elif delta_t <= 0:
        return "No Anomalous Heat"
    else:
        return "Minor/Background"

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_json(df):
    return df.to_json(orient='records', date_format='iso')

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------

# --- HEADER WITH LOGOS ---
# Using 3 columns to place logos on far left and far right
header_col1, header_col2, header_col3 = st.columns([1, 4, 1])

with header_col1:
    ame_logo = load_image("ame.png")
    if ame_logo:
        st.image(ame_logo, width=120)
    else:
        st.write("*(AME Logo)*")

with header_col2:
    st.markdown("<h1 style='text-align: center;'>Thermal-to-CAPEX Calculator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><b>EarthSavvy / JLR Clean Futures Support Tool</b></p>", unsafe_allow_html=True)

with header_col3:
    es_logo = load_image("earthsavvy.png")
    if es_logo:
        st.image(es_logo, width=120)
    else:
        st.write("*(EarthSavvy Logo)*")

st.markdown("---")

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.header("🛠 Model Parameters")

with st.sidebar.expander("Physics Assumptions", expanded=True):
    p_t_amb = st.number_input("Ambient Temp (°C)", value=10.0, step=1.0, help="Baseline annual average air temperature.")
    p_emis = st.number_input("Emissivity (0-1)", value=0.9, step=0.05, help="0.9 for brick/concrete/painted metal.")
    p_h_conv = st.number_input("Conv. Coeff (W/m²K)", value=10.0, step=1.0, help="10=Indoor/Calm, 25=Outdoor/Windy.")

with st.sidebar.expander("Financials", expanded=True):
    p_tariff = st.number_input("Elec Tariff (£/kWh)", value=0.15, format="%.3f")
    p_carbon = st.number_input("Grid Carbon (kgCO2e/kWh)", value=0.193, format="%.3f")
    p_duty = st.slider("Duty Cycle", 0.1, 1.0, 1.0, help="1.0 = 24/7/365 operation.")

st.sidebar.markdown("---")
p_default_area = st.sidebar.number_input("Default Area (m²)", value=5.0, help="Used if data column is 0 or missing.")

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
tab1, tab2, tab3, tab4 = st.tabs(["📂 1. Upload", "⚙️ 2. Map & Calibrate", "📊 3. Business Case", "💾 4. Export"])

# =============================================================================
# TAB 1: UPLOAD
# =============================================================================
with tab1:
    st.info("Upload EarthSavvy JSON exports or standard CSVs to begin.")
    
    col_u1, col_u2 = st.columns(2)
    
    with col_u1:
        st.subheader("Upload")
        uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, type=['json', 'csv'])
    
    with col_u2:
        st.subheader("Local Folder Scan")
        # Automatically find files in current directory for convenience
        local_files = [f for f in os.listdir('.') if f.endswith(('.json', '.csv')) and 'requirements' not in f]
        selected_local = st.multiselect("Select local files:", local_files)

    all_records = []
    
    # Process Files
    files_to_process = []
    
    # 1. Handle Streamlit Uploads
    if uploaded_files:
        for uf in uploaded_files:
            files_to_process.append({'name': uf.name, 'content': uf.read().decode('utf-8')})
            
    # 2. Handle Local Files
    for lf in selected_local:
        try:
            with open(lf, 'r') as f:
                files_to_process.append({'name': lf, 'content': f.read()})
        except Exception as e:
            st.warning(f"Skipped {lf}: {e}")
            
    # Parsing Logic
    for file_obj in files_to_process:
        fname = file_obj['name']
        content = file_obj['content']
        
        if fname.endswith('.json'):
            df_temp = flatten_earthsavvy_json(content, fname)
            if not df_temp.empty:
                all_records.append(df_temp)
        elif fname.endswith('.csv'):
            try:
                df_csv = pd.read_csv(io.StringIO(content))
                df_csv['source_file'] = fname
                df_csv['site_id'] = df_csv.get('site_id', 'Unknown')
                df_csv['site_name'] = df_csv.get('site_name', 'Unknown')
                # Create raw_data list for consistency
                df_csv['raw_data'] = df_csv.values.tolist()
                all_records.append(df_csv)
            except:
                pass

    if all_records:
        master_df = pd.concat(all_records, ignore_index=True)
        st.session_state['master_df'] = master_df
        
        st.success(f"✅ Loaded {len(master_df)} data points from {len(files_to_process)} files.")
        
        with st.expander("Peek at Raw Data"):
            # Convert list column to string for display safety
            disp = master_df.head(5).copy()
            if 'raw_data' in disp.columns:
                disp['raw_data'] = disp['raw_data'].astype(str)
            st.dataframe(disp)
    else:
        st.warning("No data loaded yet.")

# =============================================================================
# TAB 2: MAP
# =============================================================================
with tab2:
    if 'master_df' in st.session_state:
        df = st.session_state['master_df']
        st.subheader("Map Data Columns")
        st.markdown("Identify which columns inside the data array correspond to **Temperature** and **Area**.")
        
        # Show sample
        sample_row = df.iloc[0]['raw_data'] if not df.empty else []
        st.code(f"Sample Row Structure: {sample_row}", language="json")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            temp_idx = st.number_input("Temperature Column Index", 0, 10, 1, help="Usually index 1.")
        with c2:
            use_area = st.checkbox("Map Area Column?", value=False)
            area_idx = st.number_input("Area Column Index", 0, 10, 2, disabled=not use_area)
        with c3:
            st.markdown(f"**Default Area:** {p_default_area} m²")
            st.caption("Applied if mapped column is 0 or unmapped.")

        if st.button("🚀 Run Analysis", type="primary"):
            results = []
            valid_count = 0
            
            # Progress bar
            bar = st.progress(0)
            
            for i, row in df.iterrows():
                res = estimate_heat_loss(row, temp_idx, area_idx if use_area else None, model_params)
                if res:
                    full_row = row.to_dict()
                    if 'raw_data' in full_row: del full_row['raw_data']
                    full_row.update(res)
                    results.append(full_row)
                    valid_count += 1
                
                if i % 50 == 0:
                    bar.progress(min(i / len(df), 1.0))
            
            bar.progress(1.0)
            
            if results:
                res_df = pd.DataFrame(results)
                st.session_state['results_df'] = res_df
                
                # SANITY CHECK WARNING
                max_temp = res_df['temp_c'].max()
                if max_temp < 5.0:
                    st.warning(f"⚠️ Warning: Max temperature detected is only {max_temp:.2f}. Are the input columns normalised (0-1) instead of Celsius?")
                else:
                    st.success(f"Analysis Complete! {valid_count} records processed.")
            else:
                st.error("Analysis failed. Check indices.")
    else:
        st.info("Go to Upload tab first.")

# =============================================================================
# TAB 3: BUSINESS CASE
# =============================================================================
with tab3:
    if 'results_df' in st.session_state:
        res_df = st.session_state['results_df']
        
        # TOTALS
        total_cost = res_df['annual_cost_gbp'].sum()
        total_co2 = res_df['annual_tco2e'].sum()
        avg_temp = res_df['temp_c'].mean()
        
        st.header("Executive Summary")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Est. Annual Cost", f"£{total_cost:,.0f}", delta="Loss Opportunity", delta_color="inverse")
        kpi2.metric("Carbon Impact", f"{total_co2:,.1f} tCO2e", delta="Avoidable Emissions", delta_color="inverse")
        kpi3.metric("Sites Analyzed", len(res_df['site_name'].unique()))
        kpi4.metric("Avg Surface Temp", f"{avg_temp:.1f} °C")
        
        st.markdown("---")
        
        # CHARTS
        c_chart1, c_chart2 = st.columns(2)
        
        with c_chart1:
            st.subheader("Cost by Site")
            fig_bar = px.bar(res_df.groupby('site_name')['annual_cost_gbp'].sum().reset_index(),
                             x='site_name', y='annual_cost_gbp',
                             labels={'annual_cost_gbp': 'Annual Cost (£)'},
                             color='annual_cost_gbp', color_continuous_scale='OrRd')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_chart2:
            st.subheader("Hotspot Severity")
            fig_pie = px.pie(res_df, names='classification', title='Classification of Thermal Anomalies',
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)

        # DETAILED TABLE
        st.subheader("Detailed Asset Register")
        detailed_view = res_df[['timestamp', 'site_name', 'temp_c', 'area_m2', 'annual_cost_gbp', 'classification']].copy()
        detailed_view['timestamp'] = detailed_view['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(detailed_view.sort_values('annual_cost_gbp', ascending=False), use_container_width=True)

    else:
        st.info("No results yet.")

# =============================================================================
# TAB 4: EXPORT & REPORT
# =============================================================================
with tab4:
    st.subheader("Output & Justification")
    
    if 'results_df' in st.session_state:
        res_df = st.session_state['results_df']
        total_cost = res_df['annual_cost_gbp'].sum()
        total_co2 = res_df['annual_tco2e'].sum()
        top_site = res_df.groupby('site_name')['annual_cost_gbp'].sum().idxmax()
        
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            st.markdown("### 📥 Download Data")
            st.download_button("Download CSV", convert_df_to_csv(res_df), "capex_model_output.csv", "text/csv")
            st.download_button("Download JSON", convert_df_to_json(res_df), "capex_model_output.json", "application/json")
            
        with col_ex2:
            st.markdown("### 📧 Generate Justification Email")
            
            # Auto-generated text
            email_text = f"""
Subject: CAPEX Justification - Thermal Loss Analysis Results

Hi Team,

Using the EarthSavvy/AME data, we have modeled the thermal efficiency of our monitored assets.

Summary Findings:
- Total Estimated Annual Loss: £{total_cost:,.0f}
- Carbon Impact: {total_co2:.1f} tCO2e per year
- Primary Hotspot Location: {top_site}

Based on these findings, an investment in insulation upgrades for {top_site} is recommended to reduce operational expenditures and meet our EV production sustainability targets.

Attached is the detailed dataset.

Regards,
            """
            st.text_area("Copy this text for your report:", email_text, height=250)
            
    else:
        st.info("Run analysis to enable exports.")
