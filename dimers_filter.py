import os
import numpy as np
import pandas as pd

# ==============================================================================
# USER CONFIGURATION PARAMETERS (CONFIGURED FOR HILIC3D)
# ==============================================================================
file_dir = r"C:\David_Data\Processed Data\Mzmine_Saliva\FIlter_vs_Extract\HILIC3D\AllLibraries"
file_name = "HILIC3D_AllLibraries_MSDIAL_Format_Cleaned.csv"

max_rt_diff = 0.08           # Max allowed RT discrepancy between rows (minutes)
ms1_ppm_tolerance = 15.0     # MS1 mass tolerance (ppm)
proton_mass = 1.007825       # Exact mass of a proton
# ==============================================================================

file_path = os.path.join(file_dir, file_name)

print("Step 1: Loading aligned HILIC3D dataset...")
df = pd.read_csv(file_path, sep=None, engine='python')

# Automatically isolate intensity columns for Filtered (F) and Liquid-Liquid (LL)
f_cols = [c for c in df.columns if c.startswith('F') and not c.startswith('Final') and not c.startswith('fragment')]
ll_cols = [c for c in df.columns if c.startswith('LL')]

# Convert empty or text cells to numerical 0 across raw data columns
for col in f_cols + ll_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

results = []

print("Step 2: Pairing monomers with potential [2M+H] dimers...")
for _, m_row in df.iterrows():
    m_id = m_row['id']
    m_rt = m_row['rt']
    m_mz = m_row['mz']
    
    # Mathematical target for protonated dimer: 2M - H+
    expected_dimer_mz = (2 * m_mz) - proton_mass
    mz_tolerance_da = (expected_dimer_mz * ms1_ppm_tolerance) / 1000000.0
    
    # Filter for co-eluting features matching the target dimer mass window
    matches = df[
        (df['mz'].between(expected_dimer_mz - mz_tolerance_da, expected_dimer_mz + mz_tolerance_da)) &
        (abs(df['rt'] - m_rt) <= max_rt_diff)
    ]
    
    for _, d_row in matches.iterrows():
        if m_id == d_row['id']:
            continue
            
        # Compute mean intensities across biological/technical replicates
        m_f_avg = m_row[f_cols].mean()
        m_ll_avg = m_row[ll_cols].mean()
        d_f_avg = d_row[f_cols].mean()
        d_ll_avg = d_row[ll_cols].mean()
        
        # Calculate direct ratio factors (Dimer Proportion Relative to Monomer)
        ratio_f = (d_f_avg / m_f_avg) if m_f_avg > 0 else 0.0
        ratio_ll = (d_ll_avg / m_ll_avg) if m_ll_avg > 0 else 0.0
        
        rt_d = round(d_row['rt'] - m_rt, 4)
        ppm_d = round((abs(d_row['mz'] - expected_dimer_mz) / expected_dimer_mz) * 1e6, 2)
        
        annot = m_row['Final_Annotations'] if pd.notna(m_row['Final_Annotations']) else 'Unknown'
        
        results.append({
            'Monomer_ID': m_id, 'Monomer_RT': m_rt, 'Monomer_m/z': m_mz,
            'Dimer_ID': d_row['id'], 'Dimer_RT': d_row['rt'], 'RT_Delta_min': rt_d,
            'Dimer_m/z': d_row['mz'], 'Mass_Deviation_ppm': ppm_d, 'Annotation': annot,
            'Monomer_Avg_F': round(m_f_avg, 1), 'Monomer_Avg_LL': round(m_ll_avg, 1),
            'Dimer_Avg_F': round(d_f_avg, 1), 'Dimer_Avg_LL': round(d_ll_avg, 1),
            'Dimer_Ratio_F': round(ratio_f, 4), 'Dimer_Ratio_LL': round(ratio_ll, 4)
        })

output_df = pd.DataFrame(results)

if not output_df.empty:
    output_df = output_df.sort_values(by='Monomer_ID')
    
    # Clean division handling zero boundaries natively without epsilon artifacts
    r_f = output_df['Dimer_Ratio_F'].values
    r_ll = output_df['Dimer_Ratio_LL'].values
    
    fc_conditions = [
        (r_f == 0) & (r_ll == 0),  # Both zero -> No change (0)
        (r_f == 0) & (r_ll > 0),   # Emerged from zero -> Infinite increase
        (r_f > 0) & (r_ll >= 0)    # Standard valid division
    ]
    fc_outputs = [
        1.0, 
        np.inf, 
        np.divide(r_ll, r_f, out=np.zeros_like(r_ll), where=r_f != 0)
    ]
    output_df['Fold_Change'] = np.select(fc_conditions, fc_outputs, default=1.0)
    
    # Classify behaviors precisely based on clean calculations
    behavior_conditions = [
        (output_df['Fold_Change'] <= 0.5),
        (output_df['Fold_Change'] >= 2.0),
        (output_df['Fold_Change'].between(0.5, 2.0))
    ]
    categories = ['Dimer Suppressed\n(>2-fold Drop)', 'Dimer Amplified\n(>2-fold Rise)', 'No Change']
    output_df['Dimer_Behavior'] = np.select(behavior_conditions, categories, default='No Change')
    
    # Save the polished output report
    csv_out_path = os.path.join(file_dir, "Dimer_Ratio_Comparison_Report.csv")
    output_df.to_csv(csv_out_path, index=False)
    
    print("\n" + "="*60)
    print("POLISHED REPORT GENERATION SUCCESSFUL")
    print("="*60)
    print(f"Total Dimer Pairs Found: {len(output_df)}")
    print(f"Saved Report Path:       {csv_out_path}")
    print("="*60)
else:
    print("Completed processing. Zero feature matches satisfied your parameters.")
