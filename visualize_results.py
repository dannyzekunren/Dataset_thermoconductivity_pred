"""
Thermal Conductivity Model Results Visualization
Comprehensive comparison of different model families using Plotly and Seaborn
"""

import sys
import io

# Fix UTF-8 encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure fonts for better appearance
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 13

# Define the model data
models_data = {
    'Model': [
        'ALiEGNN', 'Orb + CNN', 'HackNIP (Orb+MODNet)', 'ViKING',
        'CGCNN (structure baseline)', 'KAN + Custom Features', 'MLP + Custom Features',
        'MACE_OMAT + MLP', 'MACE_MPA + MLP', 'WyFormer',
        'XGBoost + Custom Features', 'CrabNet (composition baseline)', 'Orb + TabPFN',
        'eqV2-MEX', 'WyCryst+', 'Tomographic + MLP'
    ],
    'Model_Family': [
        'End-to-End Deep NN', 'MLIP Embeddings + ML', 'MLIP Embeddings + ML', 'Custom Features + ML',
        'End-to-End Deep NN', 'Custom Features + ML', 'Custom Features + ML',
        'MLIP Embeddings + ML', 'MLIP Embeddings + ML', 'End-to-End Deep NN',
        'Custom Features + ML', 'End-to-End Deep NN', 'MLIP Embeddings + ML',
        'MLIP Embeddings + ML', 'End-to-End Deep NN', 'Custom Features + ML'
    ],
    'Random_MAE': [
        0.379, 0.420, 0.380, 0.487, 0.523, 0.473, 0.431,
        0.448, 0.445, 0.503, 0.444, 0.523, 0.378, np.nan, 0.491, 0.687
    ],
    'SpaceGroup_MAE': [
        0.512, 0.509, 0.501, 0.597, 0.652, 0.596, 0.627,
        0.601, 0.602, 0.625, 0.540, 0.674, 0.494, np.nan, 0.700, 0.740
    ],
    'OOD_MAE': [
        1.245, 1.386, 1.516, 1.370, 1.294, 1.424, 1.454,
        1.466, 1.484, 1.419, 1.610, 1.422, 1.764, 0.997, 1.516, 1.580
    ],
    'Random_R2': [
        0.784, 0.753, 0.762, 0.746, 0.692, 0.748, 0.751,
        0.717, 0.715, np.nan, 0.758, 0.650, 0.758, np.nan, 0.651, 0.525
    ],
    'SpaceGroup_R2': [
        0.697, 0.694, 0.675, 0.672, 0.602, 0.664, 0.631,
        0.611, 0.597, np.nan, 0.705, 0.570, 0.707, np.nan, 0.521, 0.502
    ],
    'OOD_R2': [
        -3.595, -3.94, -8.598, -4.14, -5.400, -5.074, -5.968,
        -5.126, -5.076, np.nan, -6.695, -4.499, -7.190, -2.584, -5.50, -5.638
    ]
}

df = pd.DataFrame(models_data)

# Calculate average MAE (handling NaN)
df['Average_MAE'] = df[['Random_MAE', 'SpaceGroup_MAE', 'OOD_MAE']].mean(axis=1)

# Sort by average MAE
df_sorted = df.sort_values('Average_MAE')

# Color palette for model families
family_colors = {
    'End-to-End Deep NN': '#FF6B6B',
    'MLIP Embeddings + ML': '#4ECDC4',
    'Custom Features + ML': '#FFE66D',
    'Fine-tuned MLIPs': '#95E1D3'
}

print("=" * 80)
print("THERMAL CONDUCTIVITY MODEL RESULTS VISUALIZATION")
print("=" * 80)
print(f"\nTotal Models: {len(df)}")
print(f"Model Families: {df['Model_Family'].nunique()}")
print(f"\nBest Average MAE: {df['Average_MAE'].min():.3f} ({df_sorted.iloc[0]['Model']})")
print(f"Worst Average MAE: {df['Average_MAE'].max():.3f} ({df_sorted.iloc[-1]['Model']})")

# ============================================================================
# FIGURE 1: Individual Model MAE Comparison (Bar Plot by Split)
# ============================================================================
print("\n[1/5] Creating Individual Model MAE Bar Plot...")

fig1 = go.Figure()

# Add bars for each split
splits = ['Random_MAE', 'SpaceGroup_MAE', 'OOD_MAE']
split_names = ['Random Split', 'Space Group Split', 'OOD Split']
split_colors = ['#3498db', '#2ecc71', '#e74c3c']

df_sorted_indices = df_sorted.index
x_labels = [df.loc[i, 'Model'] for i in df_sorted_indices]

for split, split_name, color in zip(splits, split_names, split_colors):
    y_vals = [df.loc[i, split] for i in df_sorted_indices]
    fig1.add_trace(go.Bar(
        x=x_labels,
        y=y_vals,
        name=split_name,
        marker_color=color,
        hovertemplate='<b>%{x}</b><br>' + split_name + ': %{y:.3f}<extra></extra>',
        showlegend=True
    ))

fig1.update_layout(
    title={
        'text': '<b>Model Performance: MAE by Data Split</b><br><sub>Lower is better</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'Arial'}
    },
    xaxis_title='Model',
    yaxis_title='Mean Absolute Error (MAE)',
    barmode='group',
    height=600,
    hovermode='x unified',
    font=dict(family='Arial, sans-serif', size=11),
    xaxis=dict(
        tickangle=-45,
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        gridcolor='lightgray',
        showgrid=True,
        range=[0.3, 1.9]  # Set y-axis range from 0.3 to 1.9
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    legend=dict(
        orientation='v',
        yanchor='top',
        y=0.99,
        xanchor='right',
        x=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='gray',
        borderwidth=1
    ),
    margin=dict(b=150, l=70, r=50, t=100)
)

fig1.write_html('results_individual_mae_by_split.html')
print("[OK] Saved: results_individual_mae_by_split.html")

# ============================================================================
# FIGURE 2: Clustered Models (Model Family Comparison)
# ============================================================================
print("\n[2/6] Creating Clustered Model Family Comparison...")

# Create data for grouped bar chart
df_families = df.groupby('Model_Family')[['Random_MAE', 'SpaceGroup_MAE', 'OOD_MAE']].mean().reset_index()
df_families['Average_MAE'] = df_families[['Random_MAE', 'SpaceGroup_MAE', 'OOD_MAE']].mean(axis=1)
df_families = df_families.sort_values('Average_MAE')

fig2 = go.Figure()

for split, split_name, color in zip(splits, split_names, split_colors):
    y_vals = df_families[split].values
    fig2.add_trace(go.Bar(
        x=df_families['Model_Family'],
        y=y_vals,
        name=split_name,
        marker_color=color,
        hovertemplate='<b>%{x}</b><br>' + split_name + ': %{y:.3f}<extra></extra>',
    ))

fig2.update_layout(
    title={
        'text': '<b>Average Performance by Model Family</b><br><sub>Clustered comparison</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'Arial'}
    },
    xaxis_title='Model Family',
    yaxis_title='Mean Absolute Error (MAE)',
    barmode='group',
    height=600,
    hovermode='x unified',
    font=dict(family='Arial, sans-serif', size=12),
    xaxis=dict(
        tickangle=-30,
        tickfont=dict(size=11)
    ),
    yaxis=dict(
        gridcolor='lightgray',
        showgrid=True,
        range=[0.3, 1.0]  # Set y-axis range from 0.3 to 1.0 for family comparison
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    legend=dict(
        orientation='v',
        yanchor='top',
        y=0.99,
        xanchor='right',
        x=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='gray',
        borderwidth=1
    ),
    margin=dict(b=120, l=70, r=50, t=100)
)

fig2.write_html('results_clustered_by_family.html')
print("[OK] Saved: results_clustered_by_family.html")

# ============================================================================
# FIGURE 3: Average MAE Ranking (All Models)
# ============================================================================
print("\n[3/6] Creating Average MAE Ranking...")

fig3 = go.Figure()

colors_ranked = [family_colors.get(family, '#999999') for family in df_sorted['Model_Family']]

fig3.add_trace(go.Bar(
    x=df_sorted['Average_MAE'],
    y=df_sorted['Model'],
    orientation='h',
    marker_color=colors_ranked,
    text=[f"{val:.3f}" for val in df_sorted['Average_MAE']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Average MAE: %{x:.3f}<extra></extra>',
))

fig3.update_layout(
    title={
        'text': '<b>Overall Model Rankings by Average MAE</b><br><sub>Best to worst performance</sub>',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16, 'family': 'Arial'}
    },
    xaxis_title='Average MAE (Lower is Better)',
    yaxis_title='Model',
    height=700,
    font=dict(family='Arial, sans-serif', size=11),
    xaxis=dict(
        gridcolor='lightgray',
        showgrid=True,
        range=[0.3, 1.1]  # Set x-axis range from 0.3 to 1.1 for horizontal bar chart
    ),
    plot_bgcolor='rgba(240, 240, 240, 0.5)',
    margin=dict(l=250, r=100, t=100, b=50),
    showlegend=False
)

fig3.write_html('results_average_mae_ranking.html')
print("[OK] Saved: results_average_mae_ranking.html")

# ============================================================================
# FIGURES 5-7: Individual Split Bar Plots (Colored by Model Family)
# ============================================================================
print("\n[5/7] Creating Individual Split Bar Plots (Colored by Family)...")

split_configs = [
    {
        'split_col': 'Random_MAE',
        'split_name': 'Random Split',
        'filename': 'results_random_split_by_family.html',
        'description': 'Random Split Performance by Model Family'
    },
    {
        'split_col': 'SpaceGroup_MAE',
        'split_name': 'Space Group Split',
        'filename': 'results_spacegroup_split_by_family.html',
        'description': 'Space Group Split Performance by Model Family'
    },
    {
        'split_col': 'OOD_MAE',
        'split_name': 'OOD Split',
        'filename': 'results_ood_split_by_family.html',
        'description': 'OOD Split Performance by Model Family'
    }
]

for split_idx, split_config in enumerate(split_configs, start=4):
    print(f"\n[{split_idx}/6] Creating {split_config['split_name']} Bar Plot...")
    
    # Filter out NaN values for this split
    df_split = df[['Model', 'Model_Family', split_config['split_col']]].copy()
    df_split = df_split.dropna(subset=[split_config['split_col']])
    df_split = df_split.sort_values(split_config['split_col'])
    
    # Determine y-axis max based on split type
    if 'OOD' in split_config['split_name']:
        y_max = 1.9  # OOD split has higher MAE values
    else:
        y_max = 0.8  # Random and Space Group splits
    
    # Create color array for each bar based on model family
    bar_colors = [family_colors.get(family, '#999999') for family in df_split['Model_Family']]
    
    # Create figure with single trace, color each bar by family
    fig_split = go.Figure()
    
    fig_split.add_trace(go.Bar(
        x=df_split['Model'],
        y=df_split[split_config['split_col']],
        marker_color=bar_colors,
        marker_line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
        hovertemplate='<b>%{x}</b><br>' + 
                     f"{split_config['split_name']} MAE: %{{y:.3f}}<br>" +
                     f"Family: %{{customdata}}<extra></extra>",
        customdata=df_split['Model_Family'],
        showlegend=False
    ))
    
    # Add legend manually with family information
    for family in df_split['Model_Family'].unique():
        family_color = family_colors.get(family, '#999999')
        fig_split.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=15, color=family_color),
            name=family,
            showlegend=True
        ))
    
    fig_split.update_layout(
        title={
            'text': f"<b>{split_config['description']}</b><br><sub>Bars colored by model family cluster</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Arial'}
        },
        xaxis_title='Model',
        yaxis_title='Mean Absolute Error (MAE)',
        height=600,
        hovermode='closest',
        font=dict(family='Arial, sans-serif', size=11),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            showgrid=True,
            title='MAE (Lower is Better)',
            range=[0.3, y_max]  # Set y-axis range from 0.3 to split-specific max
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1,
            font=dict(size=10)
        ),
        margin=dict(b=150, l=70, r=50, t=100)
    )
    
    fig_split.write_html(split_config['filename'])
    print(f"[OK] Saved: {split_config['filename']}")

# ============================================================================
# FIGURE 7: Seaborn - Advanced Visualizations (Static Plots)
# ============================================================================
print("\n[7/7] Creating Seaborn Advanced Visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Color mapping for model families
colors_sns = [family_colors.get(family, '#999999') for family in df_sorted['Model_Family']]

# Subplot 1: Horizontal bar chart - Average MAE
ax1 = axes[0, 0]
bars1 = ax1.barh(df_sorted['Model'], df_sorted['Average_MAE'], color=colors_sns, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Average MAE', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Ranking\n(Average MAE across all splits)', fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
for i, (model, mae) in enumerate(zip(df_sorted['Model'], df_sorted['Average_MAE'])):
    ax1.text(mae + 0.02, i, f'{mae:.3f}', va='center', fontsize=9)
ax1.set_xlim(0.3, 1.1)  # Set x-axis range from 0.3

# Subplot 2: Box plot - MAE distribution by split
ax2 = axes[0, 1]
mae_cols = ['Random_MAE', 'SpaceGroup_MAE', 'OOD_MAE']
data_for_box = []
labels_for_box = []
for col in mae_cols:
    # Remove NaN values
    valid_data = df[col].dropna()
    data_for_box.append(valid_data.values)
    labels_for_box.append(col.replace('_MAE', '').replace('SpaceGroup', 'Space Group'))

bp = ax2.boxplot(data_for_box, patch_artist=True)
for patch, color in zip(bp['boxes'], split_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels(labels_for_box)
ax2.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax2.set_title('MAE Distribution by Data Split', fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0.3, 1.9)  # Set y-axis range from 0.3 to 1.9

# Subplot 3: Scatter plot - Random vs OOD MAE
ax3 = axes[1, 0]
for family in df['Model_Family'].unique():
    family_data = df[df['Model_Family'] == family]
    color = family_colors.get(family, '#999999')
    ax3.scatter(family_data['Random_MAE'], family_data['OOD_MAE'], 
               s=150, alpha=0.7, label=family, color=color, edgecolors='black', linewidth=0.5)

# Add diagonal line
max_val = max(df['Random_MAE'].max(), df['OOD_MAE'].max())
ax3.plot([0.3, max_val], [0.3, max_val], 'k--', alpha=0.3, linewidth=1)
ax3.set_xlabel('Random Split MAE', fontsize=12, fontweight='bold')
ax3.set_ylabel('OOD Split MAE', fontsize=12, fontweight='bold')
ax3.set_title('Generalization Gap: Random vs OOD Performance', fontsize=13, fontweight='bold', pad=15)
ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax3.grid(alpha=0.3, linestyle='--')
ax3.set_xlim(0.3, 0.75)  # Set x-axis range from 0.3
ax3.set_ylim(0.3, 1.9)   # Set y-axis range from 0.3 to 1.9

# Subplot 4: Grouped bar chart - Family comparison
ax4 = axes[1, 1]
df_families_sorted = df_families.sort_values('Average_MAE')
x_pos = np.arange(len(df_families_sorted))
width = 0.25

ax4.bar(x_pos - width, df_families_sorted['Random_MAE'], width, 
       label='Random', color=split_colors[0], edgecolor='black', linewidth=0.5)
ax4.bar(x_pos, df_families_sorted['SpaceGroup_MAE'], width, 
       label='Space Group', color=split_colors[1], edgecolor='black', linewidth=0.5)
ax4.bar(x_pos + width, df_families_sorted['OOD_MAE'], width, 
       label='OOD', color=split_colors[2], edgecolor='black', linewidth=0.5)

ax4.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax4.set_title('Model Family Comparison by Split', fontsize=13, fontweight='bold', pad=15)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(df_families_sorted['Model_Family'], rotation=20, ha='right', fontsize=10)
ax4.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim(0.3, 1.0)  # Set y-axis range from 0.3

plt.tight_layout()
plt.savefig('results_seaborn_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Saved: results_seaborn_analysis.png")
plt.close()

# ============================================================================
# BONUS: Create a detailed summary table as image
# ============================================================================
print("\n[BONUS] Creating Summary Table Visualization...")

fig_table, ax_table = plt.subplots(figsize=(18, 10))
ax_table.axis('tight')
ax_table.axis('off')

# Prepare data
table_data = []
for idx, row in df_sorted.iterrows():
    table_data.append([
        row['Model'],
        row['Model_Family'],
        f"{row['Random_MAE']:.3f}" if not np.isnan(row['Random_MAE']) else "-",
        f"{row['SpaceGroup_MAE']:.3f}" if not np.isnan(row['SpaceGroup_MAE']) else "-",
        f"{row['OOD_MAE']:.3f}" if not np.isnan(row['OOD_MAE']) else "-",
        f"{row['Average_MAE']:.3f}",
    ])

columns = ['Model', 'Family', 'Random', 'Space Group', 'OOD', 'Average']
table = ax_table.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.20, 0.10, 0.12, 0.10, 0.10])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows based on family
for i, (idx, row) in enumerate(df_sorted.iterrows(), 1):
    color = family_colors.get(row['Model_Family'], '#ecf0f1')
    for j in range(len(columns)):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_alpha(0.7)

plt.title('Thermal Conductivity Model Results - Summary Table\n' +
         'Ranked by Average MAE (lower is better)',
         fontsize=14, fontweight='bold', pad=20)

plt.savefig('results_summary_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Saved: results_summary_table.png")
plt.close()

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\n[MODELS BY FAMILY]")
for family in df['Model_Family'].unique():
    family_df = df[df['Model_Family'] == family]
    avg_mae = family_df['Average_MAE'].mean()
    print(f"\n  {family}:")
    print(f"    - Models: {len(family_df)}")
    print(f"    - Avg MAE: {avg_mae:.3f}")
    print(f"    - Best: {family_df['Average_MAE'].min():.3f} ({family_df.loc[family_df['Average_MAE'].idxmin(), 'Model']})")
    print(f"    - Worst: {family_df['Average_MAE'].max():.3f} ({family_df.loc[family_df['Average_MAE'].idxmax(), 'Model']})")

print("\n[PERFORMANCE BY DATA SPLIT]")
print(f"  Random Split - Best: {df['Random_MAE'].min():.3f}, Avg: {df['Random_MAE'].mean():.3f}")
print(f"  Space Group  - Best: {df['SpaceGroup_MAE'].min():.3f}, Avg: {df['SpaceGroup_MAE'].mean():.3f}")
print(f"  OOD Split    - Best: {df['OOD_MAE'].min():.3f}, Avg: {df['OOD_MAE'].mean():.3f}")

print("\n[TOP 5 MODELS]")
for i, (idx, row) in enumerate(df_sorted.head(5).iterrows(), 1):
    print(f"  {i}. {row['Model']:<30} ({row['Model_Family']:<25}): {row['Average_MAE']:.3f} MAE")

print("\n" + "=" * 80)
print("[SUCCESS] All visualizations created successfully!")
print("=" * 80)
print("\nOutput Files Generated:")
print("  1. results_individual_mae_by_split.html     - Individual model comparison")
print("  2. results_clustered_by_family.html          - Model family clustering")
print("  3. results_average_mae_ranking.html          - Overall rankings")
print("  4. results_random_split_by_family.html       - Random split (colored by family)")
print("  5. results_spacegroup_split_by_family.html   - Space Group split (colored by family)")
print("  6. results_ood_split_by_family.html          - OOD split (colored by family)")
print("  7. results_seaborn_analysis.png              - Advanced statistical plots")
print("  8. results_summary_table.png                 - Results summary table")
print("\n" + "=" * 80)
