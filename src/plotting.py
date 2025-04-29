import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns # Import seaborn

# Assuming evaluation functions might be needed here too
try:
    from .evaluation import compute_adaptive_ece
except ImportError:
    try:
        from evaluation import compute_adaptive_ece
    except ImportError:
        print("Warning: Could not import compute_adaptive_ece from evaluation.")
        # Define a placeholder if ECE needed directly in plots and import fails
        def compute_adaptive_ece(df, n_bins=10): return 0.1 # Placeholder

def combined_visualization(dfs, names, output_path='../confidence_vs_accuracy_bubble.pdf'):
    """
    Creates a scatter plot visualizing model confidence vs accuracy.
    Bubble size represents ECE (inversely scaled), marker shape represents model family, 
    and color represents reasoning type.

    Args:
        dfs (dict): Dictionary of DataFrames, keyed by variant name.
        names (list): List of variant names (keys of dfs dict).
        output_path (str): Path to save the generated plot PDF.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    # Define marker styles for different model families (customize as needed)
    family_markers = {
        'GPT-4o': 'o', 'GPT-4o-mini': 's', 'df_xai_grok2': '^', 
        'df_Meta_Llama': 'D', 'df_claude': 'v', 'O3-mini-short': '<', 
        'O3-mini': '>', 'o1-short': 'p', 'o1': 'h', 'df_Qwen_QWQ': '*' 
    }
    # Define colors for reasoning types
    reasoning_colors = {
        'RL Induced Reasoning': 'blue', 'Prompted Reasoning': 'green', 'No Reasoning': 'red'
    }
    
    # --- Data Preparation --- 
    model_groups = {}
    for df_name in names:
        if df_name not in dfs: continue
        df = dfs[df_name]
        
        # Determine Family and Base Name (logic adapted from notebook)
        family, base_name = 'Unknown', 'Unknown' # Defaults
        # ... (Insert the detailed if/elif block from notebook to determine family/base_name)
        if '4o-mini' in df_name or '4omini' in df_name: base_name, family = 'GPT-4o-mini', 'GPT-4o-mini'
        elif '4o_' in df_name: base_name, family = 'GPT-4o', 'GPT-4o'
        elif 'o3mini_low' in df_name: base_name, family = 'O3-mini-short', 'O3-mini-short'
        elif 'o3mini_reason' in df_name: base_name, family = 'O3-mini', 'O3-mini'
        # ... (Add all other elif conditions for families like Claude, Llama, Grok, Qwen, o1 etc.)
        elif 'claude_haiku' in df_name: base_name, family = 'Claude-3.5-Haiku', 'df_claude'
        elif 'claude_sonnet' in df_name: base_name, family = 'Claude-3.7-Sonnet', 'df_claude'
        elif 'Llama_3' in df_name or 'Meta_Llama' in df_name: family = 'df_Meta_Llama'; base_name = 'Llama-3-70B' if '70B' in df_name else 'Llama-3-8B' if '8B' in df_name else 'Llama-3'
        elif 'grok2' in df_name or 'xai_grok2' in df_name: base_name, family = 'Grok-2', 'df_xai_grok2'
        elif 'QWQ' in df_name: base_name, family = 'QWQ', 'df_Qwen_QWQ'
        else: # Basic fallback
             try: base_name = df_name.split('_')[1]; family = f"df_{base_name}" # Guess family 
             except: pass
             if family not in family_markers: family_markers[family] = 'X' # Assign default marker
        
        # Determine Reasoning Type (logic adapted from notebook)
        if df_name.lower().endswith('noexp') or df_name.lower().endswith('no_exp'): reasoning_type = 'No Reasoning'
        elif any(df_name.lower().endswith(s) for s in ['_cot', '_ltm', '_sc']): reasoning_type = 'Prompted Reasoning'
        else: reasoning_type = 'RL Induced Reasoning' # Default / Implicit
        
        if base_name not in model_groups: model_groups[base_name] = {'internal': [], 'prompted': [], 'no_reasoning': [], 'family': family}
        
        # Group DFs by base model and reasoning type
        if reasoning_type == 'RL Induced Reasoning': model_groups[base_name]['internal'].append(df)
        elif reasoning_type == 'Prompted Reasoning': model_groups[base_name]['prompted'].append(df)
        else: model_groups[base_name]['no_reasoning'].append(df)
            
    # Calculate aggregate metrics for plotting
    scatter_metrics = []
    for base_name, group in model_groups.items():
        family = group['family']
        for r_type, dfs_list in [('RL Induced Reasoning', group['internal']), ('Prompted Reasoning', group['prompted']), ('No Reasoning', group['no_reasoning'])]:
            if dfs_list:
                # Aggregate metrics across potentially multiple DFs for the same type (e.g., different prompts)
                # Ensure confidence and correctness are numeric, drop NaNs for mean calculation
                confidences = pd.concat([pd.to_numeric(df['llm_confidence'], errors='coerce') for df in dfs_list]).dropna()
                correctness = pd.concat([pd.to_numeric(df['Correctness'], errors='coerce') for df in dfs_list]).dropna()
                # Calculate ECE across the combined data for this group
                combined_df_for_ece = pd.concat(dfs_list)
                ece = compute_adaptive_ece(combined_df_for_ece)

                if not confidences.empty and not correctness.empty:
                     scatter_metrics.append({
                         'model': base_name,
                         'family': family,
                         'reasoning': r_type,
                         'avg_confidence': confidences.mean(),
                         'accuracy': correctness.mean(),
                         'ece': ece if pd.notna(ece) else 0.5 # Default ECE if calculation failed
                     })
    
    if not scatter_metrics: 
        print("No data points generated for scatter plot.")
        return plt.figure() # Return empty figure
        
    scatter_df = pd.DataFrame(scatter_metrics)

    # --- Plotting --- 
    min_ece, max_ece = scatter_df['ece'].min(), scatter_df['ece'].max()
    def scale_size(ece): # Inverse scaling for bubble size
        if max_ece == min_ece: return 150
        normalized = 1 - ((ece - min_ece) / (max_ece - min_ece)) if max_ece > min_ece else 0.5
        return 100 + normalized * 200

    fig, ax_main = plt.subplots(figsize=(14, 8)) # Increased figure size
    
    # Plot points with unique marker/color combinations
    plotted_families = set()
    plotted_reasoning = set()
    for idx, row in scatter_df.iterrows():
         family = row['family']
         reasoning = row['reasoning']
         marker = family_markers.get(family, 'X') # Use default marker if family unknown
         color = reasoning_colors.get(reasoning, 'grey')
         size = scale_size(row['ece'])
         
         ax_main.scatter(row['avg_confidence'], row['accuracy'], 
                       marker=marker, c=color, s=size, alpha=0.75, edgecolors='w')
         plotted_families.add(family)
         plotted_reasoning.add(reasoning)
         
         # Annotate points
         ax_main.annotate(row['model'], (row['avg_confidence'], row['accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

    # Add diagonal line for perfect calibration
    limits = [min(ax_main.get_xlim()[0], ax_main.get_ylim()[0]), max(ax_main.get_xlim()[1], ax_main.get_ylim()[1])]
    ax_main.plot(limits, limits, 'k--', alpha=0.5, label='Perfect Calibration')

    ax_main.set_xlabel('Average Confidence', fontsize=12)
    ax_main.set_ylabel('Accuracy', fontsize=12)
    ax_main.set_title('Model Performance: Confidence vs Accuracy (Bubble Size ~ Inverse ECE)', fontsize=14, pad=20)
    ax_main.grid(True, alpha=0.3)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.legend() # Add legend for diagonal line

    # --- Create Legends --- 
    # Models Legend
    model_handles = []
    for family, marker in family_markers.items():
        if family in plotted_families:
             display_name = family.replace('df_', '').replace('_', ' ').title()
             model_handles.append(plt.Line2D([0], [0], marker=marker, color='grey', label=display_name, markersize=8, linestyle='None'))
    
    # Methods Legend
    method_handles = []
    for reasoning, color in reasoning_colors.items():
        if reasoning in plotted_reasoning:
            method_handles.append(plt.Line2D([0], [0], marker='o', color=color, label=reasoning, markersize=8, linestyle='None'))
            
    # Add legends outside the plot area to avoid overlap
    leg1 = fig.legend(handles=model_handles, title='Model Family', loc='center left', bbox_to_anchor=(1.01, 0.7), fontsize=10, title_fontsize=11)
    leg2 = fig.legend(handles=method_handles, title='Reasoning Method', loc='center left', bbox_to_anchor=(1.01, 0.4), fontsize=10, title_fontsize=11)
    fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legends
    
    # Save the figure
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    
    return fig

def analyze_confidence_token_relationship(dfs, names, output_path='../token_calibration_analysis.pdf'):
    """
    Analyzes and plots the relationship between token usage and calibration errors 
    (Underconfidence for correct, Overconfidence for incorrect) for reasoning models.

    Args:
        dfs (dict): Dictionary of DataFrames keyed by variant name.
        names (list): List of variant names (keys of dfs dict).
        output_path (str): Path to save the generated plot PDF.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Filter for models with specific reasoning suffixes
    reasoning_suffixes = ['_cot', '_low', '_reason', '_sc', '_ltm']
    reasoning_dfs = {k: v for k, v in dfs.items() if any(k.lower().endswith(s) for s in reasoning_suffixes)}
    
    # Define colors for model families (adjust as needed)
    family_colors = {
        'openai_4o': '#1f77b4', 'openai_4omini': '#ff7f0e', 'xai_grok2': '#2ca02c',
        'Meta_Llama': '#d62728', 'claude': '#9467bd', 'openai_o3mini': '#8c564b',
        'openai_o1': '#e377c2', 'Qwen_QWQ': '#7f7f7f'
    }
    default_color = '#000000'

    all_results = []
    for df_name, df_orig in reasoning_dfs.items():
        df = df_orig.copy()
        df['llm_confidence'] = pd.to_numeric(df['llm_confidence'], errors='coerce')
        df['Correctness'] = pd.to_numeric(df['Correctness'], errors='coerce')
        df['completion_tokens'] = pd.to_numeric(df['completion_tokens'], errors='coerce')
        df = df.dropna(subset=['llm_confidence', 'Correctness', 'completion_tokens'])
        df = df[df['Correctness'].isin([0, 1])]
        if df.empty: continue
        
        # Determine family and display name
        model_family_key = 'other'
        for key_part in family_colors.keys():
            if key_part.lower() in df_name.lower():
                model_family_key = key_part
                break
        color = family_colors.get(model_family_key, default_color)
        
        display_name = df_name.replace('df_', '') # Basic name cleaning
        # Add suffix indicators like (CoT), (SC), (LtM), (Reason) etc.
        for suffix, label in [('_cot', ' (CoT)'), ('_sc', ' (SC)'), ('_ltm', ' (LtM)'), ('_reason', ' (Reason)'), ('_low', ' (Low)')]:
             if display_name.lower().endswith(suffix):
                  display_name = display_name[:-len(suffix)] + label
                  break # Assume one suffix matches
                  
        df_correct = df[df['Correctness'] == 1]
        df_incorrect = df[df['Correctness'] == 0]
        
        # Calculate metrics, using np.nan if empty
        tokens_correct = df_correct['completion_tokens'].mean() if not df_correct.empty else np.nan
        ece_correct = compute_adaptive_ece(df_correct) if not df_correct.empty else np.nan
        tokens_incorrect = df_incorrect['completion_tokens'].mean() if not df_incorrect.empty else np.nan
        ece_incorrect = compute_adaptive_ece(df_incorrect) if not df_incorrect.empty else np.nan
        
        all_results.append({
            'model': display_name, 'family_key': model_family_key, 'color': color,
            'avg_tokens_correct': tokens_correct, 'ece_correct': ece_correct,
            'avg_tokens_incorrect': tokens_incorrect, 'ece_incorrect': ece_incorrect
        })

    # Plotting logic for both subplots
    for ax, metric_suffix, y_label, title in [
        (ax1, 'correct', 'Underconfidence Error (UCE)', 'Correct Predictions'),
        (ax2, 'incorrect', 'Overconfidence Error (OCE)', 'Incorrect Predictions')
    ]:
        valid_results = [r for r in all_results if pd.notna(r[f'avg_tokens_{metric_suffix}']) and pd.notna(r[f'ece_{metric_suffix}'])]
        
        if not valid_results: continue # Skip if no data for this plot

        for result in valid_results:
            x_val = result[f'avg_tokens_{metric_suffix}']
            y_val = result[f'ece_{metric_suffix}']
            ax.scatter(x_val, y_val, marker='o', c=result['color'], s=100, alpha=0.8, edgecolors='white')
            ax.text(x_val + 0.5, y_val + 0.001, result['model'], fontsize=9, alpha=0.9, ha='left')
        
        # Optional trend line (if enough points)
        if len(valid_results) >= 3:
             x_vals = [r[f'avg_tokens_{metric_suffix}'] for r in valid_results]
             y_vals = [r[f'ece_{metric_suffix}'] for r in valid_results]
             try:
                  # Use low degree polynomial fit (e.g., 1 or 2)
                  z = np.polyfit(x_vals, y_vals, 1)
                  p = np.poly1d(z)
                  x_smooth = np.linspace(min(x_vals) * 0.95, max(x_vals) * 1.05, 100)
                  ax.plot(x_smooth, p(x_smooth), '--', color='gray', alpha=0.7, lw=1.5)
             except np.linalg.LinAlgError:
                  print(f"Could not fit trend line for {title}.") # Handle potential fitting errors

        ax.set_xlabel('Average Tokens Used', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.margins(0.1)

    # Create Legend
    unique_families = {r['family_key']: r['color'] for r in all_results if r['family_key'] != 'other'}
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=color, label=key.replace('_', ' ').title(), markersize=8, linestyle='None')
        for key, color in unique_families.items()
    ]
    if any(r['family_key'] == 'other' for r in all_results):
         legend_elements.append(plt.Line2D([0], [0], marker='o', color=default_color, label='Other', markersize=8, linestyle='None'))

    fig.legend(handles=legend_elements, title='Model Families', loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=min(len(legend_elements), 5), fontsize=10, title_fontsize=11)
    
    fig.suptitle('Token Usage vs. Calibration Error (Reasoning Models)', fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout
    
    try:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
        
    return fig

def analyze_and_plot_grouped_calibration_changes(comparisons, output_path='../calibration_changes_grouped.png'):
    """
    Analyzes confidence changes between pairs of model results (Before vs After) 
    and plots pie charts summarizing beneficial/detrimental changes, grouped by comparison type.

    Args:
        comparisons (list): A list of tuples. Each tuple should contain:
                           (df_after, df_before, name_after, name_before)
                           Where df_after/df_before are DataFrames with 'llm_confidence' 
                           and 'Correctness', and names are strings used for grouping.
        output_path (str): Path to save the generated plot PNG.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    group1_counts = np.array([0.0] * 4) # Reasoning vs Non-Reasoning
    group2_counts = np.array([0.0] * 4) # Other (e.g., Long vs Short CoT)
    group1_n, group2_n = 0, 0
    
    required_cols = ['llm_confidence', 'Correctness']
    
    for item in comparisons:
        try:
            df_after, df_before, name_after, name_before = item
        except ValueError:
            print(f"Skipping invalid comparison item: {item}")
            continue
            
        # Basic validation
        if not all(col in df_after.columns and col in df_before.columns for col in required_cols):
             print(f"Skipping comparison ({name_after} vs {name_before}): Missing required columns.")
             continue
        
        # Ensure numeric types, handle potential errors
        for df in [df_after, df_before]:
             df['llm_confidence'] = pd.to_numeric(df['llm_confidence'], errors='coerce')
             df['Correctness'] = pd.to_numeric(df['Correctness'], errors='coerce')
             df.dropna(subset=required_cols, inplace=True)
             df = df[df['Correctness'].isin([0, 1])]
             
        # Align indices
        aligned_indices = df_after.index.intersection(df_before.index)
        if aligned_indices.empty:
            print(f"Skipping comparison ({name_after} vs {name_before}): No common indices after cleaning.")
            continue
            
        df_after_common = df_after.loc[aligned_indices]
        df_before_common = df_before.loc[aligned_indices]
        
        # Determine grouping (Group 1: one is CoT/Reasoning, other is No-Reasoning)
        na, nb = name_after.lower(), name_before.lower()
        is_group1 = ( (any(s in na for s in ['cot', 'reason', 'sc', 'ltm']) and any(s in nb for s in ['noexp', 'no_exp'])) or 
                      (any(s in nb for s in ['cot', 'reason', 'sc', 'ltm']) and any(s in na for s in ['noexp', 'no_exp'])) )
        
        # Calculate counts for this pair
        conf_diff = df_after_common['llm_confidence'] - df_before_common['llm_confidence']
        after_corr = df_after_common['Correctness'] == 1
        
        counts = np.array([
            ((conf_diff > 0) & after_corr).sum(),  # Beneficial Up
            ((conf_diff > 0) & ~after_corr).sum(), # Detrimental Up
            ((conf_diff < 0) & ~after_corr).sum(), # Beneficial Down
            ((conf_diff < 0) & after_corr).sum()   # Detrimental Down
        ], dtype=float)
        
        if is_group1:
            group1_counts += counts
            group1_n += 1
        else:
            group2_counts += counts
            group2_n += 1

    # Compute average counts
    avg_group1 = group1_counts / group1_n if group1_n > 0 else np.zeros(4)
    avg_group2 = group2_counts / group2_n if group2_n > 0 else np.zeros(4)
    total_group1 = avg_group1.sum()
    total_group2 = avg_group2.sum()
    
    labels = ["Conf Up (Benefit)", "Conf Up (Detriment)", "Conf Down (Benefit)", "Conf Down (Detriment)"]
    colors = ['#27ae60', '#c0392b', '#2980b9', '#f39c12'] # Green, Red, Blue, Orange
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#f8f9fa')
    
    pie_settings = {
        'startangle': 90, 'pctdistance': 0.80,
        'wedgeprops': dict(width=0.4, edgecolor='white', linewidth=1),
        'textprops': {'fontsize': 11, 'fontweight': 'bold'}
    }

    # Plot Group 1: Reasoning vs. Non-Reasoning
    if total_group1 > 0:
        wedges1, texts1, autotexts1 = axes[0].pie(avg_group1, labels=labels, colors=colors, autopct='%1.1f%%', **pie_settings)
        plt.setp(autotexts1, size=10, weight="bold", color="white")
    else:
        axes[0].pie([1], labels=['No Data'], colors=['lightgrey'], radius=0.6)
    axes[0].set_title("Reasoning vs. Non-Reasoning", fontsize=14, fontweight='bold', pad=15)
    axes[0].text(0, 0, f"Avg Changed:\n{total_group1:.1f}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Plot Group 2: Other Comparisons (e.g., Long vs Short CoT)
    if total_group2 > 0:
        wedges2, texts2, autotexts2 = axes[1].pie(avg_group2, labels=labels, colors=colors, autopct='%1.1f%%', **pie_settings)
        plt.setp(autotexts2, size=10, weight="bold", color="white")
    else:
        axes[1].pie([1], labels=['No Data'], colors=['lightgrey'], radius=0.6)
    axes[1].set_title("Long CoT vs. Short CoT (Example)", fontsize=14, fontweight='bold', pad=15)
    axes[1].text(0, 0, f"Avg Changed:\n{total_group2:.1f}", ha='center', va='center', fontsize=12, fontweight='bold')
    
    fig.suptitle("Effectiveness of Confidence Changes (Grouped Comparisons)", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
        
    return fig

# --- Rolling Window Plotting Helper & Main Functions --- 

def _plot_metric_by_reasoning(ax, df_subset, metric_col, ylabel, title_suffix, reasoning_types, window_size=100, min_periods=50, tokens_col='completion_tokens'):
    """Internal helper to plot rolling metrics for different reasoning types."""
    color_map = {'No Reasoning': 'red', 'Short CoT': 'blue', 'Long CoT': 'green', 
                 'Short CoTs': 'blue', 'Long CoTs': 'green'} # Map variations
    ls_map = {'No Reasoning': ':', 'Short CoT': '--', 'Long CoT': '-',
              'Short CoTs': '--', 'Long CoTs': '-'}
    marker_map = {'No Reasoning': '.', 'Short CoT': '^', 'Long CoT': 'o', 
                  'Short CoTs': '^', 'Long CoTs': 'o'}
                  
    plotted_types = []
    for rtype in reasoning_types:
        df_r = df_subset[df_subset['Reasoning_Type'] == rtype]
        if df_r.empty or pd.to_numeric(df_r[tokens_col], errors='coerce').isnull().all():
            continue
        
        # Ensure columns are numeric for rolling calculation
        df_r[tokens_col] = pd.to_numeric(df_r[tokens_col], errors='coerce')
        df_r[metric_col] = pd.to_numeric(df_r[metric_col], errors='coerce')
        df_r = df_r.dropna(subset=[tokens_col, metric_col])
        if df_r.empty or len(df_r) < min_periods:
             continue # Skip if not enough data after cleaning

        df_sorted = df_r.sort_values(tokens_col)
        roll_metric = df_sorted[metric_col].rolling(window=window_size, min_periods=min_periods).mean()
        
        ax.plot(df_sorted[tokens_col], roll_metric, 
                label=rtype, 
                lw=2, 
                color=color_map.get(rtype, 'black'),
                linestyle=ls_map.get(rtype, '-'),
                # marker=marker_map.get(rtype), markersize=3, markevery=int(window_size/4) # Optional markers
               )
        plotted_types.append(rtype)
        
    if plotted_types: # Only add labels/legend if something was plotted
        ax.set_xlabel('Completion Tokens', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title_suffix, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
         # Indicate no data plotted
         ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes, color='grey')
         ax.set_title(title_suffix, fontsize=12, fontweight='bold')
         ax.set_xlabel('Completion Tokens', fontsize=11)
         ax.set_ylabel(ylabel, fontsize=11)
         ax.tick_params(labelsize=10)

def plot_calibration_metrics_by_task_type(df_combined, output_path='../calibration_metrics_by_tasktype.pdf', **kwargs):
    """
    Generates plots showing various calibration metrics vs. token count, 
    separated by Task Type (e.g., Sci & Math, Commonsense) and Reasoning Type.

    Args:
        df_combined (pd.DataFrame): DataFrame containing columns like 'Task_Type', 
                                    'Reasoning_Type', 'completion_tokens', 
                                    'llm_confidence', 'Correctness', 
                                    'underconfidence_error', 'overconfidence_error'.
        output_path (str): Path to save the generated plot PDF.
        **kwargs: Additional arguments passed to _plot_metric_by_reasoning (window_size, etc.)

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    task_types = df_combined['Task_Type'].unique()
    reasoning_types = df_combined['Reasoning_Type'].unique()
    num_rows = 3 # Confidence, Accuracy, Calibration Error
    num_cols = len(task_types)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)
    fig.suptitle('Calibration Metrics by Task Type and Reasoning', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('llm_confidence', 'Avg Confidence', 'Average Confidence'),
        ('Correctness', 'Accuracy', 'Accuracy'),
        ('confidence_error', 'Calibration Error', 'Calibration Error') # Requires pre-calculating abs(Correctness - llm_confidence)
        # ('underconfidence_error', 'Underconfidence Error', 'Underconfidence Error (Correct Samples)'), # Requires pre-filtering for Correctness==1
        # ('overconfidence_error', 'Overconfidence Error', 'Overconfidence Error (Incorrect Samples)') # Requires pre-filtering for Correctness==0
    ]
    
    # Ensure necessary error columns exist or calculate them
    if 'confidence_error' not in df_combined.columns and 'confidence_error' in [m[0] for m in metrics_to_plot]:
        df_combined['confidence_error'] = abs(df_combined['Correctness'] - df_combined['llm_confidence'])
    if 'underconfidence_error' not in df_combined.columns and 'underconfidence_error' in [m[0] for m in metrics_to_plot]:
        df_combined['underconfidence_error'] = np.nan
        corr_idx = df_combined['Correctness'] == 1
        df_combined.loc[corr_idx, 'underconfidence_error'] = 1 - df_combined.loc[corr_idx, 'llm_confidence']
    if 'overconfidence_error' not in df_combined.columns and 'overconfidence_error' in [m[0] for m in metrics_to_plot]:
        df_combined['overconfidence_error'] = np.nan
        incorr_idx = df_combined['Correctness'] == 0
        df_combined.loc[incorr_idx, 'overconfidence_error'] = df_combined.loc[incorr_idx, 'llm_confidence']
        
    for col_idx, task_type in enumerate(task_types):
        df_task = df_combined[df_combined['Task_Type'] == task_type]
        for row_idx, (metric_col, ylabel, title_part) in enumerate(metrics_to_plot):
            ax = axes[row_idx, col_idx]
            title = f'{title_part} ({task_type})'
            
            # Handle specific error metrics requiring filtering
            df_plot = df_task
            if metric_col == 'underconfidence_error':
                df_plot = df_task[df_task['Correctness'] == 1]
            elif metric_col == 'overconfidence_error':
                df_plot = df_task[df_task['Correctness'] == 0]
                
            _plot_metric_by_reasoning(ax, df_plot, metric_col, ylabel, title, reasoning_types, **kwargs)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    return fig

def plot_calibration_metrics_by_response_type(df_combined, output_path='../calibration_metrics_by_response.pdf', **kwargs):
    """
    Generates plots showing various calibration metrics vs. token count, 
    separated by Response Type (e.g., MCQA, Free Response) and Reasoning Type.

    Args:
        df_combined (pd.DataFrame): DataFrame containing columns like 'response_type', 
                                    'Reasoning_Type', 'completion_tokens', 
                                    'llm_confidence', 'Correctness', 
                                    'underconfidence_error', 'overconfidence_error'.
        output_path (str): Path to save the generated plot PDF.
        **kwargs: Additional arguments passed to _plot_metric_by_reasoning (window_size, etc.)

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    # Assuming 'response_type' column exists
    if 'response_type' not in df_combined.columns:
        print("Error: 'response_type' column not found in DataFrame.")
        return plt.figure()
        
    response_types = df_combined['response_type'].unique()
    reasoning_types = df_combined['Reasoning_Type'].unique()
    num_rows = 4 # Confidence, Accuracy, Underconfidence, Overconfidence
    num_cols = len(response_types)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), squeeze=False)
    fig.suptitle('Calibration Metrics by Response Type and Reasoning', fontsize=16, fontweight='bold')

    metrics_to_plot = [
        ('llm_confidence', 'Avg Confidence', 'Average Confidence'),
        ('Correctness', 'Accuracy', 'Accuracy'),
        ('underconfidence_error', 'Underconfidence Error', 'Underconfidence Error (Correct Samples)'),
        ('overconfidence_error', 'Overconfidence Error', 'Overconfidence Error (Incorrect Samples)')
    ]
    
    # Ensure necessary error columns exist or calculate them (similar to task type plot)
    if 'underconfidence_error' not in df_combined.columns:
        df_combined['underconfidence_error'] = np.nan
        corr_idx = df_combined['Correctness'] == 1
        df_combined.loc[corr_idx, 'underconfidence_error'] = 1 - df_combined.loc[corr_idx, 'llm_confidence']
    if 'overconfidence_error' not in df_combined.columns:
        df_combined['overconfidence_error'] = np.nan
        incorr_idx = df_combined['Correctness'] == 0
        df_combined.loc[incorr_idx, 'overconfidence_error'] = df_combined.loc[incorr_idx, 'llm_confidence']
        
    for col_idx, resp_type in enumerate(response_types):
        df_resp = df_combined[df_combined['response_type'].str.lower() == str(resp_type).lower()]
        for row_idx, (metric_col, ylabel, title_part) in enumerate(metrics_to_plot):
            ax = axes[row_idx, col_idx]
            title = f'{title_part} ({resp_type})'
            
            df_plot = df_resp
            if metric_col == 'underconfidence_error':
                df_plot = df_resp[df_resp['Correctness'] == 1]
            elif metric_col == 'overconfidence_error':
                df_plot = df_resp[df_resp['Correctness'] == 0]
                
            _plot_metric_by_reasoning(ax, df_plot, metric_col, ylabel, title, reasoning_types, **kwargs)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    return fig


def plot_calibration_error_distributions_by_task(df_combined, output_path='../calibration_error_distributions_by_task.pdf', bins=30):
    """
    Plots the distribution of Overconfidence (for incorrect) and Underconfidence (for correct)
    errors, separated by Task Type.

    Args:
        df_combined (pd.DataFrame): DataFrame with 'Task_Type', 'Correctness', 'llm_confidence'.
        output_path (str): Path to save the generated plot PDF.
        bins (int): Number of bins for the histogram/density calculation.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    if not all(col in df_combined.columns for col in ['Task_Type', 'Correctness', 'llm_confidence']):
        print("Error: Missing required columns for error distribution plot.")
        return plt.figure()
        
    df_combined['llm_confidence'] = pd.to_numeric(df_combined['llm_confidence'], errors='coerce')
    df_combined = df_combined.dropna(subset=['llm_confidence', 'Correctness'])
    df_combined = df_combined[df_combined['Correctness'].isin([0, 1])]

    task_types = df_combined['Task_Type'].unique()
    errors_incorrect = {} 
    errors_correct = {}

    for t in task_types:
        df_t = df_combined[df_combined['Task_Type'] == t]
        errors_incorrect[t] = df_t[df_t['Correctness'] == 0]['llm_confidence'].values
        errors_correct[t] = 1 - df_t[df_t['Correctness'] == 1]['llm_confidence'].values

    def compute_density(errors, bins, range=(0, 1)):
        hist, bin_edges = np.histogram(errors, bins=bins, range=range, density=True)
        return (bin_edges[:-1] + bin_edges[1:]) / 2, hist

    color_map = { # Define colors, add more if needed
        "Sci & Math": "#2ecc71", "Commonsense": "#e74c3c", "Other": "#3498db"
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Overconfidence (Incorrect Answers)
    ax1 = axes[0]
    for t, errs in errors_incorrect.items():
        if len(errs) > 0:
            x, density = compute_density(errs, bins=bins)
            avg_err = np.mean(errs)
            label = f"{t} (Avg={avg_err:.2f})"
            ax1.plot(x, density, label=label, lw=2.5, marker='o', markersize=5, color=color_map.get(t, 'grey'))
    ax1.set_xlabel("Overconfidence Error (Confidence when Incorrect)", fontsize=12)
    ax1.set_ylabel("Density (PDF)", fontsize=12)
    ax1.set_title("Overconfidence Distribution (Incorrect)", fontsize=14, fontweight='bold')
    ax1.legend(title="Task Type", fontsize=10, title_fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(labelsize=11)

    # Plot Underconfidence (Correct Answers)
    ax2 = axes[1]
    for t, errs in errors_correct.items():
        if len(errs) > 0:
            x, density = compute_density(errs, bins=bins)
            avg_err = np.mean(errs)
            label = f"{t} (Avg={avg_err:.2f})"
            ax2.plot(x, density, label=label, lw=2.5, marker='s', markersize=5, color=color_map.get(t, 'grey'))
    ax2.set_xlabel("Underconfidence Error (1 - Confidence when Correct)", fontsize=12)
    ax2.set_ylabel("Density (PDF)", fontsize=12)
    ax2.set_title("Underconfidence Distribution (Correct)", fontsize=14, fontweight='bold')
    ax2.legend(title="Task Type", fontsize=10, title_fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(labelsize=11)

    fig.suptitle("Calibration Error Distributions by Task Type", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    return fig 