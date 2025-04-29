import pandas as pd
import numpy as np
# Attempt to import from utils in the same directory
try:
    from .utils import extract_complete_answer
except ImportError:
    # Fallback if running as script or .utils not found
    try:
        from utils import extract_complete_answer
    except ImportError:
        print("Warning: Could not import extract_complete_answer from utils.")
        # Define a placeholder if needed, or raise an error if critical
        def extract_complete_answer(q, a): return a 

def evaluate_llm_answers(df):
    """
    Evaluates LLM answers against correct answers and adds a Correctness column.
    
    - Correctness = 1: If the first letter of llm_answer matches Correct Answer 
                       OR if llm_answer is a substring of the complete_answer.
    - Correctness = 0: Otherwise.
    - Correctness = -1: If the first letter is not a standard option (A-K, N, Y). 
                       (This represents a format error).

    Args:
        df (pd.DataFrame): DataFrame containing 'llm_answer', 'Correct Answer', 
                           and 'Question' columns. It calculates 'complete_answer' internally.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'Correctness' column added.
    """
    if not all(col in df.columns for col in ['llm_answer', 'Correct Answer', 'Question']):
        raise ValueError("Input DataFrame must contain 'llm_answer', 'Correct Answer', and 'Question' columns.")

    # Apply extract_complete_answer if not already present
    if 'complete_answer' not in df.columns:
        df['complete_answer'] = df.apply(
            lambda row: extract_complete_answer(row['Question'], row['Correct Answer']), 
            axis=1
        )

    df['Correctness'] = -1 # Initialize with format error
    stats = {'total': len(df), 'correct': 0, 'incorrect': 0, 'format_errors': 0}
    valid_options = set('ABCDEFGHIJKNY')
    
    for idx, row in df.iterrows():
        llm_answer = str(row['llm_answer']).strip()
        correct_answer_letter = str(row['Correct Answer']).strip()
        complete_answer = str(row.get('complete_answer', '')).strip()
        
        first_letter = llm_answer[0].upper() if llm_answer else ''
        
        if first_letter not in valid_options:
            stats['format_errors'] += 1
            continue
            
        letter_matches = (first_letter == correct_answer_letter.upper())
        content_matches = False
        if complete_answer and (len(llm_answer) > 1 or llm_answer.lower() in ['yes', 'no']):
            content_matches = (llm_answer.lower() in complete_answer.lower())
        
        if letter_matches or content_matches:
            df.loc[idx, 'Correctness'] = 1
            stats['correct'] += 1
        else:
            df.loc[idx, 'Correctness'] = 0
            stats['incorrect'] += 1
            
    valid_answers = stats['total'] - stats['format_errors']
    accuracy = stats['correct'] / valid_answers if valid_answers > 0 else 0
    # print(f"Accuracy (excluding format errors): {accuracy:.2%}") # Optional print
    return df

def compute_adaptive_ece(df, confidence_col='llm_confidence', correctness_col='Correctness', n_bins=10):
    """
    Compute ECE using adaptive binning (equal samples per bin).
    Args: df, confidence_col, correctness_col, n_bins
    Returns: ECE value (float) or np.nan
    """
    df_clean = df[[confidence_col, correctness_col]].copy()
    df_clean[confidence_col] = pd.to_numeric(df_clean[confidence_col], errors='coerce')
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean[correctness_col].isin([0, 1])]

    if df_clean.empty or len(df_clean) < n_bins:
        return np.nan

    df_sorted = df_clean.sort_values(confidence_col)
    n_samples = len(df_sorted)
    ece = 0.0
    indices = np.linspace(0, n_samples, n_bins + 1, dtype=int)

    for i in range(n_bins):
        start_idx, end_idx = indices[i], indices[i+1]
        if start_idx == end_idx: continue
        bin_data = df_sorted.iloc[start_idx:end_idx]
        bin_conf = bin_data[confidence_col].mean()
        bin_acc = bin_data[correctness_col].mean()
        bin_prop = len(bin_data) / n_samples
        ece += bin_prop * abs(bin_acc - bin_conf)
        
    return ece

def compute_mace(df, confidence_col='llm_confidence', correctness_col='Correctness'):
    """
    Compute Mean Absolute Calibration Error (MACE).
    Args: df, confidence_col, correctness_col
    Returns: MACE value (float) or np.nan
    """
    df_clean = df[[confidence_col, correctness_col]].copy()
    df_clean[confidence_col] = pd.to_numeric(df_clean[confidence_col], errors='coerce')
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean[correctness_col].isin([0, 1])]

    if df_clean.empty: return np.nan

    return abs(df_clean[confidence_col] - df_clean[correctness_col]).mean()

def compute_group_metrics(dfs_dict, n_bins=10, confidence_col='llm_confidence', correctness_col='Correctness', tokens_col='completion_tokens', group_col='Name'):
    """
    Computes evaluation metrics for a dict of DataFrames, grouping by model family.
    Args: dfs_dict, n_bins, confidence_col, correctness_col, tokens_col, group_col
    Returns: Summary pd.DataFrame
    """
    summary_list = []
    
    for variant, df_orig in dfs_dict.items():
        df = df_orig.copy()
        required = [confidence_col, correctness_col, tokens_col]
        if group_col and group_col in df.columns: required.append(group_col)
        
        if not all(col in df.columns for col in required if col):
            print(f"Warning: Missing columns in '{variant}'. Skipping.")
            continue
            
        df[correctness_col] = pd.to_numeric(df[correctness_col], errors='coerce')
        df = df.dropna(subset=[correctness_col])
        df = df[df[correctness_col].isin([0, 1])]
        if df.empty: continue

        accuracy = df[correctness_col].mean()
        ece = compute_adaptive_ece(df, confidence_col, correctness_col, n_bins)
        avg_tokens = pd.to_numeric(df[tokens_col], errors='coerce').mean()
        
        variant_summary = {"Variant": variant, "Accuracy": accuracy, "ECE": ece, "AvgCompletionTokens": avg_tokens}
        
        if group_col and group_col in df.columns:
            # Simplified category metrics aggregation
            category_metrics = df.groupby(group_col).agg(
                Category_Accuracy=(correctness_col, 'mean'),
                Category_AvgTokens=(tokens_col, lambda x: pd.to_numeric(x, errors='coerce').mean())
            ).round(3)
            variant_summary["CategoryMetrics"] = category_metrics # Store for potential later use
        
        summary_list.append(variant_summary)
    
    if not summary_list: return pd.DataFrame()

    summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != "CategoryMetrics"} for s in summary_list])
    summary_df['Family'] = summary_df['Variant'].apply(lambda x: x.rsplit('_', 1)[0] if '_' in x else x)
    summary_df['RelativeTokensDiff'] = np.nan
    
    for family, group in summary_df.groupby('Family'):
        baseline = group[group['Variant'].str.contains('noexp$|low$', case=False, regex=True)]
        if not baseline.empty:
            baseline_tokens = baseline.iloc[0]['AvgCompletionTokens']
            if pd.notna(baseline_tokens):
                fam_idx = summary_df['Family'] == family
                summary_df.loc[fam_idx, 'RelativeTokensDiff'] = summary_df.loc[fam_idx, 'AvgCompletionTokens'] - baseline_tokens
            
    return summary_df

def analyze_confidence_calibration_ratios(dfs_dict, confidence_col='llm_confidence', correctness_col='Correctness', tokens_col='completion_tokens'):
    """
    Calculates calibration ratios (perfect, over, under) for a dict of dataframes.
    Args: dfs_dict, confidence_col, correctness_col, tokens_col
    Returns: Results pd.DataFrame
    """
    results = []
    for df_name, df_orig in dfs_dict.items():
        df_clean = df_orig[[confidence_col, correctness_col, tokens_col]].copy()
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
        df_clean = df_clean.dropna(subset=[confidence_col, correctness_col])
        df_clean = df_clean[df_clean[correctness_col].isin([0, 1])]
        if df_clean.empty: continue

        conf_thresh = df_clean[confidence_col].mean()
        avg_tokens = df_clean[tokens_col].mean()
        avg_correct = df_clean[correctness_col].mean()
        
        name_lower = df_name.lower()
        if name_lower.endswith('_reason'): r_type = 'Reasoning'
        elif any(name_lower.endswith(x) for x in ['_cot', '_sc', '_ltm']): r_type = 'Prompt-based'
        elif name_lower.endswith('noexp') or name_lower.endswith('no_exp'): r_type = 'No Reasoning'
        else: r_type = 'Other'
        
        total = len(df_clean)
        is_correct = df_clean[correctness_col] == 1
        conf_above = df_clean[confidence_col] >= conf_thresh
        
        perfect = ((is_correct & conf_above) | (~is_correct & ~conf_above)).sum()
        over = (~is_correct & conf_above).sum()
        under = (is_correct & ~conf_above).sum()
        
        display_name = df_name.replace('df_', '').replace('openai_', '') # Basic cleaning
        
        results.append({
            'Model Name': display_name, 'Reasoning Type': r_type, 'Total Samples': total,
            'Perfect Ratio': perfect / total if total else 0,
            'Over Ratio': over / total if total else 0,
            'Under Ratio': under / total if total else 0,
            'Confidence Threshold': conf_thresh, 'Avg Tokens': avg_tokens, 'Avg Correctness': avg_correct
        })
    
    if not results: return pd.DataFrame()
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['Reasoning Type', 'Perfect Ratio'], ascending=[True, False])
    # Add optional printing here if desired
    return results_df 