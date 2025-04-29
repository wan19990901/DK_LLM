import pandas as pd
import numpy as np
import glob
import os
import re

# Assuming evaluation functions might be needed for initial processing
try:
    from .evaluation import evaluate_llm_answers 
except ImportError:
    try:
        from evaluation import evaluate_llm_answers
    except ImportError:
        print("Warning: Could not import evaluate_llm_answers from evaluation.")
        # Define a placeholder if needed
        def evaluate_llm_answers(df): return df # Placeholder

def load_and_process_results(results_dir='data/Results'):
    """
    Loads all CSV files from a specified directory (e.g., 'data/Results/'), 
    performs initial cleaning and processing, and categorizes them.

    Processing includes:
    - Cleaning 'llm_confidence' (handling -1, non-numeric, extracting floats).
    - Extracting 'completion_tokens' from 'token_usage' column.
    - Categorizing DataFrames based on filename conventions (e.g., 'no_exp').
    - Optionally applies initial evaluation (like evaluate_llm_answers).

    Args:
        results_dir (str): Path to the directory containing result CSV files.

    Returns:
        tuple: A tuple containing:
            - dict: All processed DataFrames keyed by a generated df_name.
            - dict: DataFrames categorized as 'reasoning'.
            - dict: DataFrames categorized as 'no_reasoning'.
            - dict: Lists of original filenames for each category ('reasoning', 'no_reasoning').
    """
    csv_files = glob.glob(os.path.join(results_dir, '*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {results_dir}")
        return {}, {}, {}, {'reasoning': [], 'no_reasoning': []}

    all_dfs = {}
    reasoning_dfs = {}
    no_reasoning_dfs = {}
    names = {'reasoning': [], 'no_reasoning': []}

    for file in csv_files:
        filename = os.path.basename(file)
        # Create a safe df name (e.g., df_model_name_condition)
        df_name_base = filename.replace('.csv', '').replace('-', '_').replace('.', '_')
        df_name = f"df_{df_name_base}"
        
        print(f"\nProcessing {filename} -> {df_name}")
        try:
            df = pd.read_csv(file)
            print(f"  Initial rows: {len(df)}")

            # --- Confidence Cleaning --- 
            if 'llm_confidence' not in df.columns:
                 print(f"  Warning: 'llm_confidence' column not found in {filename}. Skipping confidence cleaning.")
            else:
                # Convert to string first to handle mixed types safely
                df['llm_confidence'] = df['llm_confidence'].astype(str)
                
                # Handle direct '-1' strings
                minus_one_mask = df['llm_confidence'] == '-1'
                df.loc[minus_one_mask, 'llm_confidence'] = np.nan # Replace '-1' string with NaN
                
                # Attempt numeric conversion, coerce errors to NaN
                df['llm_confidence'] = pd.to_numeric(df['llm_confidence'], errors='coerce')
                
                # Optional: Attempt to extract floats from remaining NaNs if needed
                nan_mask = df['llm_confidence'].isna()
                if nan_mask.any():
                    print(f"  {nan_mask.sum()} rows with non-numeric confidence initially.")
                    # Add regex extraction logic here if necessary, similar to notebook
                    # Example: Iterate and use re.search(r'-?\\d+\\.?\\d*', val)
                    # For now, just keep them as NaN after coerce
                    
                # Drop rows where confidence is still NaN after cleaning
                initial_rows = len(df)
                df = df.dropna(subset=['llm_confidence'])
                if len(df) < initial_rows:
                     print(f"  Dropped {initial_rows - len(df)} rows due to invalid confidence.")
            
            # --- Token Extraction --- 
            if 'token_usage' in df.columns:
                try:
                    # Attempt 1: Standard completion_tokens
                    df['completion_tokens'] = df['token_usage'].apply(
                        lambda x: eval(str(x)).get('completion_tokens') if pd.notnull(x) and isinstance(eval(str(x)), dict) else np.nan
                    )
                except Exception:
                     # Attempt 2: Fallback for different structure (e.g., output_tokens)
                    try:
                        df['completion_tokens'] = df['token_usage'].apply(
                             lambda x: 1.8 * eval(str(x)).get('output_tokens') if pd.notnull(x) and isinstance(eval(str(x)), dict) else np.nan
                        )
                        print(f"  Used fallback logic (output_tokens * 1.8) for completion_tokens in {filename}")
                    except Exception as e_inner:
                         print(f"  Warning: Could not extract tokens from 'token_usage' in {filename}. Error: {e_inner}")
                         df['completion_tokens'] = np.nan # Assign NaN if extraction fails
                         
                # Optional: Drop rows where tokens couldn't be extracted if needed
                # df = df.dropna(subset=['completion_tokens'])
            else:
                print(f"  Warning: 'token_usage' column not found in {filename}. Cannot extract completion tokens.")
                df['completion_tokens'] = np.nan # Add column with NaN
                
            print(f"  Rows after processing: {len(df)}")
            if df.empty:
                 print(f"  Skipping {filename} as it resulted in an empty DataFrame.")
                 continue
                 
            # --- Initial Evaluation (Example) --- 
            # Uncomment if evaluate_llm_answers should be run during loading
            # try:
            #     df = evaluate_llm_answers(df)
            #     print(f"  Applied initial evaluation (Correctness column added).")
            # except Exception as e_eval:
            #      print(f"  Warning: Failed to apply initial evaluation to {filename}. Error: {e_eval}")
                 
            # --- Categorization --- 
            # Uses simplified logic based on filename suffix
            filename_lower = filename.lower()
            if filename_lower.endswith('no_exp.csv') or filename_lower.endswith('noexp.csv'):
                no_reasoning_dfs[df_name] = df
                names['no_reasoning'].append(filename) # Store original filename
            else:
                reasoning_dfs[df_name] = df
                names['reasoning'].append(filename)
            
            all_dfs[df_name] = df
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue # Skip to next file on error
            
    print("\n--- Loading Summary ---")
    print(f"Total files processed: {len(csv_files)}")
    print(f"Reasoning DFs loaded: {len(reasoning_dfs)}")
    print(f"No-Reasoning DFs loaded: {len(no_reasoning_dfs)}")
    print(f"Total successfully loaded DFs: {len(all_dfs)}")
    
    return all_dfs, reasoning_dfs, no_reasoning_dfs, names

def load_free_response_data(file_path='data/Results/Free_response/final_evaluated_responses.csv'):
    """
    Loads and performs initial processing for the specific free response data CSV.
    
    Processing includes:
    - Renaming 'correctness' to 'Correctness'.
    - Processing confidence (assuming a function like process_confidence exists or is simple).
    - Calculating 'abs_calibration_error'.
    - Extracting 'completion_tokens'.
    
    Args:
        file_path (str): Path to the free response CSV file.
        
    Returns:
        pd.DataFrame or None: Processed DataFrame or None if loading fails.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded free response data from {file_path}. Rows: {len(df)}")

        # Rename column if it exists
        if 'correctness' in df.columns:
            df.rename(columns={'correctness': 'Correctness'}, inplace=True)
            print("  Renamed 'correctness' to 'Correctness'.")
            
        # Ensure Correctness column exists and is numeric (0/1)
        if 'Correctness' not in df.columns:
             print("  Warning: 'Correctness' column missing.")
             return None # Or handle as appropriate
        else:
            df['Correctness'] = pd.to_numeric(df['Correctness'], errors='coerce')
            df = df.dropna(subset=['Correctness'])
            df = df[df['Correctness'].isin([0, 1])]
             
        # Process confidence (basic example: ensure numeric, drop NaN)
        if 'llm_confidence' in df.columns:
            df['llm_confidence'] = pd.to_numeric(df['llm_confidence'], errors='coerce')
            df = df.dropna(subset=['llm_confidence'])
            # Placeholder for a potentially more complex process_confidence function if needed
            print("  Processed 'llm_confidence' (ensured numeric).")
            
            # Calculate calibration error
            df['abs_calibration_error'] = abs(df['Correctness'] - df['llm_confidence'])
            print("  Calculated 'abs_calibration_error'.")
        else:
             print("  Warning: 'llm_confidence' column missing. Cannot calculate calibration error.")
             
        # Extract completion tokens
        if 'token_usage' in df.columns:
            try:
                df['completion_tokens'] = df['token_usage'].apply(
                    lambda x: eval(str(x)).get('completion_tokens') if pd.notnull(x) and isinstance(eval(str(x)), dict) else np.nan
                )
                print("  Extracted 'completion_tokens'.")
            except Exception as e:
                print(f"  Warning: Failed to extract 'completion_tokens' from free response data. Error: {e}")
                df['completion_tokens'] = np.nan
        else:
             print("  Warning: 'token_usage' column missing.")
             df['completion_tokens'] = np.nan
             
        print(f"  Final rows in free response data: {len(df)}")
        return df

    except FileNotFoundError:
        print(f"Error: Free response data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing free response data from {file_path}: {e}")
        return None

# Example Usage (call these from your notebook):
# all_loaded_dfs, reasoning_type_dfs, no_reasoning_type_dfs, category_names = load_and_process_results()
# free_response_df = load_free_response_data() 