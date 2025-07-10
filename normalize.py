import pandas as pd


def normalize_to_minus1_1(df):
    df_normalized = df.copy()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            col_min = df[column].min()
            col_max = df[column].max()
            if col_max != col_min:
                df_normalized[column] = 2 * \
                    ((df[column] - col_min) / (col_max - col_min)) - 1
            else:
                # or leave as is, since there's no variation
                df_normalized[column] = 0
    return df_normalized


# === Step 1: Load CSV ===
# Replace with your actual CSV file path
input_file = 'sim_og.csv'
df = pd.read_csv(input_file)

# === Step 2: Normalize ===
df_normalized = normalize_to_minus1_1(df)

# === Step 3: Save Output ===
output_file = 'sim_og_norm.csv'
df_normalized.to_csv(output_file, index=False)

print("Normalization complete. Saved to:", output_file)
