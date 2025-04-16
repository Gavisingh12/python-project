import os
import re
from flask import Flask, render_template, url_for
import pandas as pd
import numpy as np
from scipy import stats

# Force Matplotlib to use a non‐interactive backend (no GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
# Point this at your CSV file:
DATA_PATH = r"D:\python-project  ca 2\sbm_ODF_Status_district.csv"

app = Flask(__name__)
# Save images under your project's static folder
app.config['STATIC_IMAGE_PATH'] = os.path.join(app.root_path, 'static', 'images')


def load_data():
    # 1. Load
    df = pd.read_csv(DATA_PATH)

    # 2. Clean column names
    df.columns = [col.strip() for col in df.columns]
    std_cols = {col: col.lower().replace(' ', '_') for col in df.columns}
    df.rename(columns=std_cols, inplace=True)

    # 3. Auto‐detect key columns
    cols = df.columns.tolist()
    # District
    district_cols = [c for c in cols if 'district' in c]
    if not district_cols:
        raise KeyError("No 'district' column found in your CSV")
    district_col = district_cols[0]
    # Total Blocks
    total_cols = [c for c in cols if re.search(r'total.*block', c)]
    if not total_cols:
        raise KeyError("No 'total blocks' column found in your CSV")
    total_col = next((c for c in total_cols if c == 'total_blocks'), total_cols[0])
    # ODF Blocks
    odf_cols = [c for c in cols if re.search(r'odf.*block', c)]
    if not odf_cols:
        raise KeyError("No 'ODF blocks' column found in your CSV")
    odf_col = next((c for c in odf_cols if c == 'odf_blocks'), odf_cols[0])

    # 4. Rename to standard
    df.rename(columns={
        district_col: 'district',
        total_col: 'total_blocks',
        odf_col: 'odf_blocks'
    }, inplace=True)

    # 5. Dedupe & forward‐fill
    df.drop_duplicates(inplace=True)
    df.ffill(inplace=True)

    return df


def preprocess(df):
    # Ensure numeric types
    df['total_blocks'] = pd.to_numeric(df['total_blocks'], errors='coerce')
    df['odf_blocks'] = pd.to_numeric(df['odf_blocks'], errors='coerce')
    df.dropna(subset=['total_blocks', 'odf_blocks'], inplace=True)

    # Compute ratio
    df['odf_ratio'] = df['odf_blocks'] / df['total_blocks']
    return df


def compute_stats(df):
    # Ensure the DataFrame is not empty to avoid errors/NaN results
    if df.empty:
        return {
            'mean_total_blocks': 0, 'median_total_blocks': 0, 'std_total_blocks': 0,
            'mean_odf_blocks': 0, 'median_odf_blocks': 0, 'std_odf_blocks': 0,
            'mean_odf_ratio': 0, 'median_odf_ratio': 0, 'std_odf_ratio': 0,
        }
    return {
        'mean_total_blocks': df['total_blocks'].mean(),
        'median_total_blocks': df['total_blocks'].median(),
        'std_total_blocks': df['total_blocks'].std(),
        'mean_odf_blocks': df['odf_blocks'].mean(),
        'median_odf_blocks': df['odf_blocks'].median(),
        'std_odf_blocks': df['odf_blocks'].std(),
        'mean_odf_ratio': df['odf_ratio'].mean(),
        'median_odf_ratio': df['odf_ratio'].median(),
        'std_odf_ratio': df['odf_ratio'].std(),
    }


def detect_outliers_zscore(df, threshold=3):
    cols_to_check = ['total_blocks', 'odf_blocks']
    # Handle empty DataFrame or missing columns gracefully
    if df.empty or not all(col in df.columns for col in cols_to_check):
        # Return an empty DataFrame with expected columns for consistency
        return pd.DataFrame(columns=df.columns.tolist() + ['zscore_total', 'zscore_odf'])

    numerical_data = df[cols_to_check]

    # Handle case where selected columns result in an empty frame (e.g., all NaNs previously)
    if numerical_data.empty:
         return pd.DataFrame(columns=df.columns.tolist() + ['zscore_total', 'zscore_odf'])

    # Calculate z-scores. This assumes no NaNs in these columns due to prior preprocessing.
    # Using .values ensures we get a NumPy array for consistent indexing
    z_scores_array = np.abs(stats.zscore(numerical_data.values))

    # Create a temporary DataFrame for z-scores with the same index as numerical_data
    z_df = pd.DataFrame(z_scores_array, index=numerical_data.index, columns=['zscore_total', 'zscore_odf'])

    # Join the z-scores back to the original DataFrame to keep all original data
    df_with_z = df.join(z_df)

    # Filter based on the threshold using the joined DataFrame
    # Handle potential NaNs in z-score columns if stats.zscore produced them unexpectedly
    outliers = df_with_z[
        (df_with_z['zscore_total'].fillna(0) > threshold) |
        (df_with_z['zscore_odf'].fillna(0) > threshold)
    ]

    return outliers # Return only the outlier rows, now including z-score columns


def detect_outliers_iqr(df):
    cols_to_check = ['total_blocks', 'odf_blocks']
    # Handle empty DataFrame or missing columns gracefully
    if df.empty or not all(col in df.columns for col in cols_to_check):
        return pd.DataFrame(columns=df.columns) # Return empty df matching input columns

    numerical_data = df[cols_to_check]

    if numerical_data.empty:
        return pd.DataFrame(columns=df.columns)

    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create the mask using the bounds. Handles NaNs correctly during comparison.
    mask = ((numerical_data < lower_bound) | (numerical_data > upper_bound)).any(axis=1)

    # Filter the original DataFrame using the mask
    return df[mask]


def save_plots(df, outliers_z): # Accept pre-calculated outliers
    os.makedirs(app.config['STATIC_IMAGE_PATH'], exist_ok=True)

    # --- Bar chart: Total vs ODF Blocks ---
    plt.figure(figsize=(14, 7)) # Wider figure for potentially many districts
    if not df.empty:
        # Sort by total blocks for potentially better visualization
        df_sorted = df.sort_values('total_blocks', ascending=False)
        # Use distinct colors
        sns.barplot(x='district', y='total_blocks', data=df_sorted, color='skyblue', label='Total Blocks')
        sns.barplot(x='district', y='odf_blocks', data=df_sorted, color='lightgreen', alpha=0.9, label='ODF Blocks')
        plt.xticks(rotation=90)
        plt.ylabel('Number of Blocks')
        plt.xlabel('District')
        plt.title('Total Blocks vs ODF Blocks per District')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No data available for Bar Chart', ha='center', va='center')
        plt.title('Total Blocks vs ODF Blocks per District')

    plt.tight_layout()
    plt.savefig(os.path.join(app.config['STATIC_IMAGE_PATH'], 'blocks_comparison.png'))
    plt.close()

    # --- Histogram: ODF Ratio ---
    plt.figure(figsize=(8, 6))
    if not df.empty and 'odf_ratio' in df.columns and not df['odf_ratio'].isnull().all():
        sns.histplot(df['odf_ratio'].dropna(), bins=20, kde=True) # Ensure NaNs dropped
        plt.xlabel('ODF Ratio (ODF Blocks / Total Blocks)')
        plt.title('Distribution of ODF Ratio')
    else:
        plt.text(0.5, 0.5, 'No data available for ODF Ratio Histogram', ha='center', va='center')
        plt.title('Distribution of ODF Ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(app.config['STATIC_IMAGE_PATH'], 'odf_ratio_hist.png'))
    plt.close()

    # --- Boxplots ---
    plt.figure(figsize=(8, 6))
    cols_for_boxplot = ['total_blocks', 'odf_blocks']
    if not df.empty and all(col in df.columns for col in cols_for_boxplot):
         # Drop rows where *both* columns are NaN for boxplot if any survived preprocessing
        plot_data = df[cols_for_boxplot].dropna(how='all')
        if not plot_data.empty:
             sns.boxplot(data=plot_data)
             plt.title('Box Plot of Total Blocks and ODF Blocks')
        else:
             plt.text(0.5, 0.5, 'No numerical data for Box Plot', ha='center', va='center')
             plt.title('Box Plot of Total Blocks and ODF Blocks')
    else:
        plt.text(0.5, 0.5, 'No data available for Box Plot', ha='center', va='center')
        plt.title('Box Plot of Total Blocks and ODF Blocks')

    plt.tight_layout()
    plt.savefig(os.path.join(app.config['STATIC_IMAGE_PATH'], 'boxplots.png'))
    plt.close()

    # --- Scatter with highlighted outliers ---
    plt.figure(figsize=(8, 6))
    if not df.empty and all(col in df.columns for col in ['total_blocks', 'odf_blocks']):
        plt.scatter(df['total_blocks'], df['odf_blocks'], label='Data', alpha=0.6)
        # Check if outliers_z DataFrame is not empty before plotting
        if not outliers_z.empty:
             plt.scatter(outliers_z['total_blocks'], outliers_z['odf_blocks'],
                        edgecolor='r', facecolors='none', s=100, label='Outliers (Z-score)')
        plt.xlabel('Total Blocks')
        plt.ylabel('ODF Blocks')
        plt.title('Total Blocks vs ODF Blocks (Outliers Highlighted)')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No data available for Scatter Plot', ha='center', va='center')
        plt.title('Total Blocks vs ODF Blocks (Outliers Highlighted)')

    plt.tight_layout()
    plt.savefig(os.path.join(app.config['STATIC_IMAGE_PATH'], 'scatter_outliers.png'))
    plt.close()


@app.route('/')
def index():
    try:
        df = load_data()
        df = preprocess(df) # This step might result in an empty DataFrame

        # Handle potentially empty DataFrame after preprocessing
        if df.empty:
            # Prepare default/empty data for the template
            stats_summary = compute_stats(df) # Will return zeros
            # Use empty DataFrames for outliers
            outliers_z = pd.DataFrame(columns=df.columns.tolist() + ['zscore_total', 'zscore_odf'])
            outliers_iqr = pd.DataFrame(columns=df.columns)
            # Generate plots (they will show "No data" messages)
            save_plots(df, outliers_z)

        else:
            # Proceed with calculations only if df is not empty
            stats_summary = compute_stats(df)
            # Calculate outliers without modifying the original df
            outliers_z = detect_outliers_zscore(df.copy()) # Pass a copy to be safe
            outliers_iqr = detect_outliers_iqr(df.copy()) # Pass a copy to be safe
            # Generate plots using the main df and the calculated z-score outliers
            save_plots(df, outliers_z)

        # Render the template with calculated or default data
        return render_template(
            'index.html',
            stats=stats_summary,
            # Convert potentially empty DataFrames to dicts safely
            outliers_z=outliers_z.to_dict(orient='records'),
            outliers_iqr=outliers_iqr.to_dict(orient='records'),
            # Pass image paths relative to static folder
            plot_urls={
                'blocks_comparison': url_for('static', filename='images/blocks_comparison.png'),
                'odf_ratio_hist': url_for('static', filename='images/odf_ratio_hist.png'),
                'boxplots': url_for('static', filename='images/boxplots.png'),
                'scatter_outliers': url_for('static', filename='images/scatter_outliers.png')
            }
        )

    except KeyError as e:
        # Handle specific errors like missing columns during loading
        # You might want to render an error template or flash a message
        error_message = f"Data loading error: {e}. Please check the CSV file format."
        # Log the error for debugging
        app.logger.error(error_message)
        # For now, return a simple error message (replace with error template later)
        return f"<h2>Error</h2><p>{error_message}</p>", 500
    except FileNotFoundError:
        error_message = f"Data file not found at {DATA_PATH}. Please ensure the file exists."
        app.logger.error(error_message)
        return f"<h2>Error</h2><p>{error_message}</p>", 500
    except Exception as e:
        # Catch other potential errors during processing or plotting
        error_message = f"An unexpected error occurred: {e}"
        app.logger.exception("Unhandled exception in index route:") # Logs the full traceback
        return f"<h2>Error</h2><p>{error_message}</p>", 500


if __name__ == '__main__':
    # Add basic logging configuration
    import logging
    logging.basicConfig(level=logging.INFO)
    # Ensure the static image directory exists before starting the app
    image_dir = os.path.join(app.root_path, 'static', 'images')
    os.makedirs(image_dir, exist_ok=True)
    app.run(debug=True) # Keep debug=True for development
