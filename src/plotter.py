import re
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob

scan_root = os.path.join(os.getcwd(), 'wfa_logs')
scan_dir = scan_root

target_dir = [
    # placeholder for specific file targeting
]

output_dir = "wfa_plots"


def parse_masterlog(log_path):
    if not os.path.exists(log_path):
        print(f"!!! error: File not found at {log_path} !!!")
        return pd.DataFrame()

    print(f"reading log file: {log_path} ...")
    fallback_hurst = "N/A"
    try:
        match = re.search(r'(\d+\.\d+)\s*HURST', log_path, re.IGNORECASE) or \
                re.search(r'HURST\s*(\d+\.\d+)', log_path, re.IGNORECASE)
        if match: fallback_hurst = float(match.group(1))
    except:
        pass

    data_rows = []
    current_entry = {}
    in_file_block = False

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("FILE:"):
                if in_file_block and 'Sharpe' in current_entry and 'MDD' in current_entry:
                    if 'Hurst_Threshold' not in current_entry: current_entry['Hurst_Threshold'] = fallback_hurst
                    data_rows.append(current_entry)
                current_entry = {}
                filename = line.replace("FILE:", "").strip()
                current_entry['Filename'] = filename
                parts = filename.split('_')
                current_entry['Pair'] = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else "Unknown"
                in_file_block = True
                continue
            if not in_file_block: continue
            # very big elif chain
            if line.startswith("Timeframe Used:"):
                try:
                    parts = line.split(',')
                    val = int(parts[1].split(':')[1].strip())
                    current_entry['Timeframe'] = '15m' if val == 15 else ('1h' if val == 1 else 'Other')
                except:
                    current_entry['Timeframe'] = 'Unknown'
            elif line.startswith("ADF Cointegration Filter Threshold:"):
                try:
                    current_entry['ADF_Threshold'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith("Hurst Filter Threshold:"):
                try:
                    current_entry['Hurst_Threshold'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith("Total Trades Executed:"):
                try:
                    current_entry['Trades'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith("Total PnL:"):
                try:
                    current_entry['PnL'] = float(line.split('$')[1].strip().replace(',', ''))
                except:
                    pass
            elif line.startswith("annaulized sharpe:"):
                try:
                    val = line.split(':')[1].strip()
                    current_entry['Sharpe'] = -999.0 if val.lower() in ['inf', '-inf', 'nan'] else float(
                        val.replace(',', ''))
                except:
                    pass
            elif line.startswith("MDD:"):
                try:
                    current_entry['MDD'] = float(line.split('$')[1].strip().replace(',', ''))
                except:
                    pass

    if in_file_block and 'Sharpe' in current_entry and 'MDD' in current_entry:
        if 'Hurst_Threshold' not in current_entry: current_entry['Hurst_Threshold'] = fallback_hurst
        data_rows.append(current_entry)
    return pd.DataFrame(data_rows)


# adding table so i dont have to interpret the best performers manually
def top25_table(df_subset):
    if df_subset.empty:
        return [], []
    grouped = df_subset.groupby('Pair')['Sharpe'].median().reset_index()
    grouped = grouped.sort_values('Sharpe', ascending=False).head(25)
    return [grouped['Pair'].tolist(), grouped['Sharpe'].round(3).tolist()]


def build_slider(steps):
    slider = [{
        "steps": [],
        "active": 0,
        "currentvalue": {"prefix": "Filter (Min Trades): "},
        "pad": {"t": 50}
    }]

    for t in steps:
        slider[0]["steps"].append({
            "method": "animate",
            "label": str(t),
            "args": [
                [str(t)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0}
                }
            ]
        })

    return slider


def apply_layout(fig, df_plot, outname, sliders):
    # figure out the hurst label (default if messy)
    hurst_val = "N/A"
    if "Hurst_Threshold" in df_plot.columns:
        vals = [v for v in df_plot["Hurst_Threshold"].unique() if str(v) != "N/A"]
        if vals:
            hurst_val = vals[0]
            try:
                hurst_val = f"{float(hurst_val):.2f}"
            except:
                pass

    fig.update_layout(
        title=f"<b>WFA Analysis: Hurst {hurst_val}</b><br><i>{outname}</i>",
        height=900,
        sliders=sliders,
        template="plotly_white",
        hovermode="closest"
    )

    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="x domain",
        y0=0, y1=0, yref="y",
        line={"color": "red", "width": 1, "dash": "dash"},
        row=1, col=1
    )


def create_plot(df, outputfile):
    if df.empty:
        print(f"dataset empty for {outputfile}")
        return

    df_plot = df[df['Sharpe'] > -10].copy()
    df_plot['Pair_Code'] = df_plot['Pair'].astype('category').cat.codes

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        specs=[[{"type": "scatter"}, {"type": "table"}]],
        horizontal_spacing=0.02,
        subplot_titles=("Performance Scatter", "Top 25 Pairs (Median Sharpe)")
    )

    max_trades = df_plot['Trades'].max() if not df_plot.empty else 10
    step_size = 5 if max_trades < 100 else 10
    sliderStep = list(range(0, int(max_trades) + 10, step_size))
    marker_size_ref = 2. * max_trades / (40. ** 2)
    # honestly, alot of this graph stuff, not sure how i got here
    scatter_trace = go.Scatter(
        x=df_plot['MDD'],
        y=df_plot['Sharpe'],
        mode='markers',
        text=df_plot['Pair'],
        customdata=np.stack((df_plot['Pair'], df_plot['Trades'], df_plot['PnL']), axis=-1),
        marker=dict(
            size=df_plot['Trades'], sizemode='area', sizeref=marker_size_ref,
            color=df_plot['Pair_Code'], colorscale='Turbo', showscale=False,
            line=dict(width=1, color='DarkSlateGrey'), opacity=0.8
        ),
        hovertemplate='<b>%{customdata[0]}</b><br>Sharpe: %{y:.2f}<br>MDD: $%{x:,.0f}<br>Trades: %{customdata[1]}<br>PnL: $%{customdata[2]:,.0f}<extra></extra>'
    )

    pair_col, sharpe_col = top25_table(df_plot)
    table_trace = go.Table(
        header=dict(values=['<b>Pair</b>', '<b>Med. Sharpe</b>'], fill_color='paleturquoise', align='left'),
        cells=dict(values=[pair_col, sharpe_col], fill_color='lavender', align='left')
    )

    fig.add_trace(scatter_trace, row=1, col=1)
    fig.add_trace(table_trace, row=1, col=2)

    frames = []
    print("generating frames for slider interaction...")
    for min_t in sliderStep:
        sub_df = df_plot[df_plot['Trades'] >= min_t]
        t_pairs, t_sharpes = top25_table(sub_df)

        frame_trace_scatter = go.Scatter(
            x=sub_df['MDD'], y=sub_df['Sharpe'],
            customdata=np.stack((sub_df['Pair'], sub_df['Trades'], sub_df['PnL']), axis=-1),
            marker=dict(
                size=sub_df['Trades'], sizemode='area', sizeref=marker_size_ref,
                color=sub_df['Pair_Code'], colorscale='Turbo', showscale=False,
                line=dict(width=1, color='DarkSlateGrey'), opacity=0.8
            ),
            text=sub_df['Pair']
        )

        frame_trace_table = go.Table(
            header=dict(values=['<b>Pair</b>', '<b>Med. Sharpe</b>'],
                        fill_color='paleturquoise', align='left'),
            cells=dict(values=[t_pairs, t_sharpes], fill_color='lavender', align='left')
        )

        frames.append(go.Frame(data=[frame_trace_scatter, frame_trace_table], name=str(min_t), traces=[0, 1]))

    fig.frames = frames
    sliders = build_slider(sliderStep)
    apply_layout(fig, df_plot, outputfile, sliders)

    output_path = os.path.join(output_dir, outputfile)
    fig.write_html(output_path, auto_open=False)
    print(f"plot saved: {output_path}")


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    target_files = []

    if scan_dir and os.path.exists(scan_dir):
        print(f"scanning directory: {scan_dir}")
        target_files = glob.glob(os.path.join(scan_dir, "**", "master_compiled*.txt"), recursive=True)
    else:
        target_files = target_dir

    if not target_files:
        print("no log files found.")
    else:
        print(f"found {len(target_files)} files to process.")
        for log_file in target_files:
            base_name = os.path.splitext(os.path.basename(log_file))[0]
            parent_folder = os.path.basename(os.path.dirname(log_file)).replace("!", "").strip()
            html_name = f"plot_{parent_folder}_{base_name}.html".replace(" ", "_")

            print(f"\nProcessing: {base_name}")
            df = parse_masterlog(log_file)

            if df is not None:
                create_plot(df, html_name)

        print(f"\nall plots generated in '{output_dir}'")
