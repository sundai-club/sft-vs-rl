import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main(args):
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found at {args.input_csv}")
        return
    try:
        df = pd.read_csv(args.input_csv)
        print(f"Loaded pass@k data from: {args.input_csv}")
        print(df)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    if 'k' not in df.columns or 'pass_k_accuracy' not in df.columns:
        print(f"Error: CSV file must contain 'k' and 'pass_k_accuracy' columns.")
        return

    df = df.sort_values(by='k')

    plt.figure(figsize=(8, 5))
    plt.plot(df['k'], df['pass_k_accuracy'], marker='o', linestyle='-')
    
    plt.title(f'Pass@k Accuracy vs. Number of Samples (k)\n{args.plot_title}')
    plt.xlabel('Number of Samples (k)')
    plt.ylabel('Accuracy (Pass@k)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(df['k']) 
    plt.ylim(0, 1.05)
    try:
        plt.savefig(args.output_plot, bbox_inches='tight')
        print(f"Plot saved to: {args.output_plot}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pass@k results from a CSV file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file containing pass@k results.")
    parser.add_argument("--output_plot", type=str, required=True, help="Path to save the output plot image (e.g., plot.png).")
    parser.add_argument("--plot_title", type=str, default="", help="Optional title suffix for the plot.")

    args = parser.parse_args()
    main(args) 