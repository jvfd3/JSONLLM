
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_and_save_hist(sims, title, out_path):
    """ Plot and save histogram of similarities. """

    plt.figure()
    plt.hist(sims, bins=40)
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # Save Figure as PNG, SVG, and PDF
    plt.savefig(f"{out_path}.png")
    plt.savefig(f"{out_path}.svg")
    plt.savefig(f"{out_path}.pdf")
    plt.close()


def plot_results(df: pd.DataFrame) -> None:
    """ Plot similarity results. """

    out_dir = "Charts"
    plot_and_save_hist(
        df['text_vs_gt_similarity_score'],
        title='Similarity: Text vs GT',
        out_path=os.path.join(out_dir, 'text_vs_gt')
    )

    plot_and_save_hist(
        df['text_vs_lx_similarity_score'],
        title='Similarity: Text vs LX',
        out_path=os.path.join(out_dir, 'text_vs_lx')
    )

    plot_and_save_hist(
        df['gt_vs_lx_similarity_score'],
        title='Similarity: GT vs LX',
        out_path=os.path.join(out_dir, 'gt_vs_lx')
    )
