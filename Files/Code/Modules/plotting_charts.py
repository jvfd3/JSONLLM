
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

    # Fixing size
    # Set fixed x-axis: 0 to 1 with steps of 0.1
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))

    # Set fixed y-axis: 0 to 200
    plt.ylim(0, 200)
    plt.yticks(np.arange(0, 201, 20))

    plt.tight_layout()
    # Save Figure as PNG, SVG, and PDF
    plt.savefig(f"{out_path}.png")
    plt.savefig(f"{out_path}.svg")
    plt.savefig(f"{out_path}.pdf")
    plt.close()


def plot_results(df: pd.DataFrame, especific: str = '') -> None:
    """ Plot similarity results. """

    out_dir = "Charts"
    plot_and_save_hist(
        df['text_vs_gt_similarity_score'],
        title=f'Similarity [{especific}]: Text vs GT',
        out_path=os.path.join(out_dir, f'{especific}_text_vs_gt')
    )

    plot_and_save_hist(
        df['text_vs_lx_similarity_score'],
        title=f'Similarity [{especific}]: Text vs LX',
        out_path=os.path.join(out_dir, f'{especific}_text_vs_lx')
    )

    plot_and_save_hist(
        df['gt_vs_lx_similarity_score'],
        title=f'Similarity [{especific}]: GT vs LX',
        out_path=os.path.join(out_dir, f'{especific}_gt_vs_lx')
    )
