
import matplotlib.pyplot as plt
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


def plot_results(sims: dict[str, np.ndarray]) -> None:
    """ Plot similarity results. """

    out_dir = "Charts"
    plot_and_save_hist(
        sims['text_vs_gt'],
        title='Similarity: Text vs GT',
        out_path=os.path.join(out_dir, "text_vs_gt")
    )

    plot_and_save_hist(
        sims['text_vs_lx'],
        title='Similarity: Text vs LX',
        out_path=os.path.join(out_dir, "text_vs_lx")
    )

    plot_and_save_hist(
        sims['gt_vs_lx'],
        title='Similarity: GT vs LX',
        out_path=os.path.join(out_dir, "gt_vs_lx")
    )
