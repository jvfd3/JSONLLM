import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from ..hyperparameters import get_hyperparameters

HYPER = get_hyperparameters()['evaluation']


def batch_embed(texts: list[str], batch_size: int = 32, desc: str = "Embeddings") -> np.ndarray:
    """ Generate embeddings for a list of texts in batches. """
    EMBEDDER = SentenceTransformer(HYPER['embedder_model'])

    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i+batch_size]
        e = EMBEDDER.encode(
            batch,
            batch_size=len(batch),
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        embs.append(e.cpu().numpy())
    return np.vstack(embs)
