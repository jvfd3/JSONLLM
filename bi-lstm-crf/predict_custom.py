import torch
import json
from bi_lstm_crf.model.model import BiLSTM_CRF
from bi_lstm_crf.app.preprocessing.preprocess import Preprocessor

MODEL_DIR = "model_xxx"
CORPUS_DIR = "corpus_dir"

sentence = "APG Camping Stove Gas Fuel"

with open(f"{MODEL_DIR}/vocab.json", encoding="utf-8") as f:
    vocab = json.load(f)
with open(f"{MODEL_DIR}/tags.json", encoding="utf-8") as f:
    tags = json.load(f)

preprocessor = Preprocessor(config_dir=CORPUS_DIR, verbose=False)
tagset_size = len(tags)
vocab_size = len(vocab)

model = BiLSTM_CRF(vocab_size=vocab_size, tagset_size=tagset_size)
model.load_state_dict(torch.load(f"{MODEL_DIR}/model.pth", map_location="cpu"))
model.eval()

tokens = sentence.split()
x = torch.tensor([preprocessor.sent_to_vector(tokens)], dtype=torch.long)

with torch.no_grad():
    predicted = model(x)
    predicted_tags = preprocessor.decode_tags(predicted)