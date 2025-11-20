from datasets import load_dataset
from config import PROMPT, EXAMPLES
from extract import extract_text
from converter import convert_to_ae110k_format
from utils import parallel_extract

dataset = load_dataset("av-generation/ae-110k-dataset")
texts = dataset["train"]["text"]

output = "ae110k_extractions.jsonl"
parallel_extract(texts, extract_text, PROMPT, EXAMPLES, output)

output_converted = "ae110k_json_structured.jsonl"
convert_to_ae110k_format(output, output_converted)