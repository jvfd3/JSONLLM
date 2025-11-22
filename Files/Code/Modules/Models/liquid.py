"""
Liquid module for handling language models and pipelines.
"""

# from Modules.hyperparameters import get_hyperparameters
from ..hyperparameters import get_hyperparameters  # local run

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import Dict

HYPER = get_hyperparameters()
LIQUID_PARAMS = HYPER['model']['liquid']


def get_tokenizer():
    """
    Load and return the tokenizer for the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=LIQUID_PARAMS['model_name']
    )
    return tokenizer


def get_model():
    """
    Load and return the language model for the specified model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=LIQUID_PARAMS['model_name'],
        device_map='auto',
        dtype='bfloat16',
        # attn_implementation='flash_attention_2' # <- uncomment on compatible GPU
    )
    return model


def decode_outputs(tokenizer, outputs, inputs) -> str:
    '''
    Decode the generated outputs from the model.
    '''
    # inputs_ids = inputs['input_ids']
    # inputs_ids_shapes = inputs_ids.shape
    # last_shape = inputs_ids_shapes[-1]
    # first_output = outputs[0]

    # to_decode = first_output[last_shape:]
    # decoded = tokenizer.decode(
    #     to_decode,
    #     skip_special_tokens=False
    # )
    # return decoded
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def liquid_desc_to_json_OG(product_description: str) -> Dict:
    '''
    Convert a product description into a structured JSON format using a language model pipeline.
    '''
    tokenizer = get_tokenizer()
    model = get_model()

    system_prompt = HYPER['prompts']['system']
    description = system_prompt['description']
    structure = system_prompt['structure']
    system_prompt_msg = f"{description}\n{structure}"

    messages = [
        {'role': 'system', 'content': system_prompt_msg},
        {'role': 'user', 'content': product_description},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=LIQUID_PARAMS['add_generation_prompt'],  # True
        tokenize=LIQUID_PARAMS['tokenize'],  # True
        # return_dict=LIQUID_PARAMS['return_dict'],  # True # Commented to test if the generation returns the ending bracket
        return_tensors=LIQUID_PARAMS['return_tensors'],  # 'pt'
    ).to(model.device)

    outputs = model.generate(
        inputs,
        do_sample=False,
        max_new_tokens=LIQUID_PARAMS['max_new_tokens']
    )
    decoded = decode_outputs(tokenizer, outputs, inputs)
    return decoded


def liquid_desc_to_json(product_description: str) -> Dict:
    '''
    Convert a product description into a structured JSON format using a language model pipeline.
    '''
    # Load model and tokenizer
    model_id = "LiquidAI/LFM2-350M-Extract"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype="bfloat16",
        # attn_implementation="flash_attention_2" # <- uncomment on compatible GPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create message
    # SYSTEM_PROMPT = HYPER['prompts']['system']

    system_prompt = """
You are an information extraction system for e-commerce product descriptions.

Your task is to extract attributes and output them as a **strict, valid JSON object**.

Follow these REQUIRED rules:

1. The output must be a valid JSON object.
2. Keys MUST:
   - be in English;
   - start with a Capital letter;
3. Values MUST:
   - be flat strings (no lists, no nested JSON, no arrays, no objects);
   - contain only textual descriptions extracted from the input.
4. You MUST NOT:
   - generate lists [] under any key,
   - generate nested JSON objects {},
   - invent attributes not present or implied by the input,
   - include commentary or explanations.
5. The output must contain ONLY the JSON. No extra text.

Below are examples of correct behavior:
    """
# The following text describes an e-commerce product.
# Present as a valid JSON object.
# The key words must start capitalized.
# All values are flat strings, meaning: they don't have any inner lists nor sub-json objects.

    examples = ''
    for idx, example in enumerate(HYPER['prompts']['examples']):
        input_text = example['input']
        output_text = example['output']
        examples += f'\n## Example {idx+1}\n\nINPUT: `{input_text}`\nOUTPUT: `{output_text}`\n'

    system_prompt += examples

    user_prompt = product_description
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Generate answer
    input_ids = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to(model.device)

    output = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=1024,
    )

    print(tokenizer.decode(output[0], skip_special_tokens=False))

# product_description = 'test'
# liquid_desc_to_json(product_description)
