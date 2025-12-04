""" Module for defining hyperparameters for various components of the system. """


def get_hyperparameters() -> dict:
    """ Returns a dictionary of hyperparameters for various components of the system."""
    hyperparams = {
        'data_preprocess': {},
        'evaluation': {},
        'load_db': {
            'verbose': True,  # Whether to print detailed logs during data loading
            'simple_columns': False,  # wheter to remove columns that aren't text and json_answer
            'proportions': {  # Proportions for splitting the dataset
                'test': 0.1,
                'validation': 0.1,
                'train': 0.8,
            },
            'seed': 42,  # Used for reproducibility when shuffling data
            'balanced_output': 'Datasets/jsonllm/balanced_jsonllm.parquet'
        },
        'prompts': {
            'system': {  # System-level instructions for the model
                'description': """The following text describes an e-commerce product.""",
                'json': """Present as a valid JSON object.""",
                'capitalization': """The key words must start capitalized.""",
                'structure': """All values are flat strings, meaning: they don't have any inner lists nor sub-json objects.""",
                'instruction': """
                Extract structured product information from the text.
                Identify product name, brand, model, category, color, size, material, and any key attributes.
                Use the exact text for extractions — do not paraphrase.
                Return relevant attributes that describe each product clearly.
                """
            },
            'examples': [  # Few-shot examples for model training or evaluation
                {
                    'input': '''New Electric Flameless Torch Battery Windproof Lighter BBQ Picnic Camping No Gas Fuel Required Fire Starter''',
                    'output': '''{"Disposable": "No", "Automatic Alarming or Not (Kettle)": "No", "Category": "Lighter", "With Fuel or Not": "No", "With Wooden Handle or Not": "No"}'''
                },
                {
                    'input': '''Miyouj Leaves Print One Piece Swimsuit Female Off Shoulder Swimwear Women Bathing Suits Brazilian Bikini May Beach XL Monokini''',
                    'output': '''{"Sport Type": "Swim", "Model Number": "Monokini", "Gender": "Women", "Pattern Type": "Print", "one-piece suit": "Swimsuit Female", "Bathers": "Off Shoulder Swimwear", "Female Swimsuits": "Swimwear Women", "Monokini": "Women Bathing Suits", "Bathing Suit Women": "Monokini", "Women Swimwear": "Brazilian Bikini", "Swimwear 2018": "Bikini May Beach", "swimsuit Bathing Suit": "Brazilian Bikini"}'''
                },
                {
                    'input': '''Outdoor Travel Cooker Stoves Ultra-Light Spirit Alcohol Stove Camping Cooking Furnace''',
                    'output': '''{"Type": "Alcohol Stove"}'''
                }
            ],
            'examples_langextract': [
                {
                    'id': 1,
                    'text': 'Camiseta PoloTech masculina de algodão, cor azul marinho, disponível nos tamanhos M, G e GG.',
                    'class': 'product',
                    'ext_text': 'Camiseta PoloTech masculina',
                    'attributes': {
                        'brand': 'PoloTech',
                        'category': 'camiseta',
                        'material': 'algodão',
                        'color': 'azul marinho',
                        'sizes': ['M', 'G', 'GG']
                    }
                },
                {
                    'id': 2,
                    'text': 'Tênis esportivo Nike Air Zoom branco, ideal para corrida.',
                    'class': 'product',
                    'ext_text': 'Tênis esportivo Nike Air Zoom branco',
                    'attributes': {
                        'brand': 'Nike',
                        'category': 'tênis esportivo',
                        'color': 'branco',
                        'intended_use': 'corrida'
                    }
                },
            ]
        },
        'model': {
            'liquid': {  # Hyperparameters for the LiquidAI model
                # 'model_name': 'LiquidAI/LFM2-350M-Extract',
                'model_name': 'LiquidAI/LFM2-1.2B-Extract',
                'max_new_tokens': 160,
                'add_generation_prompt': True,
                'tokenize': True,
                'return_dict': True,
                'return_tensors': 'pt',
            }
        },
        'mave': {
            'verbose': True,
            'starting_point': 0,  # Set to 0 to process all entries
            'workers': 10,  # Number of threads for parallel processing
            'output_path': './Outputs/mave_extractions.jsonl',
            'model_id': 'gemma2:2b',
            'model_url': 'http://localhost:11434',

        },
        'jsonllm': {
            'verbose': True,
            'starting_point': 0,
            'workers': 10,
            'output_path': './Generations/jsonllm.jsonl',
            'model_id': 'gemma2:2b',
            'model_url': 'http://localhost:11434',
        },
        'evaluation': {
            'debug': True,  # Whether to print debug information during evaluation
            # Model for generating embeddings
            'embedder_model': 'mixedbread-ai/mxbai-embed-large-v1',
            # Whether to recompute embeddings even if they exist
            'force_recompute_embeddings': False,
        }
    }
    return hyperparams
