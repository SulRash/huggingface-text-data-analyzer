# Huggingface Text Data Analyzer

![Repository Image](assets/hftxtdata.png)

A comprehensive tool for analyzing text datasets from HuggingFace's datasets library. This tool provides both basic text statistics and advanced NLP analysis capabilities with optimized performance for large datasets.

## Analysis Types

The tool supports two types of analysis that can be run independently or together:

### Basic Analysis (Default)
- Average text length per field
- Word distribution analysis
- Junk text detection (HTML tags, special characters)
- Tokenizer-based analysis (when tokenizer is specified)
- Token length statistics with batch processing

### Advanced Analysis (Optional)
- Part-of-Speech (POS) tagging
- Named Entity Recognition (NER)
- Language detection using XLM-RoBERTa
- Sentiment analysis using distilbert-sst-2-english

You can control which analyses to run using these flags:
- `--skip-basic`: Skip basic analysis (must be used with `--advanced`)
- `--advanced`: Enable advanced analysis
- `--use-pos`: Enable POS tagging
- `--use-ner`: Enable NER
- `--use-lang`: Enable language detection
- `--use-sentiment`: Enable sentiment analysis

## Installation

### From PyPI
```bash
pip install huggingface-text-data-analyzer
```

### From Source
1. Clone the repository:
```bash
git clone https://github.com/yourusername/huggingface-text-data-analyzer.git
cd huggingface-text-data-analyzer
```

2. Install in development mode:
```bash
pip install -e .
```

3. Install spaCy's English model (if using advanced analysis):
```bash
python -m spacy download en_core_web_sm
```

## Usage

The tool is available as a command-line application after installation. You can run it using the `analyze-dataset` command:

Basic usage:
```bash
analyze-dataset "dataset_name" --split "train" --output-dir "results"
```

With tokenizer analysis:
```bash
analyze-dataset "dataset_name" --tokenizer "bert-base-uncased"
```

Analyze specific fields with chat template:
```bash
analyze-dataset "dataset_name" \
    --fields instruction response \
    --chat-field response \
    --tokenizer "meta-llama/Llama-2-7b-chat-hf"
```

Run only advanced analysis:
```bash
analyze-dataset "dataset_name" --skip-basic --advanced --use-pos --use-lang
```

Run both analyses:
```bash
analyze-dataset "dataset_name" --advanced --use-sentiment
```

Run basic analysis only (default):
```bash
analyze-dataset "dataset_name"
```

Full analysis with all features:
```bash
analyze-dataset "dataset_name" \
    --advanced \
    --use-pos \
    --use-ner \
    --use-lang \
    --use-sentiment \
    --tokenizer "bert-base-uncased" \
    --output-dir "results" \
    --fields instruction response \
    --batch-size 64
```

### Command Line Arguments

- `dataset_name`: Name of the dataset on HuggingFace (required)
- `--split`: Dataset split to analyze (default: "train")
- `--output-dir`: Directory to save analysis results (default: "analysis_results")
- `--tokenizer`: HuggingFace tokenizer to use (optional)
- `--cache-tokenized`: Cache tokenized texts (default: True)
- `--batch-size`: Batch size for tokenization (default: 32)
- `--fields`: Specific fields to analyze (optional, analyzes all text fields if not specified)
- `--chat-field`: Field to apply chat template to (optional)
- `--advanced`: Run advanced analysis with models
- `--use-pos`: Include POS tagging analysis
- `--use-ner`: Include NER analysis
- `--use-lang`: Include language detection
- `--use-sentiment`: Include sentiment analysis

### Python API

You can also use the tool programmatically in your Python code:

```python
from huggingface_text_data_analyzer import BaseAnalyzer, AdvancedAnalyzer

# Basic analysis
analyzer = BaseAnalyzer(
    dataset_name="your_dataset",
    split="train",
    tokenizer="bert-base-uncased"
)
results = analyzer.analyze()

# Advanced analysis
advanced_analyzer = AdvancedAnalyzer(
    dataset_name="your_dataset",
    split="train",
    use_pos=True,
    use_ner=True
)
advanced_results = advanced_analyzer.analyze_advanced()
```
## Project Structure

```
huggingface_text_data_analyzer/
├── src/
│   ├── base_analyzer.py      # Basic text analysis functionality
│   ├── advanced_analyzer.py  # Model-based advanced analysis
│   ├── report_generator.py   # Markdown report generation
│   └── utils.py             # Utility functions and argument parsing
├── cli.py                   # Command-line interface
└── __init__.py             # Package initialization
```

## Output

The tool generates markdown reports in the specified output directory:
- `basic_stats.md`: Contains basic text statistics
- `word_distribution.md`: Word frequency analysis
- `advanced_stats.md`: Results from model-based analysis (if enabled)

## Performance and Accuracy Considerations

### Batch Sizes and Memory Usage

The tool uses two different batch sizes for processing:

1. **Basic Batch Size** (`--basic-batch-size`, default: 1):
   - Used for tokenization and basic text analysis
   - Higher values improve processing speed but may affect token count accuracy
   - Token counting in larger batches can be affected by padding, truncation, and memory constraints
   - If exact token counts are crucial, use smaller batch sizes (8-16)

2. **Advanced Batch Size** (`--advanced-batch-size`, default: 16):
   - Used for transformer models (language detection, sentiment analysis)
   - Adjust based on your GPU memory
   - Larger batches improve processing speed but require more GPU memory
   - CPU-only users might want to use smaller batches (4-8)

### GPU Support

The tool automatically detects and uses available CUDA GPUs for:
- Language detection model
- Sentiment analysis model
- Tokenizer operations

SpaCy operations (POS tagging, NER) remain CPU-bound for better compatibility.

### Examples

For exact token counting:
```bash
analyze-dataset "dataset_name" --basic-batch-size 8
```

For faster processing with GPU:
```bash
analyze-dataset "dataset_name" --advanced-batch-size 32 --basic-batch-size 64
```

For memory-constrained environments:
```bash
analyze-dataset "dataset_name" --advanced-batch-size 4 --basic-batch-size 16
```

## Performance Features

- Batch processing for tokenization
- Progress bars for long-running operations
- Tokenizer parallelism enabled
- Caching support for tokenized texts
- Memory-efficient processing of large datasets
- Optimized batch sizes for better performance

## Requirements

- Python 3.8+
- transformers
- datasets
- spacy
- rich
- torch
- pandas
- numpy
- scikit-learn (for advanced features)
- tqdm

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0