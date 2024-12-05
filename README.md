# Dataset Analysis Tool

A comprehensive tool for analyzing text datasets from HuggingFace's datasets library. This tool provides both basic text statistics and advanced NLP analysis capabilities with optimized performance for large datasets.

## Features

### Basic Analysis
- Average text length per field
- Word distribution analysis
- Junk text detection (HTML tags, special characters)
- Tokenizer-based analysis (optional)
- Token length statistics with batch processing
- Word distribution visualization
- Chat template support for conversational data
- Field-specific analysis

### Advanced Analysis (Optional)
- Part-of-Speech (POS) tagging
- Named Entity Recognition (NER)
- Language detection
- Sentiment analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dataset-analysis-tool.git
cd dataset-analysis-tool
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install spaCy's English model (if using advanced analysis):
```bash
python -m spacy download en_core_web_sm
```

## Usage

Basic usage:
```bash
python main.py "dataset_name" --split "train" --output-dir "results"
```

With tokenizer analysis:
```bash
python main.py "dataset_name" --tokenizer "bert-base-uncased"
```

Analyze specific fields with chat template:
```bash
python main.py "dataset_name" \
    --fields instruction response \
    --chat-field response \
    --tokenizer "meta-llama/Llama-2-7b-chat-hf"
```

Full analysis with all features:
```bash
python main.py "dataset_name" \
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

## Project Structure

```
dataset-analysis-tool/
├── src/
│   ├── base_analyzer.py      # Basic text analysis functionality
│   ├── advanced_analyzer.py  # Model-based advanced analysis
│   ├── report_generator.py   # Markdown report generation
│   └── utils.py             # Utility functions and argument parsing
├── main.py                  # Main script
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Output

The tool generates markdown reports in the specified output directory:
- `basic_stats.md`: Contains basic text statistics
- `word_distribution.md`: Word frequency analysis
- `advanced_stats.md`: Results from model-based analysis (if enabled)

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
- fasttext
- rich
- torch (for transformers)
- pandas
- numpy
- scikit-learn (for advanced features)
- tqdm

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0