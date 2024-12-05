from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

class AnalysisArguments(NamedTuple):
    dataset_name: str
    split: str
    output_dir: Path
    tokenizer: str
    cache_tokenized: bool
    advanced: bool
    use_pos: bool
    use_ner: bool
    use_lang: bool
    use_sentiment: bool
    chat_field: str | None
    batch_size: int
    fields: List[str] | None  # Added fields parameter

def setup_logging() -> Console:
    return Console()

def create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=setup_logging()
    )

def parse_args() -> AnalysisArguments:
    parser = ArgumentParser(description="Analyze text dataset from HuggingFace")
    parser.add_argument("dataset_name", help="Name of the dataset on HuggingFace")
    parser.add_argument("--split", default="train", help="Dataset split to analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"),
                       help="Directory to save analysis results")
    parser.add_argument("--tokenizer", help="HuggingFace tokenizer to use (optional)")
    parser.add_argument("--cache-tokenized", action="store_true", default=True,
                       help="Cache tokenized texts")
    parser.add_argument("--advanced", action="store_true", help="Run advanced analysis with models")
    parser.add_argument("--use-pos", action="store_true", help="Include POS tagging analysis")
    parser.add_argument("--use-ner", action="store_true", help="Include NER analysis")
    parser.add_argument("--use-lang", action="store_true", help="Include language detection")
    parser.add_argument("--use-sentiment", action="store_true", help="Include sentiment analysis")
    parser.add_argument("--chat-field", type=str, help="Field to apply chat template to")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for tokenization")
    parser.add_argument("--fields", type=str, nargs="+", 
                       help="Specific fields to analyze. If not specified, all text fields will be analyzed")
    
    args = parser.parse_args()
    return AnalysisArguments(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        tokenizer=args.tokenizer,
        cache_tokenized=args.cache_tokenized,
        advanced=args.advanced,
        use_pos=args.use_pos,
        use_ner=args.use_ner,
        use_lang=args.use_lang,
        use_sentiment=args.use_sentiment,
        chat_field=args.chat_field,
        batch_size=args.batch_size,
        fields=args.fields
    )