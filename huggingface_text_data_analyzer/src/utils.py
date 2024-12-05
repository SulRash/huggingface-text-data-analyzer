import pickle
import shutil

from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, List, Optional, Any

from rich.console import Console

class CacheManager:
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.cache_dir = Path.home() / ".cache" / "huggingface-text-data-analyzer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, dataset_name: str, subset: Optional[str], split: str, field_name: str, tokenizer_name: str) -> Path:
        """Generate a unique cache path for the given parameters."""
        safe_name = "".join(c if c.isalnum() else "_" for c in dataset_name)
        safe_field = "".join(c if c.isalnum() else "_" for c in field_name)
        if subset:
            safe_subset = "".join(c if c.isalnum() else "_" for c in subset)
            filename = f"{safe_name}_{safe_subset}_{split}_{safe_field}_{tokenizer_name}_tokens.pkl"
        else:
            filename = f"{safe_name}_{split}_{safe_field}_{tokenizer_name}_tokens.pkl"
        return self.cache_dir / filename
    
    def load_from_cache(self, dataset_name: str, subset: Optional[str], split: str, field_name: str, tokenizer_name: str) -> Optional[Any]:
        """Load data from cache if it exists."""
        cache_path = self.get_cache_path(dataset_name, subset, split, field_name, tokenizer_name)
        if cache_path.exists():
            try:
                self.console.log(f"Loading cached tokens for {field_name}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.console.log(f"[yellow]Warning: Failed to load cache for {field_name}: {str(e)}[/yellow]")
                return None
        return None
    
    def save_to_cache(self, data: Any, dataset_name: str, subset: Optional[str], split: str, field_name: str, tokenizer_name: str) -> None:
        """Save data to cache."""
        cache_path = self.get_cache_path(dataset_name, subset, split, field_name, tokenizer_name)
        try:
            self.console.log(f"Saving tokens to cache for {field_name}")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.console.log(f"[yellow]Warning: Failed to save cache for {field_name}: {str(e)}[/yellow]")
    
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear all cache or cache for a specific dataset."""
        try:
            if dataset_name:
                safe_name = "".join(c if c.isalnum() else "_" for c in dataset_name)
                pattern = f"{safe_name}_*"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                self.console.log(f"Cleared cache for dataset: {dataset_name}")
            else:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True)
                self.console.log("Cleared all cache")
        except Exception as e:
            self.console.log(f"[red]Error clearing cache: {str(e)}[/red]")

# Modified AnalysisArguments and parse_args
class AnalysisArguments(NamedTuple):
    dataset_name: str
    split: str
    output_dir: Path
    tokenizer: str
    advanced: bool
    use_pos: bool
    use_ner: bool
    use_lang: bool
    use_sentiment: bool
    chat_field: str | None
    batch_size: int
    fields: List[str] | None
    clear_cache: bool  # New argument



def parse_args() -> AnalysisArguments:
    parser = ArgumentParser(description="Analyze text dataset from HuggingFace")
    parser.add_argument("dataset_name", help="Name of the dataset on HuggingFace")
    parser.add_argument("--subset", help="Dataset configuration/subset name (if applicable)")
    parser.add_argument("--split", default="train", help="Dataset split to analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"),
                       help="Directory to save analysis results")
    parser.add_argument("--tokenizer", help="HuggingFace tokenizer to use (optional)")
    parser.add_argument("--advanced", action="store_true", help="Run advanced analysis with models")
    parser.add_argument("--use-pos", action="store_true", help="Include POS tagging analysis")
    parser.add_argument("--use-ner", action="store_true", help="Include NER analysis")
    parser.add_argument("--use-lang", action="store_true", help="Include language detection")
    parser.add_argument("--use-sentiment", action="store_true", help="Include sentiment analysis")
    parser.add_argument("--chat-field", type=str, help="Field to apply chat template to")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for tokenization")
    parser.add_argument("--fields", type=str, nargs="+", 
                       help="Specific fields to analyze. If not specified, all text fields will be analyzed")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before analysis")
    
    args = parser.parse_args()
    return AnalysisArguments(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        tokenizer=args.tokenizer,
        advanced=args.advanced,
        use_pos=args.use_pos,
        use_ner=args.use_ner,
        use_lang=args.use_lang,
        use_sentiment=args.use_sentiment,
        chat_field=args.chat_field,
        batch_size=args.batch_size,
        fields=args.fields,
        clear_cache=args.clear_cache
    )