import pickle
import shutil

from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, List, Optional, Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
from rich.prompt import Confirm

class AnalysisResults(NamedTuple):
    """Container for analysis results with metadata"""
    dataset_name: str
    subset: Optional[str]
    split: str
    fields: Optional[List[str]]
    tokenizer: Optional[str]
    timestamp: float
    basic_stats: Optional[Any] = None
    pos_stats: Optional[Any] = None
    ner_stats: Optional[Any] = None
    lang_stats: Optional[Any] = None
    sentiment_stats: Optional[Any] = None

class CacheManager:
    def __init__(self, console: Optional[Console] = None, no_prompt: bool = False):
        self.console = console or Console()
        self.no_prompt = no_prompt
        self.cache_dir = Path.home() / ".cache" / "huggingface-text-data-analyzer"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(
        self,
        dataset_name: str,
        subset: Optional[str],
        split: str,
        field: str,
        analysis_type: str
    ) -> Path:
        """Generate a unique cache path based on all parameters."""
        # Create safe versions of parameters
        safe_name = "".join(c if c.isalnum() else "_" for c in dataset_name)
        safe_field = "".join(c if c.isalnum() else "_" for c in field)
        safe_type = "".join(c if c.isalnum() else "_" for c in analysis_type)
        
        # Build cache path components
        components = [safe_name]
        if subset:
            safe_subset = "".join(c if c.isalnum() else "_" for c in subset)
            components.append(safe_subset)
        components.extend([split, safe_field, safe_type])
        
        # Create cache subdirectories
        cache_subdir = self.cache_dir / safe_name
        if subset:
            cache_subdir = cache_subdir / safe_subset
        cache_subdir = cache_subdir / split / safe_field
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        return cache_subdir / f"{safe_type}_results.pkl"

    def should_use_cache(
        self,
        dataset_name: str,
        field: str,
        analysis_type: str
    ) -> bool:
        """Determine if cached results should be used."""
        if self.no_prompt:
            return True
        return Confirm.ask(
            f"Found existing {analysis_type} analysis results for {dataset_name}/{field}. Use cached results?",
            default=True
        )

    def load_cached_results(
        self,
        dataset_name: str,
        subset: Optional[str],
        split: str,
        field: str,
        analysis_type: str,
        force: bool = False
    ) -> Optional[Any]:
        """Load cached analysis results if they exist."""
        cache_path = self.get_cache_path(dataset_name, subset, split, field, analysis_type)
        
        if cache_path.exists() and not force:
            try:
                with open(cache_path, 'rb') as f:
                    results = pickle.load(f)
                if self.should_use_cache(dataset_name, field, analysis_type):
                    self.console.log(f"Loading cached {analysis_type} analysis results for {dataset_name}/{field}")
                    return results
            except Exception as e:
                self.console.log(f"[yellow]Warning: Failed to load {analysis_type} results for {field}: {str(e)}[/yellow]")
        return None

    def save_results(
        self,
        data: Any,
        dataset_name: str,
        subset: Optional[str],
        split: str,
        field: str,
        analysis_type: str,
        force: bool = False
    ) -> bool:
        """Save analysis results to cache."""
        cache_path = self.get_cache_path(dataset_name, subset, split, field, analysis_type)
        
        if cache_path.exists() and not force and not self.no_prompt:
            should_overwrite = Confirm.ask(
                f"{analysis_type.title()} analysis results already exist for {dataset_name}/{field}. Overwrite?",
                default=False
            )
            if not should_overwrite:
                return False

        try:
            self.console.log(f"Saving {analysis_type} analysis results for {dataset_name}/{field}")
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            self.console.log(f"[yellow]Warning: Failed to save {analysis_type} results for {field}: {str(e)}[/yellow]")
            return False

    def clear_cache(
        self,
        dataset_name: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        field: Optional[str] = None,
        analysis_type: Optional[str] = None
    ) -> None:
        """Clear cache with granular control."""
        try:
            if dataset_name:
                base_path = self.cache_dir / dataset_name
                if not subset:
                    if base_path.exists():
                        shutil.rmtree(base_path)
                        self.console.log(f"Cleared all cache for dataset: {dataset_name}")
                    return
                
                base_path = base_path / subset if subset else base_path
                if not split:
                    if base_path.exists():
                        shutil.rmtree(base_path)
                        self.console.log(f"Cleared cache for dataset: {dataset_name}/{subset}")
                    return
                
                base_path = base_path / split if split else base_path
                if not field:
                    if base_path.exists():
                        shutil.rmtree(base_path)
                        self.console.log(f"Cleared cache for: {dataset_name}/{subset}/{split}")
                    return
                
                base_path = base_path / field if field else base_path
                if not analysis_type:
                    if base_path.exists():
                        shutil.rmtree(base_path)
                        self.console.log(f"Cleared cache for: {dataset_name}/{subset}/{split}/{field}")
                    return
                
                cache_path = self.get_cache_path(dataset_name, subset, split, field, analysis_type)
                if cache_path.exists():
                    cache_path.unlink()
                    self.console.log(f"Cleared {analysis_type} cache for: {dataset_name}/{subset}/{split}/{field}")
            else:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True)
                self.console.log("Cleared all cache")
                
        except Exception as e:
            self.console.log(f"[red]Error clearing cache: {str(e)}[/red]")

class AnalysisArguments(NamedTuple):
    dataset_name: str
    subset: Optional[str]
    split: str
    output_dir: Path
    tokenizer: str
    skip_basic: bool
    advanced: bool
    use_pos: bool
    use_ner: bool
    use_lang: bool
    use_sentiment: bool
    chat_field: str | None
    basic_batch_size: int
    advanced_batch_size: int
    fields: List[str] | None
    clear_cache: bool
    no_prompt: bool
    output_format: str

def create_progress() -> Progress:
    """Create a consistent progress bar for the application."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=Console()
    )

def setup_logging() -> Console:
    """Create and return a Console instance for logging."""
    return Console()

def parse_args() -> AnalysisArguments:
    parser = ArgumentParser(description="Analyze text dataset from HuggingFace")
    parser.add_argument("dataset_name", help="Name of the dataset on HuggingFace")
    parser.add_argument("--subset", help="Dataset configuration/subset name (if applicable)")
    parser.add_argument("--split", default="train", help="Dataset split to analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"),
                       help="Directory to save analysis results")
    parser.add_argument("--tokenizer", help="HuggingFace tokenizer to use (optional)")
    parser.add_argument("--skip-basic", action="store_true", 
                       help="Skip basic analysis (word counts, distributions, etc.)")
    parser.add_argument("--advanced", action="store_true", help="Run advanced analysis with models")
    parser.add_argument("--use-pos", action="store_true", help="Include POS tagging analysis")
    parser.add_argument("--use-ner", action="store_true", help="Include NER analysis")
    parser.add_argument("--use-lang", action="store_true", help="Include language detection")
    parser.add_argument("--use-sentiment", action="store_true", help="Include sentiment analysis")
    parser.add_argument("--chat-field", type=str, help="Field to apply chat template to")
    parser.add_argument("--basic-batch-size", type=int, default=1, 
                       help="Batch size for basic tokenization and analysis (higher values may affect token count accuracy)")
    parser.add_argument("--advanced-batch-size", type=int, default=16,
                       help="Batch size for advanced analysis models (adjust based on GPU memory)")
    parser.add_argument("--fields", type=str, nargs="+", 
                       help="Specific fields to analyze. If not specified, all text fields will be analyzed")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before analysis")
    parser.add_argument(
        "--output-format",
        choices=["markdown", "graphs", "both"],
        default="both",
        help="Output format for analysis results (default: both)"
    )
    
    parser.add_argument("--no-prompt", action="store_true",
                       help="Always use cached results without prompting")
    
    args = parser.parse_args()

    if args.skip_basic and not args.advanced:
        parser.error("Cannot skip basic analysis without enabling advanced analysis (--advanced)")
    
    return AnalysisArguments(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        output_dir=args.output_dir,
        tokenizer=args.tokenizer,
        skip_basic=args.skip_basic,
        advanced=args.advanced,
        use_pos=args.use_pos,
        use_ner=args.use_ner,
        use_lang=args.use_lang,
        use_sentiment=args.use_sentiment,
        chat_field=args.chat_field,
        basic_batch_size=args.basic_batch_size,
        advanced_batch_size=args.advanced_batch_size,
        fields=args.fields,
        clear_cache=args.clear_cache,
        no_prompt=args.no_prompt,
        output_format=args.output_format
    )