from pathlib import Path
from transformers import AutoTokenizer
from rich.panel import Panel
from rich.console import Console

from .src.base_analyzer import BaseAnalyzer
from .src.advanced_analyzer import AdvancedAnalyzer
from .src.report_generator import ReportGenerator
from .src.utils import parse_args, setup_logging, CacheManager, AnalysisResults

from time import time

def run_analysis(args, console: Console = None):
    """Main analysis function that can be called programmatically or via CLI"""
    if console is None:
        console = setup_logging()
    
    try:
        console.rule("[bold blue]Dataset Analysis Tool")
        if args.subset:
            console.print(f"Starting analysis of dataset: {args.dataset_name} (subset: {args.subset})")
        else:
            console.print(f"Starting analysis of dataset: {args.dataset_name}")

        # Initialize cache manager
        cache_manager = CacheManager(console=console)
        
        # Try to load existing results
        cached_results = cache_manager.load_cached_results(
            args.dataset_name,
            args.subset,
            args.split,
            force=args.clear_cache
        )
        
        # Initialize results with metadata
        current_results = AnalysisResults(
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=args.split,
            fields=args.fields,
            tokenizer=args.tokenizer,
            timestamp=time(),
            basic_stats=cached_results.basic_stats if cached_results else None,
            advanced_stats=cached_results.advanced_stats if cached_results else None
        )

        # Clear token cache if requested
        if args.clear_cache:
            cache_manager.clear_cache(args.dataset_name)
            console.print("[green]Cache cleared successfully")

        tokenizer = None
        if args.tokenizer:
            with console.status("Loading tokenizer..."):
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            console.print(f"Loaded tokenizer: {args.tokenizer}")

        # Run basic analysis if needed
        if not args.skip_basic and (not cached_results or not cached_results.basic_stats):
            console.rule("[bold cyan]Basic Analysis")
            base_analyzer = BaseAnalyzer(
                dataset_name=args.dataset_name,
                subset=args.subset,
                split=args.split,
                tokenizer=tokenizer,
                console=console,
                chat_field=args.chat_field,
                batch_size=args.basic_batch_size,
                fields=args.fields
            )
            current_results = current_results._replace(
                basic_stats=base_analyzer.analyze()
            )
            console.print("[green]Basic analysis complete")
            # Save intermediate results
            cache_manager.save_results(current_results, force=True)
        elif not args.skip_basic and cached_results and cached_results.basic_stats:
            console.print("[cyan]Using cached basic analysis results")

        # Run advanced analysis if needed
        if args.advanced and (not cached_results or not cached_results.advanced_stats):
            console.rule("[bold cyan]Advanced Analysis")
            advanced_analyzer = AdvancedAnalyzer(
                dataset_name=args.dataset_name,
                subset=args.subset,
                split=args.split,
                fields=args.fields,
                use_pos=args.use_pos,
                use_ner=args.use_ner,
                use_lang=args.use_lang,
                use_sentiment=args.use_sentiment,
                batch_size=args.advanced_batch_size,
                console=console
            )
            current_results = current_results._replace(
                advanced_stats=advanced_analyzer.analyze_advanced()
            )
            console.print("[green]Advanced analysis complete")
            # Save final results
            cache_manager.save_results(current_results, force=True)
        elif args.advanced and cached_results and cached_results.advanced_stats:
            console.print("[cyan]Using cached advanced analysis results")

        # Generate reports
        if current_results.basic_stats or current_results.advanced_stats:
            with console.status("Generating reports..."):
                args.output_dir.mkdir(parents=True, exist_ok=True)
                report_generator = ReportGenerator(args.output_dir, args.output_format)
                report_generator.generate_report(
                    current_results.basic_stats, 
                    current_results.advanced_stats
                )
            
            console.print(f"[green]Analysis complete! Results saved to {args.output_dir}")
            
            # Print summary of analyses performed
            console.rule("[bold blue]Analysis Summary")
            summary = []
            
            if current_results.basic_stats:
                summary.extend([
                    "✓ Basic text statistics",
                    "✓ Tokenizer analysis" if tokenizer else "",
                    f"✓ Chat template applied to {args.chat_field}" if args.chat_field else ""
                ])
            
            if current_results.advanced_stats:
                summary.extend([
                    "✓ Part-of-speech analysis" if args.use_pos else "",
                    "✓ Named entity recognition" if args.use_ner else "",
                    "✓ Language detection" if args.use_lang else "",
                    "✓ Sentiment analysis" if args.use_sentiment else ""
                ])
            
            summary = [item for item in summary if item]  # Remove empty strings
            
            console.print(Panel(
                "\n".join(summary),
                title="Completed Analysis Steps",
                border_style="blue"
            ))
        else:
            console.print("[yellow]No analysis was performed - using cached results[/yellow]")
                
    except Exception as e:
        console.print(Panel(
            f"[red]Error during analysis: {str(e)}",
            title="Error",
            border_style="red"
        ))
        raise e
    
    return 0

def main():
    """CLI entry point"""
    args = parse_args()
    return run_analysis(args)

if __name__ == "__main__":
    exit(main())