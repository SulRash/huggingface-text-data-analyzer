from typing import Optional
from transformers import AutoTokenizer
from rich.panel import Panel
from rich.console import Console

from .src.base_analyzer import BaseAnalyzer
from .src.advanced_analyzer import AdvancedAnalyzer
from .src.report_generator import ReportGenerator
from .src.utils import parse_args, setup_logging, CacheManager

def run_analysis(args, console: Optional[Console] = None) -> int:
    """Main analysis function that can be called programmatically or via CLI"""
    if console is None:
        console = setup_logging()
    
    try:
        # Initial setup and dataset info display
        console.rule("[bold blue]Dataset Analysis Tool")
        if args.subset:
            console.print(f"Starting analysis of dataset: {args.dataset_name} (subset: {args.subset})")
        else:
            console.print(f"Starting analysis of dataset: {args.dataset_name}")

        # Initialize cache manager
        cache_manager = CacheManager(console=console, no_prompt=args.no_prompt)
        
        # Handle cache clearing if requested
        if args.clear_cache:
            cache_manager.clear_cache(
                dataset_name=args.dataset_name,
                subset=args.subset,
                split=args.split
            )
            console.print("[green]Cache cleared successfully")

        basic_stats = None
        advanced_stats = None

        # Basic Analysis Section
        if not args.skip_basic:
            console.rule("[bold cyan]Basic Analysis")
            # Initialize tokenizer if needed
            tokenizer = None
            if args.tokenizer:
                with console.status("Loading tokenizer..."):
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
                console.print(f"Loaded tokenizer: {args.tokenizer}")
            
            base_analyzer = BaseAnalyzer(
                dataset_name=args.dataset_name,
                subset=args.subset,
                split=args.split,
                tokenizer=tokenizer,
                console=console,
                chat_field=args.chat_field,
                batch_size=args.basic_batch_size,
                fields=args.fields,
                cache_manager=cache_manager
            )
            basic_stats = base_analyzer.analyze()
            console.print("[green]Basic analysis complete")

        # Advanced Analysis Section
        if args.advanced:
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
                console=console,
                cache_manager=cache_manager  # Pass the cache manager to the analyzer
            )
            advanced_stats = advanced_analyzer.analyze_advanced()
            console.print("[green]Advanced analysis complete")

        # Report Generation Section
        if basic_stats or advanced_stats:
            with console.status("Generating reports..."):
                args.output_dir.mkdir(parents=True, exist_ok=True)
                report_generator = ReportGenerator(args.output_dir, args.output_format)
                report_generator.generate_report(basic_stats, advanced_stats)
            
            console.print(f"[green]Analysis complete! Results saved to {args.output_dir}")
            
            # Analysis Summary
            console.rule("[bold blue]Analysis Summary")
            summary = []
            
            # Add basic analysis steps to summary
            if basic_stats:
                summary.extend([
                    "✓ Basic text statistics",
                    "✓ Tokenizer analysis" if args.tokenizer else "",
                    f"✓ Chat template applied to {args.chat_field}" if args.chat_field else ""
                ])
            
            # Add advanced analysis steps to summary
            if advanced_stats:
                for field_stats in advanced_stats.field_stats.values():
                    if field_stats.pos_distribution is not None:
                        summary.append("✓ Part-of-speech analysis")
                    if field_stats.entities is not None:
                        summary.append("✓ Named entity recognition")
                    if field_stats.language_dist is not None:
                        summary.append("✓ Language detection")
                    if field_stats.sentiment_scores is not None:
                        summary.append("✓ Sentiment analysis")
                    break  # We only need to check one field's stats
            
            # Remove empty strings and duplicates while maintaining order
            summary = list(dict.fromkeys(item for item in summary if item))
            
            console.print(Panel(
                "\n".join(summary),
                title="Completed Analysis Steps",
                border_style="blue"
            ))
        else:
            console.print("[yellow]No analysis was performed and no cached results were used[/yellow]")
                
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