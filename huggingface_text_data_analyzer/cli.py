from pathlib import Path
from transformers import AutoTokenizer
from rich.panel import Panel
from rich.console import Console

from .src.base_analyzer import BaseAnalyzer
from .src.advanced_analyzer import AdvancedAnalyzer
from .src.report_generator import ReportGenerator
from .src.utils import parse_args, setup_logging, CacheManager

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

        # Handle cache clearing
        cache_manager = CacheManager(console=console)
        if args.clear_cache:
            cache_manager.clear_cache(args.dataset_name)
            console.print("[green]Cache cleared successfully")

        tokenizer = None
        if args.tokenizer:
            with console.status("Loading tokenizer..."):
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            console.print(f"Loaded tokenizer: {args.tokenizer}")

        console.rule("[bold cyan]Basic Analysis")
        base_analyzer = BaseAnalyzer(
            dataset_name=args.dataset_name,
            subset=args.subset,
            split=args.split,
            tokenizer=tokenizer,
            console=console,
            chat_field=args.chat_field,
            batch_size=args.basic_batch_size,  # Updated to use basic_batch_size
            fields=args.fields
        )
        basic_stats = base_analyzer.analyze()
        console.print("[green]Basic analysis complete")

        advanced_stats = None
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
                batch_size=args.advanced_batch_size,  # Updated to use advanced_batch_size
                console=console
            )
            advanced_stats = advanced_analyzer.analyze_advanced()
            console.print("[green]Advanced analysis complete")
            
        with console.status("Generating reports..."):
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report_generator = ReportGenerator(output_dir)
            report_generator.generate_report(basic_stats, advanced_stats)
        
        console.print(f"[green]Analysis complete! Results saved to {output_dir}")

        # Print summary of analyses performed
        console.rule("[bold blue]Analysis Summary")
        summary = [
            "✓ Basic text statistics",
            "✓ Tokenizer analysis" if tokenizer else "",
            f"✓ Chat template applied to {args.chat_field}" if args.chat_field else "",
            "✓ Part-of-speech analysis" if args.advanced and args.use_pos else "",
            "✓ Named entity recognition" if args.advanced and args.use_ner else "",
            "✓ Language detection" if args.advanced and args.use_lang else "",
            "✓ Sentiment analysis" if args.advanced and args.use_sentiment else ""
        ]
        summary = [item for item in summary if item]  # Remove empty strings
        console.print(Panel(
            "\n".join(summary),
            title="Completed Analysis Steps",
            border_style="blue"
        ))

        return 0
        
    except Exception as e:
        console.print(Panel(
            f"[red]Error during analysis: {str(e)}",
            title="Error",
            border_style="red"
        ))
        raise e

def main():
    """CLI entry point"""
    args = parse_args()
    return run_analysis(args)

if __name__ == "__main__":
    exit(main())