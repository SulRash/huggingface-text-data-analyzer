from transformers import AutoTokenizer
from rich.panel import Panel

from .base_analyzer import BaseAnalyzer
from .advanced_analyzer import AdvancedAnalyzer
from .report_generator import ReportGenerator
from .utils import parse_args, setup_logging

def main():
    args = parse_args()
    console = setup_logging()
    
    try:
        console.rule("[bold blue]Dataset Analysis Tool")
        console.print(f"Starting analysis of dataset: {args.dataset_name}")
        
        tokenizer = None
        if args.tokenizer:
            with console.status("Loading tokenizer..."):
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            console.print(f"Loaded tokenizer: {args.tokenizer}")
        
        console.rule("[bold cyan]Basic Analysis")
        base_analyzer = BaseAnalyzer(
            dataset_name=args.dataset_name,
            split=args.split,
            tokenizer=tokenizer,
            cache_tokenized=args.cache_tokenized,
            console=console,
            chat_field=args.chat_field,
            batch_size=args.batch_size,
            fields=args.fields
        )
        basic_stats = base_analyzer.analyze()
        console.print("[green]Basic analysis complete")
        
        advanced_stats = None
        if args.advanced:
            console.rule("[bold cyan]Advanced Analysis")
            advanced_analyzer = AdvancedAnalyzer(
                dataset_name=args.dataset_name,
                split=args.split,
                use_pos=args.use_pos,
                use_ner=args.use_ner,
                use_lang=args.use_lang,
                use_sentiment=args.use_sentiment,
                console=console
            )
            advanced_stats = advanced_analyzer.analyze_advanced()
            console.print("[green]Advanced analysis complete")
        
        with console.status("Generating reports..."):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            report_generator = ReportGenerator(args.output_dir)
            report_generator.generate_report(basic_stats, advanced_stats)
        
        console.print(f"[green]Analysis complete! Results saved to {args.output_dir}")
        
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
                
    except Exception as e:
        console.print(Panel(
            f"[red]Error during analysis: {str(e)}",
            title="Error",
            border_style="red"
        ))
        raise e
    
    return 0

if __name__ == "__main__":
    exit(main())