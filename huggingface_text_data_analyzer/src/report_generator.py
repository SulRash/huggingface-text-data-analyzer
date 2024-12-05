from typing import Optional
from pathlib import Path

from .base_analyzer import DatasetStats
from .advanced_analyzer import AdvancedDatasetStats
from .viz_generator import VisualizationGenerator

class ReportGenerator:
    def __init__(self, output_dir: Path, output_format: str = "both"):
        self.output_dir = output_dir
        self.output_format = output_format
        self.markdown_dir = output_dir / "markdown"
        if self.output_format in ["markdown", "both"]:
            self.markdown_dir.mkdir(parents=True, exist_ok=True)
        if self.output_format in ["graphs", "both"]:
            self.viz_generator = VisualizationGenerator(output_dir)
        
    def generate_basic_stats_table(self, stats: DatasetStats) -> str:
        markdown = "# Basic Dataset Statistics\n\n"
        
        headers = ["Field", "Avg Length", "Unique Words", "Junk Frequency"]
        if stats.field_stats[list(stats.field_stats.keys())[0]].avg_token_length is not None:
            headers.append("Avg Token Length")
            
        markdown += "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---" for _ in headers]) + " |\n"
        
        for field, field_stat in stats.field_stats.items():
            row = [
                field,
                f"{field_stat.avg_length:.2f}",
                str(len(field_stat.word_distribution)),
                f"{field_stat.junk_frequency:.2%}"
            ]
            if field_stat.avg_token_length is not None:
                row.append(f"{field_stat.avg_token_length:.2f}")
            markdown += "| " + " | ".join(row) + " |\n"
            
        # Add overall stats
        overall_row = [
            "Overall",
            f"{stats.overall_stats.avg_length:.2f}",
            str(len(stats.overall_stats.word_distribution)),
            f"{stats.overall_stats.junk_frequency:.2%}"
        ]
        if stats.overall_stats.avg_token_length is not None:
            overall_row.append(f"{stats.overall_stats.avg_token_length:.2f}")
        markdown += "| " + " | ".join(overall_row) + " |\n"
        
        return markdown
    
    def generate_advanced_stats_table(self, stats: AdvancedDatasetStats) -> str:
        if not stats.field_stats:
            return "# Advanced Dataset Statistics\n\nNo advanced statistics available.\n"
            
        markdown = "# Advanced Dataset Statistics\n\n"
        first_field_stats = next(iter(stats.field_stats.values()))
        
        # POS Distribution
        if first_field_stats.pos_distribution:
            markdown += "## Part of Speech Distribution\n\n"
            pos_headers = ["Field"] + list(first_field_stats.pos_distribution.keys())
            markdown += "| " + " | ".join(pos_headers) + " |\n"
            markdown += "| " + " | ".join(["---" for _ in pos_headers]) + " |\n"
            
            for field, field_stat in stats.field_stats.items():
                row = [field] + [f"{field_stat.pos_distribution.get(pos, 0):.2%}" for pos in pos_headers[1:]]
                markdown += "| " + " | ".join(row) + " |\n"
            
        # Entity Distribution
        if first_field_stats.entities:
            markdown += "\n## Named Entity Distribution\n\n"
            entity_headers = ["Field"] + list(first_field_stats.entities.keys())
            markdown += "| " + " | ".join(entity_headers) + " |\n"
            markdown += "| " + " | ".join(["---" for _ in entity_headers]) + " |\n"
            
            for field, field_stat in stats.field_stats.items():
                row = [field] + [str(field_stat.entities.get(ent, 0)) for ent in entity_headers[1:]]
                markdown += "| " + " | ".join(row) + " |\n"
            
        # Language Distribution
        if first_field_stats.language_dist:
            markdown += "\n## Language Distribution\n\n"
            lang_headers = ["Field"] + list(first_field_stats.language_dist.keys())
            markdown += "| " + " | ".join(lang_headers) + " |\n"
            markdown += "| " + " | ".join(["---" for _ in lang_headers]) + " |\n"
            
            for field, field_stat in stats.field_stats.items():
                row = [field] + [f"{field_stat.language_dist.get(lang, 0):.2%}" for lang in lang_headers[1:]]
                markdown += "| " + " | ".join(row) + " |\n"
        
        # If no statistics were added
        if markdown == "# Advanced Dataset Statistics\n\n":
            markdown += "No advanced statistics available.\n"
            
        return markdown
    
    def generate_word_distribution_section(self, stats: DatasetStats) -> str:
        markdown = "# Word Distribution Analysis\n\n"
        
        for field, field_stat in stats.field_stats.items():
            markdown += f"## {field}\n\n"
            top_words = sorted(field_stat.word_distribution.items(), key=lambda x: x[1], reverse=True)[:20]
            
            markdown += "| Word | Frequency |\n"
            markdown += "| --- | --- |\n"
            for word, freq in top_words:
                markdown += f"| {word} | {freq} |\n"
            markdown += "\n"
            
        return markdown
    
    def generate_report(
        self,
        basic_stats: DatasetStats,
        advanced_stats: Optional[AdvancedDatasetStats] = None
    ) -> None:
        if self.output_format in ["markdown", "both"]:
            # Generate markdown reports
            basic_stats_md = self.generate_basic_stats_table(basic_stats)
            word_dist_md = self.generate_word_distribution_section(basic_stats)
            
            with open(self.markdown_dir / "basic_stats.md", "w") as f:
                f.write(basic_stats_md)
                
            with open(self.markdown_dir / "word_distribution.md", "w") as f:
                f.write(word_dist_md)
                
            if advanced_stats:
                advanced_stats_md = self.generate_advanced_stats_table(advanced_stats)
                with open(self.markdown_dir / "advanced_stats.md", "w") as f:
                    f.write(advanced_stats_md)

        if self.output_format in ["graphs", "both"]:
            # Generate visualization plots
            self.viz_generator.generate_basic_stats_plots(basic_stats)
            if advanced_stats:
                self.viz_generator.generate_advanced_stats_plots(advanced_stats)