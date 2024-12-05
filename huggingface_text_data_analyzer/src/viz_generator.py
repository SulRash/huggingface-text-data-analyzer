from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_analyzer import DatasetStats
from .advanced_analyzer import AdvancedDatasetStats

class VisualizationGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Set style for all plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 6]
        
    def generate_basic_stats_plots(self, stats: DatasetStats) -> None:
        """Generate visualizations for basic statistics."""
        # Text Length Distribution
        plt.figure()
        stats_data = []
        for field, field_stat in stats.field_stats.items():
            stats_data.append({
                'Field': field,
                'Average Length': field_stat.avg_length,
                'Token Length': field_stat.avg_token_length or 0,
                'Junk Frequency': field_stat.junk_frequency * 100
            })
        
        df = pd.DataFrame(stats_data)
        
        # Create a grouped bar plot
        ax = plt.figure(figsize=(10, 6))
        x = np.arange(len(df))
        width = 0.25
        
        plt.bar(x - width, df['Average Length'], width, label='Avg Length', color='skyblue')
        plt.bar(x, df['Token Length'], width, label='Token Length', color='lightgreen')
        plt.bar(x + width, df['Junk Frequency'], width, label='Junk %', color='salmon')
        
        plt.xlabel('Fields')
        plt.ylabel('Values')
        plt.title('Basic Text Statistics by Field')
        plt.xticks(x, df['Field'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'basic_stats.png')
        plt.close()
        
        # Word Distribution
        for field, field_stat in stats.field_stats.items():
            plt.figure(figsize=(12, 6))
            # Create DataFrame and sort by frequency
            word_freq = pd.DataFrame(
                list(field_stat.word_distribution.items()), 
                columns=['Word', 'Frequency']
            ).sort_values('Frequency', ascending=False).head(20)
            
            # Create sorted bar plot
            sns.barplot(
                data=word_freq,
                x='Word',
                y='Frequency',
                color='skyblue',
                order=word_freq['Word']  # Ensure bars follow sorted order
            )
            plt.title(f'Top 20 Words Distribution - {field}')
            plt.xticks(rotation=45, ha='right')  # Improved label readability
            plt.tight_layout()
            plt.savefig(self.output_dir / f'word_dist_{field}.png')
            plt.close()

    def generate_advanced_stats_plots(self, stats: AdvancedDatasetStats) -> None:
        """Generate visualizations for advanced statistics."""
        if not stats.field_stats:
            return
            
        for field, field_stat in stats.field_stats.items():
            # POS Distribution
            if field_stat.pos_distribution:
                plt.figure(figsize=(14, 6))
                pos_df = pd.DataFrame(
                    list(field_stat.pos_distribution.items()),
                    columns=['POS', 'Frequency']
                )
                sns.barplot(data=pos_df, x='POS', y='Frequency', color='lightgreen')
                plt.title(f'Part of Speech Distribution - {field}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'pos_dist_{field}.png')
                plt.close()

            # Named Entity Distribution
            if field_stat.entities:
                plt.figure(figsize=(14, 6))
                entity_df = pd.DataFrame(
                    list(field_stat.entities.items()),
                    columns=['Entity', 'Count']
                )
                sns.barplot(data=entity_df, x='Entity', y='Count', color='salmon')
                plt.title(f'Named Entity Distribution - {field}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'entity_dist_{field}.png')
                plt.close()

            # Language Distribution
            if field_stat.language_dist:
                plt.figure(figsize=(10, 6))
                lang_df = pd.DataFrame(
                    list(field_stat.language_dist.items()),
                    columns=['Language', 'Percentage']
                )
                sns.barplot(data=lang_df, x='Language', y='Percentage', color='lightblue')
                plt.title(f'Language Distribution - {field}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'lang_dist_{field}.png')
                plt.close()

            # Sentiment Distribution
            if field_stat.sentiment_scores:
                plt.figure(figsize=(10, 6))
                sentiment_df = pd.DataFrame(
                    list(field_stat.sentiment_scores.items()),
                    columns=['Sentiment', 'Percentage']
                )
                sns.barplot(data=sentiment_df, x='Sentiment', y='Percentage', color='purple')
                plt.title(f'Sentiment Distribution - {field}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / f'sentiment_dist_{field}.png')
                plt.close()