from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datasets import load_dataset, Features
from collections import Counter
import re
import statistics
from transformers import PreTrainedTokenizer
from rich.console import Console
import torch
from itertools import islice
import os
import pickle
from pathlib import Path

from .utils import create_progress

@dataclass
class FieldStats:
    avg_length: float
    word_distribution: Dict[str, int]
    junk_frequency: float
    avg_token_length: Optional[float] = None

@dataclass
class DatasetStats:
    field_stats: Dict[str, FieldStats]
    overall_stats: FieldStats

class BaseAnalyzer:
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        cache_tokenized: bool = True,
        console: Optional[Console] = None,
        chat_field: Optional[str] = None,
        batch_size: int = 32,
        fields: Optional[List[str]] = None,
        cache_dir: Optional[str] = None
    ):
        self.console = console or Console()
        self.console.log(f"Loading dataset: {dataset_name} (split: {split})")
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.cache_tokenized = cache_tokenized
        self.cached_tokens = {}
        self.chat_field = chat_field
        self.batch_size = batch_size
        self.fields = fields
        self.cache_dir = cache_dir or Path("./cache")
        self.dataset_name = dataset_name
        self.split = split
        
        if self.cache_tokenized:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        if tokenizer:
            self.console.log(f"Using tokenizer: {tokenizer.__class__.__name__}")
            if chat_field:
                if hasattr(tokenizer, "chat_template"):
                    self.console.log(f"Chat template will be applied to field: {chat_field}")
                else:
                    self.console.log("[yellow]Warning: Tokenizer does not have a chat template[/yellow]")

    def get_cache_path(self, field_name: str) -> Path:
        tokenizer_name = self.tokenizer.__class__.__name__ if self.tokenizer else "no_tokenizer"
        return Path(self.cache_dir) / f"{self.dataset_name}_{self.split}_{field_name}_{tokenizer_name}_tokens.pkl"

    def load_cached_tokens(self, field_name: str) -> Optional[List[List[int]]]:
        cache_path = self.get_cache_path(field_name)
        if cache_path.exists():
            self.console.log(f"Loading cached tokens for {field_name}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_tokens_to_cache(self, field_name: str, tokens: List[List[int]]):
        cache_path = self.get_cache_path(field_name)
        self.console.log(f"Saving tokens to cache for {field_name}")
        with open(cache_path, 'wb') as f:
            pickle.dump(tokens, f)

    def calculate_overall_stats(self, field_stats: Dict[str, FieldStats]) -> FieldStats:
        total_texts = sum(len(self.dataset[field]) for field in field_stats.keys())
        
        combined_word_dist = Counter()
        for stats in field_stats.values():
            combined_word_dist.update(stats.word_distribution)
        
        weighted_avg_length = 0
        weighted_junk_freq = 0
        weighted_token_length = 0
        total_weight = 0
        
        for field, stats in field_stats.items():
            field_weight = len(self.dataset[field])
            total_weight += field_weight
            weighted_avg_length += stats.avg_length * field_weight
            weighted_junk_freq += stats.junk_frequency * field_weight
            if stats.avg_token_length is not None:
                weighted_token_length += stats.avg_token_length * field_weight
        
        return FieldStats(
            avg_length=weighted_avg_length / total_weight if total_weight > 0 else 0,
            word_distribution=dict(combined_word_dist),
            junk_frequency=weighted_junk_freq / total_weight if total_weight > 0 else 0,
            avg_token_length=weighted_token_length / total_weight if total_weight > 0 and self.tokenizer else None
        )

    def batch_tokenize(self, texts: List[str], field_name: str) -> List[List[int]]:
        if not self.tokenizer:
            return []

        if self.cache_tokenized:
            cached_tokens = self.load_cached_tokens(field_name)
            if cached_tokens is not None:
                return cached_tokens

        all_tokens = []
        is_chat = field_name == self.chat_field and hasattr(self.tokenizer, "chat_template")

        with create_progress() as progress:
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            task = progress.add_task(
                f"Tokenizing {field_name}...", 
                total=total_batches
            )

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                if is_chat:
                    batch_texts = [
                        self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
                        for text in batch_texts
                    ]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=2048
                )
                
                attention_mask = encoded.attention_mask
                lengths = attention_mask.sum(dim=1).tolist()
                
                for ids, length in zip(encoded.input_ids, lengths):
                    all_tokens.append(ids[:int(length)].tolist())
                
                progress.advance(task)

        if self.cache_tokenized:
            self.save_tokens_to_cache(field_name, all_tokens)

        return all_tokens

    def analyze_field(self, texts: List[Any], field_name: str) -> FieldStats:
        self.console.log(f"Analyzing field: {field_name}")
        
        # Process texts in batches with progress bar
        processed_texts = self.process_texts_in_batches(texts)
        
        total_texts = len(processed_texts)
        word_counts = []
        word_dist = {}
        junk_scores = []
        
        with create_progress() as progress:
            stats_task = progress.add_task(
                f"Processing text statistics for {field_name}",
                total=total_texts,
                visible=True  # Make sure it's visible
            )
            
            for i in range(0, total_texts, self.batch_size):
                batch = processed_texts[i:i + self.batch_size]
                for text in batch:
                    if text:
                        word_counts.append(self.count_words(text))
                        dist = self.get_word_distribution(text)
                        for word, count in dist.items():
                            word_dist[word] = word_dist.get(word, 0) + count
                        junk_scores.append(self.detect_junk(text))
                progress.advance(stats_task, len(batch))
        
        if not word_counts:
            self.console.log(f"[yellow]Warning: No valid texts found in field {field_name}[/yellow]")
            return FieldStats(
                avg_length=0.0,
                word_distribution={},
                junk_frequency=0.0,
                avg_token_length=0.0 if self.tokenizer else None
            )
        
        token_length = None
        if self.tokenizer:
            self.console.log(f"Batch tokenizing {field_name}")
            valid_texts = [text for text in processed_texts if text]
            tokens = self.batch_tokenize(valid_texts, field_name)
            
            if self.cache_tokenized:
                self.cached_tokens[field_name] = tokens
            
            if tokens:
                token_length = statistics.mean([len(t) for t in tokens])
        
        return FieldStats(
            avg_length=statistics.mean(word_counts),
            word_distribution=word_dist,
            junk_frequency=statistics.mean(junk_scores) if junk_scores else 0.0,
            avg_token_length=token_length
        )

    def analyze(self) -> DatasetStats:
        # Find text fields
        available_text_fields = [
            field for field, feature in self.dataset.features.items()
            if self.is_text_feature(feature)
        ]
        
        if self.fields:
            invalid_fields = [f for f in self.fields if f not in available_text_fields]
            if invalid_fields:
                self.console.log(f"[yellow]Warning: Following fields were not found or are not text fields: {', '.join(invalid_fields)}[/yellow]")
            
            text_fields = [f for f in self.fields if f in available_text_fields]
            if not text_fields:
                raise ValueError("None of the specified fields were found or are text fields")
        else:
            text_fields = available_text_fields
            
        if not text_fields:
            self.console.log("[red]Error: No text fields found in dataset[/red]")
            raise ValueError("No text fields found in dataset")
            
        self.console.log(f"Found {len(text_fields)} text fields to analyze: {', '.join(text_fields)}")
        
        field_stats = {}
        
        for field in text_fields:
            texts = self.dataset[field]
            field_stats[field] = self.analyze_field(texts, field)
            
        self.console.log("Calculating overall statistics")
        overall_stats = self.calculate_overall_stats(field_stats)
        
        return DatasetStats(field_stats=field_stats, overall_stats=overall_stats)

    def process_texts_in_batches(self, texts: List[str]) -> List[str]:
        """Process texts in batches, ensuring they're all strings."""
        processed_texts = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            processed_texts.extend([self.extract_text(text) for text in batch])
        return processed_texts

    def is_text_feature(self, feature) -> bool:
        """Check if a dataset feature is a text field."""
        feature_str = str(feature).lower()
        
        text_patterns = [
            'string',
            'text',
            'sequence',
            'str',
            '[str]',
            'value(dtype=\'string\'',
            'listvalue(feature=value(dtype=\'string\'',
        ]
        
        return any(pattern in feature_str for pattern in text_patterns)
    
    def extract_text(self, field_data: Any) -> str:
        """Safely extract text from field data."""
        if isinstance(field_data, (str, bytes)):
            return str(field_data)
        elif isinstance(field_data, (list, tuple)) and len(field_data) > 0:
            return " ".join(str(x) for x in field_data)
        elif hasattr(field_data, '__str__'):
            return str(field_data)
        return ""
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def get_word_distribution(self, text: str) -> Dict[str, int]:
        """Get distribution of words in text."""
        return dict(Counter(text.lower().split()))
    
    def detect_junk(self, text: str) -> float:
        """Detect proportion of junk characters in text."""
        html_pattern = r'<[^>]+>'
        special_chars_pattern = r'[^\w\s]'
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        junk_chars = len(re.findall(html_pattern, text)) + len(re.findall(special_chars_pattern, text))
        return junk_chars / total_chars