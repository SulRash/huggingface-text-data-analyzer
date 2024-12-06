from typing import Dict, List, Optional, Sequence
from transformers import pipeline
import spacy
from dataclasses import dataclass
from collections import Counter
from rich.console import Console
import os
from itertools import islice
import torch

from .base_analyzer import BaseAnalyzer
from .utils import CacheManager, create_progress

@dataclass
class AdvancedFieldStats:
    pos_distribution: Optional[Dict[str, float]] = None
    entities: Optional[Dict[str, int]] = None
    language_dist: Optional[Dict[str, float]] = None
    sentiment_scores: Optional[Dict[str, float]] = None
    topics: Optional[List[str]] = None

@dataclass
class AdvancedDatasetStats:
    field_stats: Dict[str, AdvancedFieldStats]

class AdvancedAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: Optional[str] = None,
        fields: Optional[List[str]] = None,
        use_pos: bool = True,
        use_ner: bool = True,
        use_lang: bool = True,
        use_sentiment: bool = True,
        use_topics: bool = True,
        batch_size: int = 32,
        console: Optional[Console] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        super().__init__(dataset_name, split=split, subset=subset, console=console, fields=fields, cache_manager=cache_manager)
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.use_lang = use_lang
        self.use_sentiment = use_sentiment
        self.use_topics = use_topics
        self.batch_size = batch_size
        
        # GPU detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.console.log("[green]GPU detected - will use for transformer models[/green]")
        else:
            self.console.log("[yellow]No GPU detected - using CPU[/yellow]")

        # Load required models based on enabled features
        self._load_models()

    def _load_models(self) -> None:
        """Load all required models based on enabled features."""
        if self.use_pos or self.use_ner:
            self._load_spacy_model()
        if self.use_lang:
            self._load_language_model()
        if self.use_sentiment:
            self._load_sentiment_model()

    def _load_spacy_model(self) -> None:
        """Load and configure spaCy model."""
        with self.console.status("Loading spaCy model..."):
            try:
                self.nlp = spacy.load("en_core_web_sm")
                if spacy.__version__ >= "3.0.0":
                    self.nlp.enable_pipe("parser")
                    self.nlp.enable_pipe("ner")
                self.console.log("Loaded spaCy model")
            except OSError:
                self.console.log("[yellow]Downloading spaCy model...[/yellow]")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.console.log("Loaded spaCy model")

    def _load_language_model(self) -> None:
        """Load language detection model."""
        with self.console.status("Loading language detection model..."):
            try:
                self.lang_model = pipeline(
                    "text-classification",
                    model="papluca/xlm-roberta-base-language-detection",
                    batch_size=self.batch_size,
                    device=self.device
                )
                self.console.log("Loaded language detection model")
            except Exception as e:
                self.console.log(f"[red]Failed to load language detection model: {str(e)}[/red]")
                self.use_lang = False

    def _load_sentiment_model(self) -> None:
        """Load sentiment analysis model."""
        with self.console.status("Loading sentiment analysis model..."):
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    batch_size=self.batch_size,
                    device=self.device
                )
                self.console.log("Loaded sentiment analysis model")
            except Exception as e:
                self.console.log(f"[red]Failed to load sentiment analysis model: {str(e)}[/red]")
                self.use_sentiment = False

    def batch_texts(self, texts: Sequence[str]) -> List[List[str]]:
        """Split texts into batches."""
        return [
            list(islice(texts, i, i + self.batch_size))
            for i in range(0, len(texts), self.batch_size)
        ]

    def analyze_pos_ner(self, texts: List[str], field_name: str) -> tuple[Dict[str, float], Dict[str, int]]:
        """Analyze POS and NER for a field."""
        pos_dist = Counter()
        entities = Counter()
        
        batches = self.batch_texts(texts)
        with create_progress() as progress:
            task = progress.add_task(f"POS/NER analysis - {field_name}", total=len(batches))
            
            for batch in batches:
                docs = list(self.nlp.pipe(batch))
                for doc in docs:
                    if self.use_pos:
                        pos_dist.update(token.pos_ for token in doc)
                    if self.use_ner:
                        entities.update(ent.label_ for ent in doc.ents)
                progress.advance(task)

        total_texts = len(texts)
        pos_distribution = {pos: count/total_texts for pos, count in pos_dist.items()}
        return pos_distribution, dict(entities)

    def analyze_language(self, texts: List[str], field_name: str) -> Dict[str, float]:
        """Analyze language distribution for a field."""
        lang_dist = Counter()
        batches = self.batch_texts(texts)
        
        with create_progress() as progress:
            task = progress.add_task(f"Language detection - {field_name}", total=len(batches))
            
            for batch in batches:
                try:
                    results = self.lang_model(batch, truncation=True, max_length=512)
                    lang_dist.update(result['label'] for result in results)
                except Exception as e:
                    self.console.log(f"[yellow]Warning: Language detection failed for batch: {str(e)}[/yellow]")
                    lang_dist.update(["unknown"] * len(batch))
                progress.advance(task)

        total_texts = len(texts)
        return {lang: count/total_texts for lang, count in lang_dist.items()}

    def analyze_sentiment(self, texts: List[str], field_name: str) -> Dict[str, float]:
        """Analyze sentiment distribution for a field."""
        sentiment_scores = Counter()
        batches = self.batch_texts(texts)
        
        with create_progress() as progress:
            task = progress.add_task(f"Sentiment analysis - {field_name}", total=len(batches))
            
            for batch in batches:
                truncated_batch = [" ".join(text.split()[:512]) for text in batch]
                try:
                    results = self.sentiment_analyzer(
                        truncated_batch,
                        truncation=True,
                        max_length=512,
                        padding=True
                    )
                    sentiment_scores.update(result['label'] for result in results)
                except Exception as e:
                    self.console.log(f"[yellow]Warning: Sentiment analysis failed for batch: {str(e)}[/yellow]")
                    sentiment_scores.update(["NEUTRAL"] * len(batch))
                progress.advance(task)

        total_texts = len(texts)
        return {label: count/total_texts for label, count in sentiment_scores.items()}

    def analyze_field_advanced(self, texts: List[str], field_name: str) -> AdvancedFieldStats:
        """Analyze a single field with all enabled advanced features."""
        self.console.log(f"Running advanced analysis on field: {field_name}")
        
        # Filter out empty texts
        texts = [t for t in texts if t]
        if not texts:
            return AdvancedFieldStats()
        
        stats = AdvancedFieldStats()
        
        # Run enabled analyses
        if self.use_pos or self.use_ner:
            pos_dist, entities = self.analyze_pos_ner(texts, field_name)
            stats.pos_distribution = pos_dist if self.use_pos else None
            stats.entities = entities if self.use_ner else None
            
        if self.use_lang:
            stats.language_dist = self.analyze_language(texts, field_name)
            
        if self.use_sentiment:
            stats.sentiment_scores = self.analyze_sentiment(texts, field_name)
        
        return stats

    def analyze_advanced(self) -> AdvancedDatasetStats:
        """Run advanced analysis on all text fields in the dataset."""
        available_text_fields = [
            field for field, feature in self.dataset.features.items()
            if self.is_text_feature(feature)
        ]
        
        if self.fields:
            text_fields = [f for f in self.fields if f in available_text_fields]
            if not text_fields:
                raise ValueError("None of the specified fields were found or are text fields")
        else:
            text_fields = available_text_fields
            
        if not text_fields:
            raise ValueError("No text fields found in dataset")
            
        self.console.log(f"Running advanced analysis on {len(text_fields)} fields")
        field_stats = {}
        
        for field in text_fields:
            texts = [self.extract_text(text) for text in self.dataset[field]]
            if not texts:  # Skip empty fields
                continue
                
            # Filter out empty texts
            texts = [t for t in texts if t]
            if not texts:
                continue
            
            # Initialize empty stats for this field
            field_stats[field] = AdvancedFieldStats()
            
            # Run POS and NER analysis
            if self.use_pos or self.use_ner:
                pos_cache = self.cache_manager.load_cached_results(
                    self.dataset_name, self.subset, self.split, field, "pos"
                ) if self.use_pos else None
                
                ner_cache = self.cache_manager.load_cached_results(
                    self.dataset_name, self.subset, self.split, field, "ner"
                ) if self.use_ner else None
                
                if pos_cache is None or ner_cache is None:
                    pos_dist, entities = self.analyze_pos_ner(texts, field)
                    
                    if self.use_pos and pos_cache is None:
                        field_stats[field].pos_distribution = pos_dist
                        self.cache_manager.save_results(
                            pos_dist,
                            self.dataset_name,
                            self.subset,
                            self.split,
                            field,
                            "pos",
                            force=True
                        )
                    elif self.use_pos:
                        field_stats[field].pos_distribution = pos_cache
                        
                    if self.use_ner and ner_cache is None:
                        field_stats[field].entities = entities
                        self.cache_manager.save_results(
                            entities,
                            self.dataset_name,
                            self.subset,
                            self.split,
                            field,
                            "ner",
                            force=True
                        )
                    elif self.use_ner:
                        field_stats[field].entities = ner_cache
                else:
                    if self.use_pos:
                        field_stats[field].pos_distribution = pos_cache
                    if self.use_ner:
                        field_stats[field].entities = ner_cache
            
            # Language detection
            if self.use_lang:
                lang_cache = self.cache_manager.load_cached_results(
                    self.dataset_name, self.subset, self.split, field, "language"
                )
                
                if lang_cache is None:
                    lang_dist = self.analyze_language(texts, field)
                    field_stats[field].language_dist = lang_dist
                    self.cache_manager.save_results(
                        lang_dist,
                        self.dataset_name,
                        self.subset,
                        self.split,
                        field,
                        "language",
                        force=True
                    )
                else:
                    field_stats[field].language_dist = lang_cache
            
            # Sentiment analysis
            if self.use_sentiment:
                sentiment_cache = self.cache_manager.load_cached_results(
                    self.dataset_name, self.subset, self.split, field, "sentiment"
                )
                
                if sentiment_cache is None:
                    sentiment_scores = self.analyze_sentiment(texts, field)
                    field_stats[field].sentiment_scores = sentiment_scores
                    self.cache_manager.save_results(
                        sentiment_scores,
                        self.dataset_name,
                        self.subset,
                        self.split,
                        field,
                        "sentiment",
                        force=True
                    )
                else:
                    field_stats[field].sentiment_scores = sentiment_cache
                    
        return AdvancedDatasetStats(field_stats=field_stats)
