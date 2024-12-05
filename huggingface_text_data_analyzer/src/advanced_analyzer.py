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
from .utils import create_progress

@dataclass
class AdvancedFieldStats:
    pos_distribution: Dict[str, float]
    entities: Dict[str, int]
    language_dist: Dict[str, float]
    sentiment_scores: Dict[str, float]
    topics: List[str]

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
        batch_size: int = 32,  # Added batch_size parameter
        console: Optional[Console] = None
    ):
        super().__init__(dataset_name, split=split, subset=subset, console=console, fields=fields)
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.use_lang = use_lang
        self.use_sentiment = use_sentiment
        self.use_topics = use_topics
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.console.log("Loading advanced analysis models")
        
        if use_pos or use_ner:
            with self.console.status("Loading spaCy model..."):
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    # Enable batch processing in spaCy
                    if spacy.__version__ >= "3.0.0":
                        self.nlp.enable_pipe("parser")
                        self.nlp.enable_pipe("ner")
                    self.console.log("Loaded spaCy model")
                except OSError:
                    self.console.log("[yellow]Downloading spaCy model...[/yellow]")
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    self.console.log("Loaded spaCy model")
        
        if use_lang:
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
            
        if use_sentiment:
            with self.console.status("Loading sentiment analysis model..."):
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    batch_size=self.batch_size,
                    device=self.device
                )
                self.console.log("Loaded sentiment analysis model")

    def batch_texts(self, texts: Sequence[str]) -> List[List[str]]:
        """Split texts into batches."""
        return [
            list(islice(texts, i, i + self.batch_size))
            for i in range(0, len(texts), self.batch_size)
        ]

    def process_spacy_batch(self, batch: List[str]) -> List[spacy.tokens.Doc]:
        """Process a batch of texts with spaCy."""
        return list(self.nlp.pipe(batch))

    def process_language_batch(self, batch: List[str]) -> List[str]:
        """Process a batch of texts with language detection."""
        try:
            results = self.lang_model(batch, truncation=True, max_length=512)
            return [result['label'] for result in results]
        except Exception as e:
            self.console.log(f"[yellow]Warning: Language detection failed for batch: {str(e)}[/yellow]")
            return ["unknown"] * len(batch)

    def process_sentiment_batch(self, batch: List[str]) -> List[str]:
        """Process a batch of texts with sentiment analysis."""
        try:
            # Truncate long texts
            truncated_batch = [" ".join(text.split()[:512]) for text in batch]
            results = self.sentiment_analyzer(
                truncated_batch,
                truncation=True,
                max_length=512,
                padding=True
            )
            return [result['label'] for result in results]
        except Exception as e:
            self.console.log(f"[yellow]Warning: Sentiment analysis failed for batch: {str(e)}[/yellow]")
            return ["NEUTRAL"] * len(batch)

    def analyze_field_advanced(self, texts: List[str], field_name: str) -> AdvancedFieldStats:
        """Analyze a single field with advanced NLP features using batching."""
        self.console.log(f"Running advanced analysis on field: {field_name}")
        
        # Initialize counters
        pos_dist = Counter()
        entities = Counter()
        lang_dist = Counter()
        sentiment_scores = Counter()
        
        # Filter out empty texts
        texts = [t for t in texts if t]
        if not texts:
            return AdvancedFieldStats(
                pos_distribution={},
                entities={},
                language_dist={},
                sentiment_scores={},
                topics=[]
            )
        
        # Create batches
        batches = self.batch_texts(texts)
        total_batches = len(batches)
        
        with create_progress() as progress:
            # Create progress tasks
            pos_task = progress.add_task(
                f"POS/NER tagging - {field_name}", 
                total=total_batches if (self.use_pos or self.use_ner) else 0
            )
            lang_task = progress.add_task(
                f"Language detection - {field_name}", 
                total=total_batches if self.use_lang else 0
            )
            sent_task = progress.add_task(
                f"Sentiment analysis - {field_name}", 
                total=total_batches if self.use_sentiment else 0
            )
            
            # Process batches
            for batch in batches:
                # SpaCy processing (POS and NER)
                if self.use_pos or self.use_ner:
                    docs = self.process_spacy_batch(batch)
                    for doc in docs:
                        if self.use_pos:
                            pos_dist.update(token.pos_ for token in doc)
                        if self.use_ner:
                            entities.update(ent.label_ for ent in doc.ents)
                    progress.advance(pos_task)
                
                # Language detection
                if self.use_lang:
                    langs = self.process_language_batch(batch)
                    lang_dist.update(langs)
                    progress.advance(lang_task)
                
                # Sentiment analysis
                if self.use_sentiment:
                    sentiments = self.process_sentiment_batch(batch)
                    sentiment_scores.update(sentiments)
                    progress.advance(sent_task)
        
        # Calculate distributions
        total_texts = len(texts)
        pos_distribution = {pos: count/total_texts for pos, count in pos_dist.items()}
        language_dist = {lang: count/total_texts for lang, count in lang_dist.items()}
        sentiment_dist = {label: count/total_texts for label, count in sentiment_scores.items()}

        return AdvancedFieldStats(
            pos_distribution=pos_distribution,
            entities=dict(entities),
            language_dist=language_dist,
            sentiment_scores=sentiment_dist,
            topics=[]  # Topics feature not implemented yet
        )

    def analyze_advanced(self) -> AdvancedDatasetStats:
        """Run advanced analysis on all text fields in the dataset."""
        # Find text fields (reuse from parent class)
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
            if texts:  # Only analyze fields with non-empty texts
                field_stats[field] = self.analyze_field_advanced(texts, field)
                        
        return AdvancedDatasetStats(field_stats=field_stats)