from typing import Dict, List, Optional
from transformers import pipeline
import spacy
from dataclasses import dataclass
from collections import Counter
from rich.console import Console
import os

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
        console: Optional[Console] = None
    ):
        super().__init__(dataset_name, split=split, subset=subset, console=console, fields=fields)
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.use_lang = use_lang
        self.use_sentiment = use_sentiment
        self.use_topics = use_topics
        
        self.console.log("Loading advanced analysis models")
        
        if use_pos or use_ner:
            with self.console.status("Loading spaCy model..."):
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.console.log("Loaded spaCy model")
                except OSError:
                    self.console.log("[yellow]Downloading spaCy model...[/yellow]")
                    os.system("python -m spacy download en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    self.console.log("Loaded spaCy model")
            
        if use_lang:
            with self.console.status("Loading language detection model..."):
                try:
                    self.lang_model = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
                    self.console.log("Loaded language detection model")
                except Exception as e:
                    self.console.log(f"[red]Failed to load language detection model: {str(e)}[/red]")
                    self.use_lang = False
            
        if use_sentiment:
            with self.console.status("Loading sentiment analysis model..."):
                self.sentiment_analyzer = pipeline("sentiment-analysis")
                self.console.log("Loaded sentiment analysis model")

    def get_pos_distribution(self, text: str) -> Dict[str, float]:
        if not self.use_pos:
            return {}
        doc = self.nlp(text)
        pos_counts = Counter([token.pos_ for token in doc])
        total = sum(pos_counts.values())
        return {pos: count/total for pos, count in pos_counts.items()}
    
    def get_entities(self, text: str) -> Dict[str, int]:
        if not self.use_ner:
            return {}
        doc = self.nlp(text)
        return Counter([ent.label_ for ent in doc.ents])

    def detect_language(self, text: str) -> str:
        """Detect language of text with proper error handling."""
        if not self.use_lang or not text:
            return "unknown"
            
        try:
            result = self.lang_model(text, truncation=True, max_length=512)
            return result[0]['label'] if result else "unknown"
        except Exception as e:
            self.console.log(f"[yellow]Warning: Language detection failed: {str(e)}[/yellow]")
            return "unknown"

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of a text, with proper error handling."""
        try:
            # First truncate the text if it's too long
            if len(text.split()) > 512:
                text = " ".join(text.split()[:512])
            
            # Get the prediction and extract just the label
            result = self.sentiment_analyzer(
                text,
                truncation=True,
                max_length=512,
                padding=True
            )
            # Result is a list with one dict, extract just the label
            return result[0]['label'] if result else "NEUTRAL"
        except Exception as e:
            self.console.log(f"[yellow]Warning: Sentiment analysis failed: {str(e)}[/yellow]")
            return "NEUTRAL"

    def analyze_field_advanced(self, texts: List[str], field_name: str) -> AdvancedFieldStats:
        """Analyze a single field with advanced NLP features."""
        self.console.log(f"Running advanced analysis on field: {field_name}")
        
        pos_dist = Counter()
        entities = Counter()
        lang_dist = Counter()
        sentiment_scores = Counter()
        
        with create_progress() as progress:
            total = len(texts)
            pos_task = progress.add_task(f"POS tagging - {field_name}", total=total if self.use_pos else 0)
            ner_task = progress.add_task(f"NER analysis - {field_name}", total=total if self.use_ner else 0)
            lang_task = progress.add_task(f"Language detection - {field_name}", total=total if self.use_lang else 0)
            sent_task = progress.add_task(f"Sentiment analysis - {field_name}", total=total if self.use_sentiment else 0)
            
            for text in texts:
                if not text:  # Skip empty texts
                    continue
                    
                # try:
                if self.use_pos or self.use_ner:
                    doc = self.nlp(text)
                    
                    if self.use_pos:
                        pos_dist.update(token.pos_ for token in doc)
                        progress.advance(pos_task)
                        
                    if self.use_ner:
                        entities.update(ent.label_ for ent in doc.ents)
                        progress.advance(ner_task)
                
                if self.use_lang:
                    lang = self.detect_language(text)
                    if lang:  # Only update if we got a valid language
                        lang_dist.update([lang])
                    progress.advance(lang_task)
                    
                if self.use_sentiment:
                    sentiment_label = self.analyze_sentiment(text)
                    if sentiment_label:  # Only update if we got a valid sentiment
                        sentiment_scores.update([sentiment_label])
                    progress.advance(sent_task)
                        
                # except Exception as e:
                #     self.console.log(f"[yellow]Warning: Error processing text in {field_name}: {str(e)}[/yellow]")
                #     continue

        # Calculate distributions
        total_texts = len([t for t in texts if t])  # Count non-empty texts
        if total_texts > 0:
            pos_distribution = {pos: count/total_texts for pos, count in pos_dist.items()}
            language_dist = {lang: count/total_texts for lang, count in lang_dist.items()}
            sentiment_dist = {label: count/total_texts for label, count in sentiment_scores.items()}
        else:
            pos_distribution = {}
            language_dist = {}
            sentiment_dist = {}

        return AdvancedFieldStats(
            pos_distribution=pos_distribution,
            entities=dict(entities),
            language_dist=language_dist,
            sentiment_scores=sentiment_dist,
            topics=[]
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
        all_texts = []
        
        for field in text_fields:
            texts = [self.extract_text(text) for text in self.dataset[field]]
            texts = [t for t in texts if t]  # Filter out empty texts
            if texts:  # Only analyze fields with non-empty texts
                all_texts.extend(texts)
                field_stats[field] = self.analyze_field_advanced(texts, field)
                        
        return AdvancedDatasetStats(field_stats=field_stats)