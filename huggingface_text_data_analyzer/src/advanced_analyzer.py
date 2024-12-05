from typing import Dict, List, Optional
from transformers import pipeline
import spacy
import fasttext
from dataclasses import dataclass
from collections import Counter
from rich.console import Console

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
    overall_stats: AdvancedFieldStats

class AdvancedAnalyzer(BaseAnalyzer):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        use_pos: bool = True,
        use_ner: bool = True,
        use_lang: bool = True,
        use_sentiment: bool = True,
        use_topics: bool = True,
        console: Optional[Console] = None
    ):
        super().__init__(dataset_name, split, console=console)
        self.use_pos = use_pos
        self.use_ner = use_ner
        self.use_lang = use_lang
        self.use_sentiment = use_sentiment
        self.use_topics = use_topics
        
        self.console.log("Loading advanced analysis models")
        
        if use_pos or use_ner:
            with self.console.status("Loading spaCy model..."):
                self.nlp = spacy.load("en_core_web_sm")
            self.console.log("Loaded spaCy model")
            
        if use_lang:
            with self.console.status("Loading FastText model..."):
                self.lang_model = fasttext.load_model("lid.176.bin")
            self.console.log("Loaded FastText model")
            
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
    
    def detect_language(self, text: str) -> Dict[str, float]:
        if not self.use_lang:
            return {}
        predictions = self.lang_model.predict(text, k=3)
        return {lang: prob for lang, prob in zip(*predictions)}
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not self.use_sentiment:
            return {}
        result = self.sentiment_analyzer(text)[0]
        return {"score": result["score"], "label": result["label"]}
    
    def analyze_field_advanced(self, texts: List[str], field_name: str) -> AdvancedFieldStats:
        self.console.log(f"Running advanced analysis on field: {field_name}")
        
        pos_dist = {}
        entities = {}
        lang_dist = {}
        sentiment_scores = {}
        
        with create_progress() as progress:
            total = len(texts)
            pos_task = progress.add_task(f"POS tagging - {field_name}", total=total if self.use_pos else 0)
            ner_task = progress.add_task(f"NER analysis - {field_name}", total=total if self.use_ner else 0)
            lang_task = progress.add_task(f"Language detection - {field_name}", total=total if self.use_lang else 0)
            sent_task = progress.add_task(f"Sentiment analysis - {field_name}", total=total if self.use_sentiment else 0)
            
            for text in texts:
                if self.use_pos:
                    text_pos = self.get_pos_distribution(text)
                    for pos, freq in text_pos.items():
                        pos_dist[pos] = pos_dist.get(pos, 0) + freq
                    progress.advance(pos_task)
                    
                if self.use_ner:
                    text_ents = self.get_entities(text)
                    for ent, count in text_ents.items():
                        entities[ent] = entities.get(ent, 0) + count
                    progress.advance(ner_task)
                    
                if self.use_lang:
                    text_lang = self.detect_language(text)
                    for lang, prob in text_lang.items():
                        lang_dist[lang] = lang_dist.get(lang, 0) + prob
                    progress.advance(lang_task)
                    
                if self.use_sentiment:
                    sentiment = self.analyze_sentiment(text)
                    sentiment_scores[sentiment["label"]] = sentiment_scores.get(sentiment["label"], 0) + 1
                    progress.advance(sent_task)
        
        n_texts = len(texts)
        if n_texts > 0:
            for pos in pos_dist:
                pos_dist[pos] /= n_texts
            for lang in lang_dist:
                lang_dist[lang] /= n_texts
                
        return AdvancedFieldStats(
            pos_distribution=pos_dist,
            entities=entities,
            language_dist=lang_dist,
            sentiment_scores={k: v/n_texts for k, v in sentiment_scores.items()},
            topics=[]
        )
    
    def analyze_advanced(self) -> AdvancedDatasetStats:
        text_fields = [field for field in self.dataset.features if isinstance(self.dataset.features[field], (str, dict))]
        self.console.log(f"Running advanced analysis on {len(text_fields)} fields")
        
        field_stats = {}
        all_texts = []
        
        for field in text_fields:
            texts = self.dataset[field]
            all_texts.extend(texts)
            field_stats[field] = self.analyze_field_advanced(texts, field)
            
        self.console.log("Calculating overall advanced statistics")
        overall_stats = self.analyze_field_advanced(all_texts, "overall")
        
        return AdvancedDatasetStats(field_stats=field_stats, overall_stats=overall_stats)