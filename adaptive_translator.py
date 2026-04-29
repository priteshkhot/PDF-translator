# adaptive_translator.py
import sqlite3
import re
import time
import logging
import os
import google.generativeai as genai  # Add this import
import nltk
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import torch
from textstat import flesch_kincaid_grade
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

class Language(Enum):
    HINDI    = "hi_IN"
    BENGALI  = "bn_IN"
    TAMIL    = "ta_IN"
    TELUGU   = "te_IN"
    MARATHI  = "mr_IN"
    GUJARATI = "gu_IN"
    KANNADA  = "kn_IN"
    MALAYALAM= "ml_IN"
    PUNJABI  = "pa_IN"
    URDU     = "ur_PK"

# Regional context mapping for cultural adaptation
REGIONAL_CONTEXTS = {
    Language.HINDI: {
        "region": "North India",
        "crops": ["गेहूं (wheat)", "चावल (rice)", "गन्ना (sugarcane)", "आम (mango)"],
        "occupations": ["किसान (farmer)", "दुकानदार (shopkeeper)", "मजदूर (laborer)", "शिल्पकार (craftsman)"],
        "examples": ["गाँव की दुकान (village shop)", "खेत में काम (farm work)", "मेला (village fair)"],
        "currency": "रुपया",
        "measurement": "एकड़ (acre), किलो (kg)"
    },
    # ... [Other languages] ...
}

@dataclass
class TranslationConfig:
    target_language: Language
    grade_level: int
    use_gemini: bool = False
    gemini_api_key: Optional[str] = None
    cultural_adaptation: bool = True

@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    confidence_score: float
    grade_level: float
    technical_terms_used: List[str]
    processing_time: float
    cultural_adaptations: List[str] = None

class GlossaryManager:
    """Manages a simple Hindi–English glossary stored in SQLite."""

    def __init__(self, db_path: str = "Actual.db"):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._initialize_glossary()

    def _connect(self):
        """Create database connection with error handling"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logger.info(f"Connected to glossary database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Creating in-memory database as fallback")
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)

    def _initialize_glossary(self):
        """Create glossary table and add sample data if needed"""
        try:
            # Create table if not exists
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS glossary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    english_translation TEXT NOT NULL,
                    hindi_term TEXT NOT NULL,
                    definition TEXT
                )
            """)
            self.conn.commit()
            
            # Check if table is empty and add sample data
            cur = self.conn.execute("SELECT COUNT(*) FROM glossary")
            if cur.fetchone()[0] == 0:
                self._add_sample_terms()
                logger.info("Added sample terms to glossary")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize glossary: {e}")

    def _add_sample_terms(self):
        """Add sample terms to the glossary"""
        sample_terms = [
            ("GDP", "सकल घरेलू उत्पाद", "Total market value of goods and services"),
            ("inflation", "मुद्रास्फीति", "General increase in prices"),
            ("export", "निर्यात", "Sending goods to another country for sale"),
            ("import", "आयात", "Bringing goods from another country for sale"),
            ("investment", "निवेश", "Putting money into something to make a profit"),
            ("trade", "व्यापार", "Buying and selling of goods and services"),
            ("market", "बाजार", "Place where buyers and sellers meet"),
            ("economy", "अर्थव्यवस्था", "System of production, distribution, and consumption")
        ]
        
        self.conn.executemany(
            "INSERT INTO glossary (english_translation, hindi_term, definition) VALUES (?, ?, ?)",
            sample_terms
        )
        self.conn.commit()

    def get_term(self, english: str) -> Optional[str]:
        """Get translated term from glossary"""
        if not self.conn:
            return None
        try:
            cur = self.conn.execute(
                "SELECT hindi_term FROM glossary WHERE english_translation = ? COLLATE NOCASE LIMIT 1",
                (english.strip(),)
            )
            row = cur.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Database query failed: {e}")
            return None

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

class AdaptiveSTEMTranslator:
    """Enhanced STEM translator with cultural adaptation for rural education."""

    def __init__(self, glossary_db: str, config: TranslationConfig):
        self.config = config
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize mBART
        try:
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
            logger.info(f"Loading mBART model: {model_name}")
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logger.info("mBART model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load mBART model: {e}")
            raise

        # Initialize glossary
        self.glossary = GlossaryManager(glossary_db)

        # Technical terms list
        self.technical_terms = [
            "inflation", "gdp", "gross domestic product", "investment", "market", "trade", 
            "export", "import", "policy", "growth", "development", "productivity",
            "foreign direct investment", "balance of payments", "tariff", "subsidy",
            "supply", "demand", "price", "cost", "profit", "revenue", "budget"
        ]
        
        # Initialize Gemini if enabled
        self.gemini_model = None
        if config.use_gemini and config.gemini_api_key:
            try:
                genai.configure(api_key=config.gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini model initialized for refinement")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                logger.warning("Continuing without Gemini refinement")

    def _extract_terms(self, text: str) -> List[str]:
        """Extract technical terms from text"""
        found = []
        low = text.lower()
        for term in sorted(self.technical_terms, key=len, reverse=True):
            if term in low:
                found.append(term)
                low = low.replace(term, " " * len(term))
        return list(set(found))

    def _get_cultural_context(self, lang: Language) -> str:
        """Get regional cultural context for the target language"""
        if lang not in REGIONAL_CONTEXTS:
            return "Use familiar local examples from rural Indian context."
        
        ctx = REGIONAL_CONTEXTS[lang]
        return f"""
REGIONAL CONTEXT for {ctx['region']}:
- Local crops: {', '.join(ctx['crops'])}
- Common occupations: {', '.join(ctx['occupations'])}
- Familiar examples: {', '.join(ctx['examples'])}
- Currency: {ctx['currency']}
- Measurements: {ctx['measurement']}

Use these familiar concepts to explain economic and STEM concepts.
"""

    def _mbart_translate(self, text: str) -> str:
        """Perform mBART translation"""
        try:
            self.tokenizer.src_lang = "en_XX"
            bos = self.tokenizer.lang_code_to_id[self.config.target_language.value]
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=bos,
                    num_beams=5,
                    max_length=512,
                    early_stopping=True
                )
            
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
        except Exception as e:
            logger.error(f"mBART translation failed: {e}")
            return f"[Translation failed: {str(e)}]"
    
    def _gemini_refine(self, original: str, translation: str, terms: List[str]) -> Tuple[str, List[str]]:
        """Use Gemini to refine the translation for educational purposes"""
        if not self.gemini_model:
            return translation, []
        
        try:
            # Build glossary context
            glossary_ctx = ""
            for t in terms:
                gt = self.glossary.get_term(t)
                if gt:
                    glossary_ctx += f"- {t} → {gt}\n"

            # Get cultural context
            cultural_ctx = self._get_cultural_context(self.config.target_language)
            
            # Grade-level specific instructions
            grade_instructions = {
                6: "Use very simple words and short sentences. Compare to things children see daily.",
                7: "Use simple language with basic examples from village life.",
                8: "Use clear examples from farming, local business, and daily activities.",
                9: "Include practical examples from agriculture, small business, and local economy.",
                10: "Connect concepts to real economic situations in rural areas.",
                11: "Use detailed examples from agricultural economics and rural development.",
                12: "Include comprehensive examples from rural economic development and policy."
            }

            grade_instruction = grade_instructions.get(self.config.grade_level, 
                                                     "Use age-appropriate language and examples.")

            prompt = f"""
You are an expert educational translator specializing in STEM education for rural Indian students.

ORIGINAL ENGLISH: {original}

INITIAL TRANSLATION: {translation}

MANDATORY TECHNICAL TERMS (use these exact translations):
{glossary_ctx}

{cultural_ctx}

EDUCATIONAL REQUIREMENTS:
- Target Grade: {self.config.grade_level}
- Language: {self.config.target_language.name}
- Context: Rural Indian students
- Instruction: {grade_instruction}

ADAPTATION GUIDELINES:
1. Replace abstract economic concepts with concrete rural examples
2. Use familiar agricultural and village scenarios
3. Simplify complex sentences while maintaining accuracy
4. Include relatable analogies (farming, local trade, village economy)
5. Use active voice and direct statements
6. Avoid jargon; explain technical terms simply
7. Make examples gender-inclusive and culturally sensitive

EXAMPLES OF GOOD ADAPTATIONS:
- "Economic growth" → "When a village's total production and income increases"
- "Market forces" → "How prices change based on what people want to buy and sell"
- "Investment" → "Using money to start or improve a business, like buying better seeds or tools"

Provide ONLY the refined, culturally adapted translation in {self.config.target_language.name}.
DO NOT include any English text or explanations.
"""

            response = self.gemini_model.generate_content(prompt)
            refined_text = response.text.strip()
            
            # Track cultural adaptations
            adaptations = []
            if "village" in refined_text.lower() or "गांव" in refined_text:
                adaptations.append("Added village context")
            if any(crop in refined_text.lower() for crop in ["farm", "crop", "खेत", "फसल"]):
                adaptations.append("Used agricultural examples")
            if any(word in refined_text.lower() for word in ["local", "स्थानीय"]):
                adaptations.append("Localized context")
                
            return refined_text, adaptations
            
        except Exception as e:
            logger.error(f"Gemini refinement failed: {e}")
            return translation, []

    def translate_chunk(self, text: str) -> TranslationResult:
        """Main translation pipeline with optional Gemini refinement"""
        start = time.time()
        cultural_adaptations = []
        confidence = 0.85  # Base confidence without Gemini

        try:
            # 1) Extract technical terms
            tech_terms = self._extract_terms(text)
            logger.info(f"Extracted {len(tech_terms)} technical terms: {tech_terms}")

            # 2) mBART translate
            trans = self._mbart_translate(text)
            logger.info("mBART translation completed")

            # 3) Enforce glossary consistency
            for t in tech_terms:
                gt = self.glossary.get_term(t)
                if gt:
                    patterns = [
                        rf'\b{re.escape(t)}\b',
                        rf'\b{re.escape(t.title())}\b',
                        rf'\b{re.escape(t.upper())}\b'
                    ]
                    for pattern in patterns:
                        trans = re.sub(pattern, gt, trans, flags=re.IGNORECASE)

            # 4) Gemini refinement (if enabled)
            if self.gemini_model:
                trans, cultural_adaptations = self._gemini_refine(text, trans, tech_terms)
                logger.info("Gemini refinement completed")
                confidence = 0.95  # Higher confidence with Gemini

            # 5) Calculate metrics
            try:
                grade = flesch_kincaid_grade(trans)
            except:
                grade = self.config.grade_level
            
            secs = time.time() - start

            return TranslationResult(
                original_text=text.strip(),
                translated_text=trans.strip(),
                confidence_score=confidence,
                grade_level=grade,
                technical_terms_used=tech_terms,
                processing_time=secs,
                cultural_adaptations=cultural_adaptations
            )
            
        except Exception as e:
            logger.error(f"Translation pipeline failed: {e}")
            return TranslationResult(
                original_text=text.strip(),
                translated_text=f"[Translation Error: {str(e)}]",
                confidence_score=0.0,
                grade_level=0.0,
                technical_terms_used=[],
                processing_time=time.time() - start,
                cultural_adaptations=[]
            )

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'glossary'):
            self.glossary.close()

def main():
    """Main function with proper database initialization"""
    try:
        # Initialize database
        db_path = "Actual.db"
        logger.info(f"Initializing database at: {os.path.abspath(db_path)}")
        
        # Create glossary manager
        gm = GlossaryManager(db_path)
        
        # Test glossary
        print("Glossary Search Demo:")
        for term in ["GDP", "export", "inflation", "trade", "investment"]:
            translation = gm.get_term(term)
            print(f" {term} → {translation if translation else 'Not found'}")
        print()
        
        # Configuration with Gemini enabled
        cfg = TranslationConfig(
            target_language=Language.HINDI,
            grade_level=2,
            use_gemini=True,
            gemini_api_key=os.getenv("API_KEY"),  # Replace with your actual API key
            cultural_adaptation=True
        )
        
        print("=== Translator Initialization ===")
        translator = AdaptiveSTEMTranslator(db_path, cfg)

        sample = """
        Current Socio-Economic state of the world has been diminshed.
        """
        
        print("=== Translation Process ===")
        result = translator.translate_chunk(sample)
        
        print("\n=== Enhanced Translation Result ===")
        print("Original:", result.original_text)
        print("Translated:", result.translated_text)
        print(f"Grade Level: {result.grade_level:.1f}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Technical Terms: {result.technical_terms_used}")
        print(f"Cultural Adaptations: {result.cultural_adaptations}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error running demo: {e}")

if __name__ == "__main__":
    main()