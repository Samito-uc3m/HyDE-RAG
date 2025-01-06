from pathlib import Path
from typing import Dict

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # User query
    USER_QUERY: str = "Estoy investigando sobre las quasi-partículas renormalizadas en estados antiferromagnéticos del modelo de Hubbard. ¿Podrías, por favor, buscar y encontrar documentos relevantes?"

    # General
    DATABASE_PATH: Path = Field(
        default=Path("./data/llama.db"), description="Path to the database file."
    )
    MODELS_PATH: Path = Field(
        default=Path("./models/"), description="Path to the models directory."
    )

    # Embeddings model
    EMBED_MODEL_NAME: str = Field(
        default="BAAI/bge-small-en", description="Name of the embedding model."
    )

    # OpenAI model
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini", description="Name of the OpenAI model."
    )
    OPENAI_API_KEY: str = Field(default="", description="API key for OpenAI access.")

    # Vector store configuration
    CHUNK_SIZE: int = Field(
        default=128, description="Size of text chunks for processing."
    )
    CHUNK_OVERLAP: int = Field(
        default=50, description="Overlap size between text chunks."
    )
    NODE_TOP_K: int = Field(default=20, description="Number of top nodes to retrieve.")
    DOCUMENT_TOP_K: int = Field(
        default=3, description="Number of top documents to retrieve."
    )
    QUERY_MODE: str = Field(default="default", description="Mode for querying.")
    RETRIEVER_CONFIDENCE_THRESHOLD: float = Field(
        default=0.8, description="Confidence threshold for retriever."
    )

    # Language model configuration
    FASTTEXT_MODEL: str = Field(
        default="lid.176.ftz", description="Path to the FastText model file."
    )
    FASTTEXT_LANGUAGES_MAP: Dict[str, str] = Field(
        default={
            "af": "Afrikaans",
            "als": "Alemannic German",
            "am": "Amharic",
            "an": "Aragonese",
            "ar": "Arabic",
            "arz": "Egyptian Arabic",
            "as": "Assamese",
            "ast": "Asturian",
            "av": "Avaric",
            "az": "Azerbaijani",
            "azb": "South Azerbaijani",
            "ba": "Bashkir",
            "bar": "Bavarian",
            "bcl": "Central Bikol",
            "be": "Belarusian",
            "bg": "Bulgarian",
            "bh": "Bihari languages",
            "bn": "Bengali",
            "bo": "Tibetan",
            "bpy": "Bishnupriya Manipuri",
            "br": "Breton",
            "bs": "Bosnian",
            "bxr": "Buriat",
            "ca": "Catalan",
            "cbk": "Chavacano (Zamboanga)",
            "ce": "Chechen",
            "ceb": "Cebuano",
            "ckb": "Central Kurdish (Sorani)",
            "co": "Corsican",
            "cs": "Czech",
            "cv": "Chuvash",
            "cy": "Welsh",
            "da": "Danish",
            "de": "German",
            "diq": "Dimli (Zaza)",
            "dsb": "Lower Sorbian",
            "dty": "Doteli",
            "dv": "Dhivehi",
            "el": "Greek",
            "eml": "Emilian-Romagnol",
            "en": "English",
            "eo": "Esperanto",
            "es": "Spanish",
            "et": "Estonian",
            "eu": "Basque",
            "fa": "Persian",
            "fi": "Finnish",
            "fr": "French",
            "frr": "Northern Frisian",
            "fy": "Western Frisian",
            "ga": "Irish",
            "gd": "Scottish Gaelic",
            "gl": "Galician",
            "gn": "Guarani",
            "gom": "Goan Konkani",
            "gu": "Gujarati",
            "gv": "Manx",
            "he": "Hebrew",
            "hi": "Hindi",
            "hif": "Fiji Hindi",
            "hr": "Croatian",
            "hsb": "Upper Sorbian",
            "ht": "Haitian Creole",
            "hu": "Hungarian",
            "hy": "Armenian",
            "ia": "Interlingua",
            "id": "Indonesian",
            "ie": "Interlingue",
            "ilo": "Ilocano",
            "io": "Ido",
            "is": "Icelandic",
            "it": "Italian",
            "ja": "Japanese",
            "jbo": "Lojban",
            "jv": "Javanese",
            "ka": "Georgian",
            "kk": "Kazakh",
            "km": "Khmer",
            "kn": "Kannada",
            "ko": "Korean",
            "krc": "Karachay-Balkar",
            "ku": "Kurdish",
            "kv": "Komi",
            "kw": "Cornish",
            "ky": "Kyrgyz",
            "la": "Latin",
            "lb": "Luxembourgish",
            "lez": "Lezghian",
            "li": "Limburgish",
            "lmo": "Lombard",
            "lo": "Lao",
            "lrc": "Northern Luri",
            "lt": "Lithuanian",
            "lv": "Latvian",
            "mai": "Maithili",
            "mg": "Malagasy",
            "mhr": "Eastern Mari",
            "min": "Minangkabau",
            "mk": "Macedonian",
            "ml": "Malayalam",
            "mn": "Mongolian",
            "mr": "Marathi",
            "mrj": "Western Mari",
            "ms": "Malay",
            "mt": "Maltese",
            "mwl": "Mirandese",
            "my": "Burmese",
            "myv": "Erzya",
            "mzn": "Mazanderani",
            "nah": "Nahuatl",
            "nap": "Neapolitan",
            "nds": "Low German",
            "ne": "Nepali",
            "new": "Newar (Newari)",
            "nl": "Dutch",
            "nn": "Norwegian Nynorsk",
            "no": "Norwegian",
            "oc": "Occitan",
            "or": "Oriya (Odia)",
            "os": "Ossetian",
            "pa": "Punjabi",
            "pam": "Pampanga",
            "pfl": "Palatine German",
            "pl": "Polish",
            "pms": "Piedmontese",
            "pnb": "Western Punjabi",
            "ps": "Pashto",
            "pt": "Portuguese",
            "qu": "Quechua",
            "rm": "Romansh",
            "ro": "Romanian",
            "ru": "Russian",
            "rue": "Rusyn",
            "sa": "Sanskrit",
            "sah": "Yakut",
            "sc": "Sardinian",
            "scn": "Sicilian",
            "sco": "Scots",
            "sd": "Sindhi",
            "sh": "Serbo-Croatian",
            "si": "Sinhala",
            "sk": "Slovak",
            "sl": "Slovenian",
            "so": "Somali",
            "sq": "Albanian",
            "sr": "Serbian",
            "su": "Sundanese",
            "sv": "Swedish",
            "sw": "Swahili",
            "ta": "Tamil",
            "te": "Telugu",
            "tg": "Tajik",
            "th": "Thai",
            "tk": "Turkmen",
            "tl": "Tagalog",
            "tr": "Turkish",
            "tt": "Tatar",
            "tyv": "Tuvinian",
            "ug": "Uyghur",
            "uk": "Ukrainian",
            "ur": "Urdu",
            "uz": "Uzbek",
            "vec": "Venetian",
            "vep": "Veps",
            "vi": "Vietnamese",
            "vls": "West Flemish",
            "vo": "Volapük",
            "wa": "Walloon",
            "war": "Waray",
            "wuu": "Wu Chinese",
            "xal": "Kalmyk",
            "xmf": "Mingrelian",
            "yi": "Yiddish",
            "yo": "Yoruba",
            "yue": "Cantonese",
            "zh": "Chinese",
        }
    )


settings = Settings()
