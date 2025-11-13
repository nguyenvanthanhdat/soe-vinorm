from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Union

from tqdm import tqdm
from bs4 import BeautifulSoup
from markdown import markdown
import re
import os

from soe_vinorm.nsw_detector import CRFNSWDetector
from soe_vinorm.nsw_expander import RuleBasedNSWExpander
from soe_vinorm.text_processor import TextPreprocessor, ReplaceExpandHotfixMultiWord
from soe_vinorm.utils import load_abbreviation_dict, load_vietnamese_syllables, load_expand_hotfix_dict, load_expand_hotfix_multi_word_dict


class Normalizer(ABC):
    """
    Abstract base class for text normalizers.
    """

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text in spoken form.
        """
        ...

    @abstractmethod
    def batch_normalize(
        self, texts: List[str], n_jobs: int = 1, show_progress: bool = False
    ) -> List[str]:
        """
        Normalize multiple texts efficiently.

        Args:
            texts: List of input texts to normalize.
            n_jobs: Number of jobs to run in parallel.
            show_progress: Whether to show progress bar.

        Returns:
            List of normalized texts.
        """
        ...


def _worker_initializer(
    vn_dict: Union[List[str], None] = None,
    abbr_dict: Union[Dict[str, List[str]], None] = None,
    expand_hotfix_dict: Union[Dict[str, str], None] = None,
    expand_hotfix_multi_word_dict: Union[Dict[str, str], None] = None,
    kwargs: Dict[str, Any] = {},
):
    """Initialize worker instance."""
    global worker_normalizer
    worker_normalizer = SoeNormalizer(vn_dict=vn_dict, abbr_dict=abbr_dict, expand_hotfix_dict=expand_hotfix_dict, expand_hotfix_multi_word_dict=expand_hotfix_multi_word_dict, **kwargs)


def _worker_normalize(text: str) -> str:
    """Normalize text in worker instance."""
    global worker_normalizer
    return worker_normalizer.normalize(text)


class SoeNormalizer(Normalizer):
    """
    Effective Vietnamese text normalizer.
    """

    def __init__(
        self,
        vn_dict: Union[List[str], None] = None,
        abbr_dict: Union[Dict[str, List[str]], None] = None,
        expand_hotfix_dict: Union[Dict[str, str], None] = None,
        expand_hotfix_multi_word_dict: Union[Dict[str, str], None] = None,
        **kwargs,
    ):
        """
        Initialize the effective Vietnamese normalizer.

        Args:
            vn_dict: List of Vietnamese words for dictionary lookup. If None, use default Vietnamese dictionary.
            abbr_dict: Dictionary of abbreviations and their expansions. If None, use default abbreviation dictionary.
            expand_hotfix_dict: Dictionary of hotfix expansions for single words. If None, use default hotfix dictionary.
            expand_hotfix_multi_word_dict: Dictionary of hotfix expansions for multi-word expressions. If None, use default multi-word hotfix dictionary.
        """
        self._vn_dict = vn_dict or load_vietnamese_syllables()
        self._abbr_dict = abbr_dict or load_abbreviation_dict()
        self._expand_hotfix_dict = expand_hotfix_dict or load_expand_hotfix_dict()
        self._expand_hotfix_multi_word_dict = expand_hotfix_multi_word_dict or load_expand_hotfix_multi_word_dict()
        self._kwargs = kwargs

        self._preprocessor = TextPreprocessor(self._vn_dict, **kwargs)
        self._replace_expand_hotfix_multi_word = ReplaceExpandHotfixMultiWord(self._expand_hotfix_multi_word_dict)
        self._nsw_detector = CRFNSWDetector(
            vn_dict=self._vn_dict,
            abbr_dict=self._abbr_dict,
            **kwargs,
        )
        self._nsw_expander = RuleBasedNSWExpander(
            vn_dict=self._vn_dict,
            abbr_dict=self._abbr_dict,
            **kwargs,
        )

        self.UNIT_MAP = {
            "mm²": "mili mét vuông",
            "cm²": "xen ti mét vuông",
            "dm²": "đề xi mét vuông",
            "m²": "mét vuông",
            "dam²": "đề ca mét vuông",
            "hm²": "hec ta mét vuông",
            "km²": "ki lô mét vuông",
        }

        self.VALID_TLDS = [
            "com", "org", "net", "edu", "gov", "mil", "int", "biz", "info", "name", "pro",
            "coop", "museum", "travel", "jobs", "mobi", "tel", "asia", "cat", "xxx",
            "aero", "arpa", "bg", "br", "ca", "cn", "de", "es", "eu", "fr", "in", "it",
            "jp", "mx", "nl", "ru", "uk", "us", "io", "co"
        ]

        self.MULTIPART_TLDS = [
            "com.vn", "gov.vn", "org.vn", "edu.vn", "net.vn", "biz.vn",
            "co.uk", "ac.uk", "gov.uk",
        ]

    def _replace_tlds(self, text: str) -> str:
        if getattr(self, "MULTIPART_TLDS", None):
            mp_pat = "|".join(re.escape(t) for t in sorted(self.MULTIPART_TLDS, key=len, reverse=True))
            pattern_mp = re.compile(
                rf'(?i)\b(?P<label>(?=[a-z-]*[a-z])[a-z0-9-]{{1,63}})\.(?P<mp>{mp_pat})(?![a-z0-9-])'
            )
            text = pattern_mp.sub(
                lambda m: f"{m.group('label')} chấm {m.group('mp').replace('.', ' chấm ')}",
                text
            )

        if getattr(self, "VALID_TLDS", None):
            sp_pat = "|".join(re.escape(t) for t in self.VALID_TLDS)
            pattern_sp = re.compile(
                rf'(?i)\b(?P<label>(?=[a-z-]*[a-z])[a-z0-9-]{{1,63}})\.(?P<tld>{sp_pat})(?![a-z0-9-])'
            )
            text = pattern_sp.sub(
                lambda m: f"{m.group('label')} chấm {m.group('tld')}",
                text
            )

        return text
    
    def _clean_punctuation(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)

        def replace_group(m):
            group = m.group()
            puncts = re.findall(r'[.,]', group)
            return '.' if '.' in puncts else ','

        text = re.sub(r'[.,](?:\s*[.,])+', replace_group, text)

        return text.strip()

    def _remove_emojis(self, text: str) -> str:
        emoji_pattern = re.compile(
            "[" 
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    def preprocess(self, text: str) -> str:
        # remove citations and markdown formatting
        text = re.sub(r'\s*\[citation:\d+\]', '', text)
        text = re.sub(r'(?m)^(\s*)(\d+)\.\s+', r'\1\2\. ', text)

        # convert markdown to plain text
        text = markdown(text)
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator="\n")

        # concatenate multiple newlines
        text = re.sub(r'\n+', '\n', text).strip()

        # convert 1. -> 1, 2. -> 2, at the beginning of lines
        text = re.sub(r'(?m)^\s*(\d{1,3})\.\s+', r'\1, ', text)

        # Replace \n to "." for add [pause] time
        # text = text.replace('\n', '. ')
        text = re.sub(r'(?<![:;(\[])\n+', '. ', text)

        # handle URL
        text = self._replace_tlds(text)

        # replace dashes between words with commas for short pauses
        # text = re.sub(r'(?<!\d)\s*[-–—]\s*(?!\d)', ', ', text)
        text = re.sub(r'(?<![A-Za-z0-9])\s*[-–—]\s*(?![A-Za-z0-9])', ', ', text)

        # normalize spaces around punctuation
        text = re.sub(r"\s+", " ", text)
        
        # Normalize the "." for case "3.7" or "5.1"
        text = re.sub(r'(?<!\d)\s*\.\s*(?!\d)', '. ', text)

        # remove multiple dots
        text = re.sub(r'\.(\s*\.)+', '. ', text)

        # clean up leading/trailing spaces and dots
        text = text.strip()
        text = re.sub(r'\.\s*$', '.', text)

        # remove space before punctuation like " ." or " ,"
        text = re.sub(r'\s+([,.:;!?])', r'\1', text)
        # remove space after opening brackets and before closing brackets
        text = re.sub(r'\s+([)\]\}”’])', r'\1', text)

        # handle measure units
        for unit, expansion in self.UNIT_MAP.items():
            pattern = rf'(\d+)\s*{re.escape(unit)}'
            replacement = rf'\1 {expansion}'
            text = re.sub(pattern, replacement, text) 

        # replace "\"" to "," for short pause
        text = text.replace(' " ', ', ').replace('" ', ', ').replace('"', ', ').replace('“', ', ').replace('”', ', ')

        # final cleanup
        text = self._remove_emojis(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = self._clean_punctuation(text)


        if os.environ.get("SOE_VINORM_DEBUG", "false").lower() == "true" or os.environ.get("SOE_VINORM_DEBUG") == "1":
            print("DEBUG preprocess:", text)

        return text

    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text in spoken form.
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        text = self.preprocess(text)
        tokens = self._preprocessor(text) # .split()
        # print(f"DEBUG preprocessed tokens: {tokens}")
        # print(f"DEBUG tokens: {tokens}")
        # print(f"Info: {self._expand_hotfix_dict}")
        # add hotfix replacements before NSW detection
        # replace multi-word first
        # for phrase, expansion in self._expand_hotfix_multi_word_dict.items():
        #     if phrase in tokens:
        #         tokens = tokens.replace(phrase, expansion)
        tokens = self._replace_expand_hotfix_multi_word(tokens)
        # then single-word
        tokens = tokens.split()
        for token_idx, token in enumerate(tokens):
            if token in self._expand_hotfix_dict:
                tokens[token_idx] = self._expand_hotfix_dict[token]
        if not tokens:
            return text.strip()
        


        nsw_tags = self._nsw_detector.detect(tokens)
        expanded_tokens = self._nsw_expander.expand(tokens, nsw_tags)

        return " ".join(expanded_tokens)

    def batch_normalize(
        self, texts: List[str], n_jobs: int = 1, show_progress: bool = False
    ) -> List[str]:
        """
        Normalize multiple texts efficiently.

        Args:
            texts: List of input texts to normalize.
            n_jobs: Number of jobs to run in parallel.
            show_progress: Whether to show progress bar.

        Returns:
            List of normalized texts.
        """
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise TypeError("Input must be a list of strings")

        if n_jobs <= 0:
            raise ValueError("Number of jobs must be greater than 0")

        if n_jobs == 1:
            return [
                self.normalize(text)
                for text in tqdm(
                    texts,
                    desc="Normalizing texts",
                    total=len(texts),
                    disable=not show_progress,
                )
            ]

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_worker_initializer,
            initargs=(self._vn_dict, self._abbr_dict, self._expand_hotfix_dict, self._expand_hotfix_multi_word_dict, self._kwargs),
        ) as executor:
            return list(
                tqdm(
                    executor.map(_worker_normalize, texts),
                    desc="Normalizing texts",
                    total=len(texts),
                    disable=not show_progress,
                )
            )


def normalize_text(
    text: str,
    vn_dict: Union[List[str], None] = None,
    abbr_dict: Union[Dict[str, List[str]], None] = None,
    expand_hotfix_dict: Union[Dict[str, str], None] = None,
    expand_hotfix_multi_word_dict: Union[Dict[str, str], None] = None,
    **kwargs,
) -> str:
    """
    Quick normalization function.

    Args:
        text: Input text to normalize.
        vn_dict: Optional Vietnamese dictionary. If None, use default Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary. If None, use default abbreviation dictionary.

    Returns:
        Normalized text.
    """
    normalizer = SoeNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        expand_hotfix_dict=expand_hotfix_dict,
        expand_hotfix_multi_word_dict=expand_hotfix_multi_word_dict,
        **kwargs,
    )
    return normalizer.normalize(text)


def batch_normalize_texts(
    texts: List[str],
    vn_dict: Union[List[str], None] = None,
    abbr_dict: Union[Dict[str, List[str]], None] = None,
    expand_hotfix_dict: Union[Dict[str, str], None] = None,
    expand_hotfix_multi_word_dict: Union[Dict[str, str], None] = None,
    n_jobs: int = 1,
    show_progress: bool = False,
    **kwargs,
) -> List[str]:
    """
    Batch normalization function.

    Args:
        texts: List of input texts to normalize.
        vn_dict: Optional Vietnamese dictionary. If None, use default Vietnamese dictionary.
        abbr_dict: Optional abbreviation dictionary. If None, use default abbreviation dictionary.
        n_jobs: Number of jobs to run in parallel.
        show_progress: Whether to show progress bar.

    Returns:
        List of normalized texts.
    """
    normalizer = SoeNormalizer(
        vn_dict=vn_dict,
        abbr_dict=abbr_dict,
        expand_hotfix_dict=expand_hotfix_dict,
        expand_hotfix_multi_word_dict=expand_hotfix_multi_word_dict,
        **kwargs,
    )
    return normalizer.batch_normalize(texts, n_jobs=n_jobs, show_progress=show_progress)
