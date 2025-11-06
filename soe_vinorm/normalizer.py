from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Union

from tqdm import tqdm
from bs4 import BeautifulSoup
from markdown import markdown
import re

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

    def preprocess(self, text: str) -> str:
        pattern = r'\s*\[citation:\d+\]'
        text = markdown(text)
        text = ''.join(BeautifulSoup(text).find_all(string=True))
        text =  text.replace('\n', ' ').replace('  ', ' ')
        text = re.sub(pattern, '', text)
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
