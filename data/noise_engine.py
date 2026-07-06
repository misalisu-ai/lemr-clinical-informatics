# data/noise_engine.py
import random
import torch

class ClinicalNoiseEngine:
    """
    LEMR Robustness Engine: Simulates typographical entry mutations 
    and element-wise clinical missingness patterns.
    """
    def __init__(self, char_prob: float = 0.05):
        self.char_prob = char_prob
        self.qwerty_map = {
            'a': ['s', 'q', 'z'], 's': ['a', 'd', 'w', 'x'],
            'd': ['s', 'f', 'e', 'c'], 'f': ['d', 'g', 'r', 'v'],
            'g': ['f', 'h', 't', 'b'], 'h': ['g', 'j', 'y', 'n'],
            'm': ['n', 'k', 'j'], 'i': ['u', 'o', 'k', 'j'],
            'o': ['i', 'p', 'l', 'k']
        }

    def inject_keyboard_typos(self, text: str) -> str:
        """Injects localized keyboard substitutions to simulate charting rush."""
        words = text.split()
        perturbed_words = []
        for word in words:
            word_chars = list(word)
            for idx, char in enumerate(word_chars):
                char_lower = char.lower()
                if char_lower in self.qwerty_map and random.random() < self.char_prob:
                    replacement = random.choice(self.qwerty_map[char_lower])
                    word_chars[idx] = replacement.upper() if char.isupper() else replacement
            perturbed_words.append("".join(word_chars))
        return " ".join(perturbed_words)

    @staticmethod
    def apply_hadamard_mask(vitals_tensor: torch.Tensor, drop_prob: float = 0.3) -> torch.Tensor:
        """Simulates clinical channel dropping using element-wise binary masks."""
        mask = (torch.rand_like(vitals_tensor) > drop_prob).float()
        return vitals_tensor * mask