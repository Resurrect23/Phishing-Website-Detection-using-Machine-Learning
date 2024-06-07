# In test_tokenizer_stemmer.py

import unittest
from your_code.tokenizer_stemmer_module import RegexpTokenizer, SnowballStemmer

class TestTokenizerStemmer(unittest.TestCase):
    def test_tokenizer(self):
        tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        result = tokenizer.tokenize("http://example.com")
        self.assertEqual(result, ["http", "example", "com"])

    def test_stemmer(self):
        stemmer = SnowballStemmer("english")
        result = stemmer.stem("phishing")
        self.assertEqual(result, "phish")

if __name__ == '__main__':
    unittest.main()
