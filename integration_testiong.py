# In test_integration.py

import unittest
from your_code.tokenizer_stemmer_module import RegexpTokenizer, SnowballStemmer
from your_code.count_vectorizer_module import CountVectorizer
from your_code.model_module import LogisticRegression, MultinomialNB
from your_code.fastapi_endpoint_module import your_fastapi_endpoint_function

class TestIntegration(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        # Create instances of components
        tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        stemmer = SnowballStemmer("english")
        vectorizer = CountVectorizer()
        lr_model = LogisticRegression()

        # Simulate a URL
        url = "http://example.com"

        # Tokenization and stemming
        tokenized_text = tokenizer.tokenize(url)
        stemmed_text = [stemmer.stem(word) for word in tokenized_text]

        # Feature extraction
        feature_matrix = vectorizer.transform([' '.join(stemmed_text)])

        # Model prediction
        prediction = lr_model.predict(feature_matrix)

        # FastAPI endpoint integration
        endpoint_result = your_fastapi_endpoint_function(url)

        # Assertions
        self.assertEqual(prediction[0], endpoint_result['prediction'])

if __name__ == '__main__':
    unittest.main()
