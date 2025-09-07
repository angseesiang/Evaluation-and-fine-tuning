import unittest
from text_evaluator import TextEvaluator

class TestTextEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = TextEvaluator("gpt2")

    def test_evaluate(self):
        text = "This is a test sentence."
        loss = self.evaluator.evaluate(text)
        self.assertIsInstance(loss, float)

    def test_save_model(self):
        self.evaluator.save_model("model/test_model_save")
        # Check if the model and tokenizer files exist
        import os
        self.assertTrue(os.path.exists("model/test_model_save"))
        self.assertTrue(os.path.exists("model/test_model_save/tokenizer.json"))

if __name__ == "__main__":
    unittest.main()

