import unittest
from models.cnn_model import build_cnn_model

class TestModel(unittest.TestCase):
    def test_model_creation(self):
        model = build_cnn_model()
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
