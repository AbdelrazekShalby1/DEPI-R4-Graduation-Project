"""Regression checks for the artifact layout shipped in this repository.

Run from the repository root: python -m unittest discover -s tests -v
No external API calls or medical accuracy claims are part of these checks.
"""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("medical_app_under_test", ROOT / "DEPI" / "app.py")
medical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(medical)


class RepositoryModelPathsTests(unittest.TestCase):
    def test_all_ml_artifacts_resolve_to_existing_files(self):
        for name, model in medical.ML_MODELS.items():
            for attribute in ("model_path", "scaler_path", "feature_encoders_path", "target_encoder_path"):
                path = getattr(model, attribute)
                if path is not None:
                    with self.subTest(model=name, artifact=attribute):
                        self.assertTrue(path.is_file(), str(path))

    def test_packaged_cv_classifiers_resolve(self):
        for path in (
            medical.MODELS_DIR / "x-ray" / "pneumonia_model.keras",
            medical.MODELS_DIR / "Brain_tumor classification" / "brain_tumor_mobilenet.keras",
        ):
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file(), str(path))

    def test_chatbot_source_resolves(self):
        self.assertTrue((medical.CHATBOT_DIR / "chatbot_engine.py").is_file())

    def test_kidney_prediction_loads_packaged_model(self):
        # The application accepts an empty payload and uses scaler defaults.
        # This exercises real artifact loading and inference, not a mock model.
        with medical.app.test_client() as client:
            response = client.post("/api/ml/kidney/predict", json={})
        self.assertEqual(response.status_code, 200, response.get_json())
        data = response.get_json()
        self.assertIsInstance(data["prediction"], str)
        self.assertTrue(data["prediction"])
        self.assertIn("confidence", data)


if __name__ == "__main__":
    unittest.main()
