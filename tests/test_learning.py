# tests/test_learning.py
import unittest
from unittest.mock import MagicMock
import json
from src.learning_engine import FeedbackLearningEngine
from src.models import DischargeSummaryDraft

class TestLearningEngine(unittest.TestCase):
    def setUp(self):
        self.engine = FeedbackLearningEngine()

    def test_calculate_normalized_edit_distance(self):
        str1 = "Acute Bronchitis"
        str2 = "Acute Bronchitis"
        # Perfect match
        self.assertEqual(self.engine.calculate_normalized_edit_distance(str1, str2), 0.0)
        
        # Completely different
        str3 = ""
        self.assertEqual(self.engine.calculate_normalized_edit_distance(str1, str3), 1.0)
        
        # Partial match
        str4 = "Acute Bronchitis [Clinically Verified]"
        dist = self.engine.calculate_normalized_edit_distance(str1, str4)
        self.assertTrue(0.0 < dist < 1.0)

    def test_extract_feedback_rules_offline(self):
        # Create a mock draft and mock doctor-edited draft
        draft = MagicMock()
        draft.principal_diagnosis = "Acute Bronchitis"
        draft.follow_up_instructions = "Follow up with PCP."
        
        edited = MagicMock()
        edited.principal_diagnosis = "Acute Bronchitis [Clinically Verified]"
        edited.follow_up_instructions = "ATTENTION PATIENT: Follow up with PCP."
        
        # Test offline rule extraction (no live API key)
        new_rules = self.engine.extract_feedback_rules(draft, edited)
        
        self.assertEqual(len(new_rules), 2)
        self.assertIn("For principal_diagnosis: Always append ' [Clinically Verified]'", self.engine.correction_memory)
        self.assertIn("For follow_up_instructions: Always prepend 'ATTENTION PATIENT: '", self.engine.correction_memory)

    def test_register_performance(self):
        self.engine.register_iteration_performance("Patient X", "Diagnosis", "Diagnosis [Verified]")
        self.assertIn("Patient X", self.engine.performance_history)
        self.assertEqual(len(self.engine.performance_history["Patient X"]), 1)
        self.assertTrue(0.0 < self.engine.performance_history["Patient X"][0] < 1.0)

if __name__ == "__main__":
    unittest.main()
