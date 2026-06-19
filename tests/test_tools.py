# tests/test_tools.py
import unittest
from src.agent_loop import ClinicalAgentLoop

class TestTools(unittest.TestCase):
    def setUp(self):
        self.agent = ClinicalAgentLoop()
        self.raw_clinical_text = (
            "PATIENT RECORD\n"
            "Past medical history: K/C/O Thyroid disorder on Tab. Thyronorm 50mcg daily.\n"
            "Admission note: Outpatient home medications include Metformin 500mg BID.\n"
            "Investigations:\n"
            "Serum creatinine was elevated at 1.65 mg/dL on admission. Repeat creatinine was 1.17 mg/dL.\n"
            "Blood and urine cultures drawn on 27/02/2026 - reports awaited at discharge.\n"
            "Advice on discharge:\n"
            "1. Tab. Raciper 40mg daily\n"
            "2. Tab. Emeset 4mg daily\n"
            "Follow-up: Review on 09.03.2026."
        )

    def test_medication_reconciliation_offline(self):
        # When running offline/local, it parses matching lines
        result = self.agent._execute_tool("Patient X", self.raw_clinical_text, "MedicationReconciliation", "")
        self.assertIn("Admission medications", result)
        self.assertIn("Discharge medications", result)
        self.assertIn("Tab. Thyronorm", result)
        self.assertIn("Tab. Raciper", result)
        self.assertIn("Discrepancy", result)

    def test_pending_results_check_offline(self):
        result = self.agent._execute_tool("Patient X", self.raw_clinical_text, "PendingResultsCheck", "")
        self.assertIn("culture", result.lower())
        self.assertIn("awaited", result.lower())

    def test_diagnostic_check_offline(self):
        result = self.agent._execute_tool("Patient X", self.raw_clinical_text, "DiagnosticCheck", "")
        self.assertIn("creatinine", result.lower())

if __name__ == "__main__":
    unittest.main()
