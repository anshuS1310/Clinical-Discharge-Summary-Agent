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

    def test_extractive_local_loop_robustness(self):
        messy_text = (
            "PATIENT RECORD DETAILS\n"
            "Admission No. / Dates IP-02 | Admitted: 15-Nov-2016 – Discharged: 22-Nov-2016\n"
            "Patient Name: Jane Smith\n"
            "Age: 53 Years\n"
            "Sex: Female\n"
            "DIAGNOSIS:\n"
            "Diabetic ketoacidosis; Urinary tract infection\n"
            "ADVICE ON DISCHARGE:\n"
            "Tab. Thyronorm 50mcg once daily\n"
            "Tab. Metformin 500mg twice daily\n"
            "FOLLOW-UP:\n"
            "Review in clinic."
        )
        payload = self.agent._run_extractive_local_loop("Jane Smith", messy_text)
        draft = payload.final_draft
        
        self.assertEqual(draft.patient_name, "Jane Smith")
        self.assertEqual(draft.medical_record_number, "IP-02")
        self.assertIn("53 Years", draft.age_and_gender)
        self.assertIn("Female", draft.age_and_gender)
        self.assertEqual(draft.admission_date, "15-Nov-2016")
        self.assertEqual(draft.discharge_date, "22-Nov-2016")
        self.assertEqual(draft.principal_diagnosis, "Diabetic ketoacidosis")
        self.assertEqual(len(draft.discharge_medications), 2)
        self.assertEqual(draft.discharge_medications[0].name, "Tab. Thyronorm")
        self.assertEqual(draft.discharge_medications[0].dosage, "50mcg")
        self.assertEqual(draft.discharge_medications[0].frequency, "once daily")

if __name__ == "__main__":
    unittest.main()
