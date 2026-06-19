# tests/test_agents.py
import unittest
from unittest.mock import MagicMock, patch
import json
from src.agents import ExtractionAgent, SafetyAuditorAgent, ClinicalWriterAgent
from src.models import DischargeSummaryDraft, ClinicalFlag, MedicationItem

class TestAgents(unittest.TestCase):
    def setUp(self):
        # Sample structured draft
        self.draft_data = {
            "patient_name": "Test Patient",
            "medical_record_number": "MRN12345",
            "age_and_gender": "40y/M",
            "admission_date": "01/01/2026",
            "discharge_date": "05/01/2026",
            "principal_diagnosis": "Acute Bronchitis",
            "secondary_diagnoses": [],
            "hospital_course": "Treated with supportive care.",
            "procedures_performed": [],
            "discharge_medications": [
                {
                    "name": "Amoxicillin",
                    "dosage": "500mg",
                    "frequency": "TID",
                    "status": "ADDED",
                    "reconciliation_note": "New prescription."
                }
            ],
            "allergies": ["None"],
            "follow_up_instructions": "Follow up with PCP.",
            "pending_results": [],
            "discharge_condition": "Stable",
            "clinical_safety_flags": []
        }
        self.draft = DischargeSummaryDraft.model_validate(self.draft_data)

    def test_extraction_agent(self):
        # Mock LLM response for ExtractionAgent
        mock_call_llm = MagicMock(return_value={
            "status": "SUCCESS",
            "content": json.dumps(self.draft_data)
        })
        
        agent = ExtractionAgent(mock_call_llm)
        result = agent.extract("raw clinical notes text", {"provider": "mock"})
        
        self.assertEqual(result.patient_name, "Test Patient")
        self.assertEqual(result.principal_diagnosis, "Acute Bronchitis")
        mock_call_llm.assert_called_once()

    def test_safety_auditor_agent(self):
        mock_call_llm = MagicMock(return_value={
            "status": "SUCCESS",
            "content": json.dumps([
                {
                    "category": "PENDING_RESULT_WARNING",
                    "item_involved": "Blood Culture",
                    "description": "Blood culture was sent, report is pending.",
                    "action_taken": "Marked pending in draft."
                }
            ])
        })
        
        mock_execute_tool = MagicMock(return_value="Blood culture pending details.")
        
        agent = SafetyAuditorAgent(mock_call_llm, mock_execute_tool)
        flags, traces = agent.audit("patient_id", "raw notes text", self.draft, {"provider": "mock"})
        
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].category, "PENDING_RESULT_WARNING")
        self.assertEqual(flags[0].item_involved, "Blood Culture")
        
        # Tools should be executed
        mock_execute_tool.assert_any_call("patient_id", "raw notes text", "MedicationReconciliation", "")
        mock_execute_tool.assert_any_call("patient_id", "raw notes text", "PendingResultsCheck", "")
        mock_execute_tool.assert_any_call("patient_id", "raw notes text", "DiagnosticCheck", "")

    def test_clinical_writer_agent(self):
        # Writer output incorporates feedback rules and safety flags
        edited_draft_data = dict(self.draft_data)
        edited_draft_data["principal_diagnosis"] += " [Clinically Verified]"
        edited_draft_data["clinical_safety_flags"] = [
            {
                "category": "MEDICATION_MISMATCH",
                "item_involved": "Thyroid Medication",
                "description": "Omission discrepancy",
                "action_taken": "Flagged"
            }
        ]
        
        mock_call_llm = MagicMock(return_value={
            "status": "SUCCESS",
            "content": json.dumps(edited_draft_data)
        })
        
        agent = ClinicalWriterAgent(mock_call_llm)
        safety_flags = [ClinicalFlag.model_validate(edited_draft_data["clinical_safety_flags"][0])]
        feedback_memory = ["For principal_diagnosis: Always append ' [Clinically Verified]'"]
        
        result = agent.compile("raw notes", self.draft, safety_flags, feedback_memory, {"provider": "mock"})
        
        self.assertEqual(result.principal_diagnosis, "Acute Bronchitis [Clinically Verified]")
        self.assertEqual(len(result.clinical_safety_flags), 1)
        self.assertEqual(result.clinical_safety_flags[0].category, "MEDICATION_MISMATCH")

if __name__ == "__main__":
    unittest.main()
