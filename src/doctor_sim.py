# src/doctor_sim.py
from src.models import DischargeSummaryDraft

class DoctorSimulator:
    """
    Simulates a clinician reviewer applying a consistent, hidden editing policy
    to the agent's generated draft discharge summaries.
    """
    def __init__(self):
        # Hidden policy preference: The doctor prefers a specific format for instructions 
        # and wants explicit prefixes added to diagnoses for insurance tracking.
        self.preferred_suffix = " [Clinically Verified via Discharge Evaluation Policy]"

    def apply_hidden_doctor_policy(self, draft: DischargeSummaryDraft) -> DischargeSummaryDraft:
        """
        Ingests an agent's draft and returns an edited version reflecting the 
        doctor's strict preferences.
        """
        # Create a deep copy clone of the draft to simulate manual edits
        edited_draft = draft.model_copy(deep=True)
        
        # Policy Adjustment 1: Standardizing principal diagnosis format
        if edited_draft.principal_diagnosis and not edited_draft.principal_diagnosis.endswith(self.preferred_suffix):
            edited_draft.principal_diagnosis += self.preferred_suffix
            
        # Policy Adjustment 2: Enforcing style constraints on follow-up instructions
        if "Please visit the clinic" not in edited_draft.follow_up_instructions:
            edited_draft.follow_up_instructions = (
                "CRITICAL CLINICAL FOLLOW-UP: Please visit the clinic as scheduled. " 
                + edited_draft.follow_up_instructions
            )
            
        return edited_draft