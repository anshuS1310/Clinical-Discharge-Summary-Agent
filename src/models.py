# src/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ==========================================
# 1. CLINICAL SAFETY & RECONCILIATION MODELS
# ==========================================

class ClinicalFlag(BaseModel):
    category: Literal["MISSING_DATA", "MEDICATION_MISMATCH", "CONFLICTING_DIAGNOSES", "PENDING_RESULT_WARNING"] = Field(
        ..., description="The classification of the clinical safety or verification anomaly."
    )
    item_involved: str = Field(..., description="The specific medication, diagnosis, or laboratory test name.")
    description: str = Field(..., description="Detailed explanation of the discrepancy or what missing item requires manual review.")
    action_taken: str = Field(..., description="How the agent handled it (e.g., 'Flagged for clinician override', 'Set field status to pending').")

class MedicationItem(BaseModel):
    name: str = Field(..., description="Brand or generic name of the medication.")
    dosage: str = Field(..., description="Dosage details (e.g., 500mg, 5ml).")
    frequency: str = Field(..., description="How often taken (e.g., BID, Daily, PRN).")
    status: Literal["UNCHANGED", "ADDED", "DISCONTINUED", "DOSAGE_CHANGED"] = Field(
        ..., description="The reconciliation status compared to admission medications."
    )
    reconciliation_note: str = Field(
        "No issues noted.", 
        description="Must document the clinical reason for change, or flag as 'UNEXPLAINED DISCREPANCY' if undocumented."
    )

# ==========================================
# 2. THE 10 REQUIRED DISCHARGE SECTIONS
# ==========================================

class DischargeSummaryDraft(BaseModel):
    # Section 1: Patient Demographics
    patient_name: str = Field(default="missing", description="Full name of the patient. Defaults to 'missing' if absent.")
    medical_record_number: str = Field(default="missing", description="MRN or Registration ID.")
    age_and_gender: str = Field(default="missing", description="Age and gender details.")
    
    # Section 2: Dates
    admission_date: str = Field(default="missing", description="Date of hospital admission.")
    discharge_date: str = Field(default="missing", description="Date of hospital discharge.")
    
    # Section 3: Diagnoses
    principal_diagnosis: str = Field(..., description="The primary reason for hospital admission.")
    secondary_diagnoses: List[str] = Field(default_factory=list, description="Associated or secondary clinical conditions.")
    
    # Section 4: Hospital Course
    hospital_course: str = Field(..., description="Chronological and detailed narrative summarizing the patient's hospital stay.")
    
    # Section 5: Procedures
    procedures_performed: List[str] = Field(default_factory=list, description="Surgical or diagnostic procedures performed.")
    
    # Section 6: Discharge Medications
    discharge_medications: List[MedicationItem] = Field(..., description="Full reconciled list of discharge medications.")
    
    # Section 7: Allergies
    allergies: List[str] = Field(default_factory=list, description="Known drug, food, or environmental allergies. Explicitly state 'No known allergies' if noted.")
    
    # Section 8: Follow-up Instructions
    follow_up_instructions: str = Field(..., description="Actionable details regarding outpatient appointments, timelines, and clinician roles.")
    
    # Section 9: Pending Results
    pending_results: List[str] = Field(
        default_factory=list, 
        description="Lab values, imaging reports, or cultures that were not finalized before discharge. Explicitly mark as 'pending'."
    )
    
    # Section 10: Discharge Condition
    discharge_condition: str = Field(..., description="The clinical status of the patient at the precise time of discharge (e.g., Stable, Guarded).")
    
    # System & Safety Overhead
    clinical_safety_flags: List[ClinicalFlag] = Field(
        default_factory=list, 
        description="A mandatory collection of all clinical safety, conflict, or validation flags raised during draft formulation."
    )

# ==========================================
# 3. AGENT OBSERVABILITY & TRACE SCHEMA
# ==========================================

class AgentStepTrace(BaseModel):
    step_number: int
    reasoning: str = Field(..., description="The internal clinical/operational logic explaining why this action is selected.")
    action_chosen: str = Field(..., description="The tool or internal parsing routine invoked.")
    inputs: str = Field(..., description="Raw inputs transferred to the tool or processing function.")
    result: str = Field(..., description="The output text or response payload returned by the tool/action.")
    next_decision: str = Field(..., description="Evaluation of the result and determining the immediate subsequent objective.")

class CompleteExecutionPayload(BaseModel):
    patient_id: str
    final_draft: DischargeSummaryDraft
    execution_trace: List[AgentStepTrace]
    total_steps_executed: int
    loop_status: Literal["COMPLETED_SUCCESSFULLY", "PARTIAL_TIMEOUT_FALLBACK"]