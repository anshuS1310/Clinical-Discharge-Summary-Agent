# src/agents.py
import json
import re
from typing import List, Dict, Any, Tuple
from src.models import DischargeSummaryDraft, ClinicalFlag, MedicationItem, AgentStepTrace

class ExtractionAgent:
    """
    Specialized agent for raw clinical text extraction.
    Focuses on parsing unstructured patient records and mapping them to structured schema fields
    without fabricating any clinical facts. Gaps are explicitly marked 'missing' or 'pending'.
    """
    def __init__(self, call_llm_fn):
        self.call_llm_fn = call_llm_fn

    def extract(self, raw_clinical_text: str, cfg: dict) -> DischargeSummaryDraft:
        prompt = (
            "You are a clinical extraction agent. Your job is to extract raw patient facts from "
            "the provided patient clinical notes and organize them into the schema below.\n\n"
            "CRITICAL SAFETY RULE:\n"
            "Never guess, invent, or extrapolate any clinical facts. If a field is not explicitly "
            "documented in the notes, set it to 'missing' or 'pending'. "
            "Do not fill in plausible values.\n\n"
            "RAW PATIENT CLINICAL NOTES:\n"
            f"{raw_clinical_text}\n\n"
            "Return the extracted information in a raw JSON block matching this exact schema (no backticks, no other text):\n"
            "{\n"
            "  \"patient_name\": \"string\",\n"
            "  \"medical_record_number\": \"string\",\n"
            "  \"age_and_gender\": \"string\",\n"
            "  \"admission_date\": \"string\",\n"
            "  \"discharge_date\": \"string\",\n"
            "  \"principal_diagnosis\": \"string\",\n"
            "  \"secondary_diagnoses\": [\"string\"],\n"
            "  \"hospital_course\": \"string\",\n"
            "  \"procedures_performed\": [\"string\"],\n"
            "  \"discharge_medications\": [\n"
            "    {\n"
            "      \"name\": \"string\",\n"
            "      \"dosage\": \"string\",\n"
            "      \"frequency\": \"string\",\n"
            "      \"status\": \"UNCHANGED\" | \"ADDED\" | \"DISCONTINUED\" | \"DOSAGE_CHANGED\",\n"
            "      \"reconciliation_note\": \"string\"\n"
            "    }\n"
            "  ],\n"
            "  \"allergies\": [\"string\"],\n"
            "  \"follow_up_instructions\": \"string\",\n"
            "  \"pending_results\": [\"string\"],\n"
            "  \"discharge_condition\": \"string\"\n"
            "}"
        )
        
        res = self.call_llm_fn(prompt, cfg)
        if res["status"] != "SUCCESS":
            raise RuntimeError(f"ExtractionAgent LLM call failed: {res.get('error')}")
            
        content = res["content"].strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"ExtractionAgent failed to locate valid JSON: {content}")
            
        json_str = content[start_idx:end_idx + 1]
        # Validate using Pydantic model by adding empty clinical_safety_flags (will be populated by Auditor)
        data = json.loads(json_str)
        data["clinical_safety_flags"] = []
        return DischargeSummaryDraft.model_validate(data)


class SafetyAuditorAgent:
    """
    Safety-first auditor agent. Reconciles medications, checks diagnostic trends,
    and reviews clinical consistency to identify and raise safety flags for review.
    Uses tools to audit patient clinical records.
    """
    def __init__(self, call_llm_fn, execute_tool_fn):
        self.call_llm_fn = call_llm_fn
        self.execute_tool = execute_tool_fn
        self.active_flags: List[ClinicalFlag] = []

    def audit(self, patient_id: str, raw_clinical_text: str, draft: DischargeSummaryDraft, cfg: dict) -> Tuple[List[ClinicalFlag], List[AgentStepTrace]]:
        traces: List[AgentStepTrace] = []
        self.active_flags = []
        
        # Step 1: Medication Reconciliation
        reason_med = "Checking admission medications against discharge medications to detect omissions or anomalies."
        med_result = self.execute_tool(patient_id, raw_clinical_text, "MedicationReconciliation", "")
        traces.append(AgentStepTrace(
            step_number=1,
            reasoning=reason_med,
            action_chosen="CALL_TOOL: MedicationReconciliation",
            inputs="Patient raw clinical text & draft discharge medications",
            result=med_result,
            next_decision="check pending results"
        ))
        
        # Step 2: Pending Results Check
        reason_pending = "Scanning raw record for outstanding culture, imaging, or laboratory results that must be monitored."
        pending_result = self.execute_tool(patient_id, raw_clinical_text, "PendingResultsCheck", "")
        traces.append(AgentStepTrace(
            step_number=2,
            reasoning=reason_pending,
            action_chosen="CALL_TOOL: PendingResultsCheck",
            inputs="Patient raw clinical text",
            result=pending_result,
            next_decision="check diagnostic trends"
        ))

        # Step 3: Diagnostic Check
        reason_diag = "Evaluating critical laboratory values (e.g., creatinine, electrolytes) for trends or unresolved concerns."
        diag_result = self.execute_tool(patient_id, raw_clinical_text, "DiagnosticCheck", "")
        traces.append(AgentStepTrace(
            step_number=3,
            reasoning=reason_diag,
            action_chosen="CALL_TOOL: DiagnosticCheck",
            inputs="Patient raw clinical text & key labs",
            result=diag_result,
            next_decision="evaluate discrepancy flags"
        ))

        # Let the Auditor decide what safety flags to raise based on the tool execution results
        prompt = (
            "You are a clinical Safety Auditor agent. Review the intermediate discharge draft "
            "and the results of the clinical audit checks. Determine if any safety flags "
            "need to be logged. "
            "Specifically, identify:\n"
            "1. Medication mismatches or unexplained additions/stops (MEDICATION_MISMATCH)\n"
            "2. Unresolved outstanding labs or cultures (PENDING_RESULT_WARNING)\n"
            "3. Conflicting information or discharge at request vs clinical advice (CONFLICTING_DIAGNOSES)\n"
            "4. Missing critical patient information (MISSING_DATA)\n\n"
            "DRAFT SUMMARY DATA:\n"
            f"{draft.model_dump_json(indent=2)}\n\n"
            "AUDIT CHECKS RESULTS:\n"
            f"- Medication Reconciliation: {med_result}\n"
            f"- Pending Results Check: {pending_result}\n"
            f"- Diagnostic Trends Check: {diag_result}\n\n"
            "For each discrepancy or concern, generate a safety flag matching the schema below. "
            "If no issues are found, return an empty list.\n"
            "Return the list of flags in a raw JSON block matching this exact schema (no backticks, no other text):\n"
            "[\n"
            "  {\n"
            "    \"category\": \"MISSING_DATA\" | \"MEDICATION_MISMATCH\" | \"CONFLICTING_DIAGNOSES\" | \"PENDING_RESULT_WARNING\",\n"
            "    \"item_involved\": \"string\",\n"
            "    \"description\": \"string\",\n"
            "    \"action_taken\": \"string\"\n"
            "  }\n"
            "]"
        )

        res = self.call_llm_fn(prompt, cfg)
        if res["status"] == "SUCCESS":
            content = res["content"].strip()
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    json_str = content[start_idx:end_idx + 1]
                    flag_dicts = json.loads(json_str)
                    for fd in flag_dicts:
                        flag = ClinicalFlag.model_validate(fd)
                        self.active_flags.append(flag)
                        traces.append(AgentStepTrace(
                            step_number=len(traces) + 1,
                            reasoning=f"Safety Auditor identified a safety issue regarding {flag.item_involved}.",
                            action_chosen="CALL_TOOL: FlagContradiction",
                            inputs=json.dumps(fd),
                            result=f"Successfully registered {flag.category} safety flag for {flag.item_involved}.",
                            next_decision="continue review"
                        ))
                except Exception as exc:
                    # Fallback if parser fails
                    traces.append(AgentStepTrace(
                        step_number=len(traces) + 1,
                        reasoning="Safety Auditor failed to parse LLM flag decisions.",
                        action_chosen="AUDITOR_PARSING_FAILURE",
                        inputs=content,
                        result=str(exc),
                        next_decision="continue review"
                    ))

        return self.active_flags, traces


class ClinicalWriterAgent:
    """
    Synthesizes the final discharge summary draft.
    Applies safety auditor flags and learned rules from clinician feedback (correction memory)
    to format and phrase the final summary correctly.
    """
    def __init__(self, call_llm_fn):
        self.call_llm_fn = call_llm_fn

    def compile(
        self,
        raw_clinical_text: str,
        intermediate_draft: DischargeSummaryDraft,
        safety_flags: List[ClinicalFlag],
        feedback_memory: List[str],
        cfg: dict
    ) -> DischargeSummaryDraft:
        memory_context = ""
        if feedback_memory:
            memory_context = "\nCRITICAL CLINICAL STYLE & FORMATTING RULES (MUST BE APPLIED):\n"
            for rule in feedback_memory:
                memory_context += f"- {rule}\n"

        flags_text = ""
        if safety_flags:
            flags_text = "\nAUDITED CLINICAL SAFETY FLAGS (MUST BE POPULATED IN DRAFT):\n"
            for flag in safety_flags:
                flags_text += f"- Category: {flag.category} | Item: {flag.item_involved} | Description: {flag.description} | Action: {flag.action_taken}\n"

        prompt = (
            "You are a Clinical Writer agent. Synthesize the final schema-compliant discharge summary.\n"
            "Integrate the intermediate draft, the safety flags raised, and apply the clinician style preferences.\n"
            f"{memory_context}\n"
            f"{flags_text}\n"
            "INTERMEDIATE DRAFT:\n"
            f"{intermediate_draft.model_dump_json(indent=2)}\n\n"
            "RAW PATIENT NOTES REFERENCE:\n"
            f"{raw_clinical_text[:2500]}\n\n"
            "Return the final updated draft as a raw JSON matching the DischargeSummaryDraft schema (no backticks, no other text):\n"
            "{\n"
            "  \"patient_name\": \"string\",\n"
            "  \"medical_record_number\": \"string\",\n"
            "  \"age_and_gender\": \"string\",\n"
            "  \"admission_date\": \"string\",\n"
            "  \"discharge_date\": \"string\",\n"
            "  \"principal_diagnosis\": \"string\",\n"
            "  \"secondary_diagnoses\": [\"string\"],\n"
            "  \"hospital_course\": \"string\",\n"
            "  \"procedures_performed\": [\"string\"],\n"
            "  \"discharge_medications\": [\n"
            "    {\n"
            "      \"name\": \"string\",\n"
            "      \"dosage\": \"string\",\n"
            "      \"frequency\": \"string\",\n"
            "      \"status\": \"UNCHANGED\" | \"ADDED\" | \"DISCONTINUED\" | \"DOSAGE_CHANGED\",\n"
            "      \"reconciliation_note\": \"string\"\n"
            "    }\n"
            "  ],\n"
            "  \"allergies\": [\"string\"],\n"
            "  \"follow_up_instructions\": \"string\",\n"
            "  \"pending_results\": [\"string\"],\n"
            "  \"discharge_condition\": \"string\",\n"
            "  \"clinical_safety_flags\": [\n"
            "    {\n"
            "      \"category\": \"MISSING_DATA\" | \"MEDICATION_MISMATCH\" | \"CONFLICTING_DIAGNOSES\" | \"PENDING_RESULT_WARNING\",\n"
            "      \"item_involved\": \"string\",\n"
            "      \"description\": \"string\",\n"
            "      \"action_taken\": \"string\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        res = self.call_llm_fn(prompt, cfg)
        if res["status"] != "SUCCESS":
            raise RuntimeError(f"ClinicalWriterAgent compile call failed: {res.get('error')}")

        content = res["content"].strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"ClinicalWriterAgent failed to locate valid JSON: {content}")

        json_str = content[start_idx:end_idx + 1]
        return DischargeSummaryDraft.model_validate_json(json_str)
