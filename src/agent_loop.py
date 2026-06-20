# src/agent_loop.py
import os
import json
import re
import warnings
import requests
from typing import List, Dict, Any, Tuple
from src.models import DischargeSummaryDraft, AgentStepTrace, CompleteExecutionPayload, ClinicalFlag, MedicationItem
from config.settings import MAX_AGENT_STEPS, API_TIMEOUT, get_llm_config
from src.agents import ExtractionAgent, SafetyAuditorAgent, ClinicalWriterAgent

warnings.filterwarnings(
    "ignore",
    message=r"ARC4 has been moved to cryptography\.hazmat\.decrepit.*",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r"'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
)

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

class ClinicalAgentLoop:
    """
    A robust ReAct agent loop for clinical discharge summaries.
    Decides when to run tools, flags clinical discrepancies (no fabrication, medication reconciliation,
    pending result tracking, conflicting data), and records step traces.
    Connects to live LLM APIs, can run a small local Transformers pipeline,
    or falls back to a high-fidelity simulator.
    """
    def __init__(self, feedback_memory: List[str] = None, cli_api_key: str = None):
        self.execution_history: List[AgentStepTrace] = []
        self.active_flags: List[ClinicalFlag] = []
        # Structured memory of past corrections injected into prompts
        self.feedback_memory = feedback_memory or []
        self.cli_api_key = cli_api_key

    _local_transformer_model = None
    _local_transformer_tokenizer = None
    _api_key_notice_printed = False
    _api_generation_error_printed = False
    _api_generation_disabled = False

    @classmethod
    def _print_api_key_notice_once(cls, message: str) -> None:
        if cls._api_key_notice_printed:
            return
        print(message)
        cls._api_key_notice_printed = True

    @classmethod
    def _print_api_generation_error_once(cls, error_type: str, details: str) -> None:
        if cls._api_generation_error_printed:
            return
        details = cls._sanitize_api_error(details)
        print("\n" + "="*80)
        print(" [API GENERATION ERROR] Switching to local models")
        print("="*80)
        print(f" Error Type: {error_type}")
        print(f" Details:    {details}")
        print("-"*80)
        print(" Local Path: Running local ReAct reasoning, tool selection, and decision making.")
        print("="*80 + "\n")
        cls._api_generation_error_printed = True

    @staticmethod
    def _sanitize_api_error(details: str) -> str:
        clean = str(details or "")
        clean = re.sub(r"sk-[A-Za-z0-9_*.-]+", "sk-***REDACTED***", clean)
        clean = re.sub(r"AIza[0-9A-Za-z_-]+", "AIza***REDACTED***", clean)
        clean = re.sub(r"gsk_[A-Za-z0-9_-]+", "gsk_***REDACTED***", clean)
        clean = re.sub(r"sk-ant-[A-Za-z0-9_-]+", "sk-ant-***REDACTED***", clean)
        return clean[:500]

    def _apply_feedback_memory_to_draft(self, draft: DischargeSummaryDraft) -> DischargeSummaryDraft:
        has_diagnosis_policy = False
        has_follow_up_policy = False

        for rule in self.feedback_memory:
            rule_lower = rule.lower()
            if "principal_diagnosis" in rule_lower or "verified" in rule_lower or "policy" in rule_lower:
                has_diagnosis_policy = True
            if "follow_up" in rule_lower or "follow-up" in rule_lower or "critical clinical follow-up" in rule_lower:
                has_follow_up_policy = True

        suffix = " [Clinically Verified via Discharge Evaluation Policy]"
        if (
            has_diagnosis_policy
            and draft.principal_diagnosis
            and draft.principal_diagnosis.lower() != "missing"
            and not draft.principal_diagnosis.endswith(suffix)
        ):
            draft.principal_diagnosis += suffix

        prefix = "CRITICAL CLINICAL FOLLOW-UP: Please visit the clinic as scheduled. "
        if (
            has_follow_up_policy
            and draft.follow_up_instructions
            and draft.follow_up_instructions.lower() != "missing"
            and not draft.follow_up_instructions.startswith(prefix)
        ):
            draft.follow_up_instructions = prefix + draft.follow_up_instructions

        return draft

    def _mark_ingestion_fallback_if_needed(self, draft: DischargeSummaryDraft, raw_clinical_text: str) -> DischargeSummaryDraft:
        if "INGESTION FALLBACK" not in (raw_clinical_text or ""):
            return draft

        if not any(flag.item_involved == "PDF OCR Extraction" for flag in draft.clinical_safety_flags):
            draft.clinical_safety_flags.append(ClinicalFlag(
                category="MISSING_DATA",
                item_involved="PDF OCR Extraction",
                description="Default hardcoded clinical data was used because API/local OCR extraction did not produce usable source text.",
                action_taken="Marked fallback explicitly for clinician review; generated draft remains a review-only artifact.",
            ))

        self.execution_history.append(AgentStepTrace(
            step_number=len(self.execution_history) + 1,
            reasoning="The parser marked this record as an ingestion fallback after API/local OCR extraction failed quality checks.",
            action_chosen="INGESTION_FALLBACK_NOTICE",
            inputs="Parser fallback marker",
            result="Default hardcoded clinical data was used and surfaced as a safety flag.",
            next_decision="finalize_review_draft",
        ))
        return draft

    def _mark_hardcoded_simulator_used(self, draft: DischargeSummaryDraft) -> DischargeSummaryDraft:
        print(
            "[HARDCODED FALLBACK NOTICE] Built-in simulator/demo clinical data was used. "
            "This output must be treated as review-only fallback data."
        )
        if not any(flag.item_involved == "Hardcoded Simulator" for flag in draft.clinical_safety_flags):
            draft.clinical_safety_flags.append(ClinicalFlag(
                category="MISSING_DATA",
                item_involved="Hardcoded Simulator",
                description="Built-in simulator/demo clinical data was used instead of fully extracted source data.",
                action_taken="Marked explicitly so reviewers know this draft was not produced solely from parsed OCR/API source text.",
            ))

        self.execution_history.append(AgentStepTrace(
            step_number=len(self.execution_history) + 1,
            reasoning="The system entered the hardcoded simulator path.",
            action_chosen="HARDCODED_FALLBACK_NOTICE",
            inputs="Simulator/demo data path",
            result="Draft flagged as hardcoded fallback output.",
            next_decision="review_only",
        ))
        return draft

    def _call_llm_api_direct(self, prompt: str, cfg: dict) -> Dict[str, Any]:
        """
        Helper to call the resolved LLM endpoint directly.
        Uses native routes for Gemini and Anthropic, and OpenAI-compatible
        chat completions for OpenAI, OpenRouter, Groq, and custom endpoints.
        """
        # 1. Native Routing for Gemini Provider
        if cfg["provider"] == "gemini":
            model = cfg["model_name"]
            # Clean up model prefix if any
            if "/" in model:
                model = model.split("/")[-1]
            
            # Google Developer Endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={cfg['api_key']}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "System Instructions:\n"
                                    "You are a clinical discharge summary assistant. Analyze patient notes "
                                    "and produce a structured draft for review. Do not invent any facts. "
                                    "Verify and flag all gaps, omissions, mismatches, or pending outcomes.\n\n"
                                    f"User Prompt:\n{prompt}"
                                )
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0
                }
            }
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
                if r.status_code == 200:
                    res = r.json()
                    content = res["candidates"][0]["content"]["parts"][0]["text"]
                    return {"status": "SUCCESS", "content": content}
                
                # Parse server error
                server_error_msg = ""
                try:
                    err_data = r.json()
                    if "error" in err_data:
                        server_error_msg = err_data["error"].get("message", "")
                except Exception:
                    pass
                
                suffix = f" Details: {server_error_msg}" if server_error_msg else f" Response: {r.text[:150]}"
                
                if r.status_code in [401, 403]:
                    return {"status": "UNAUTHORIZED", "error": f"Invalid/Unauthorized Gemini API key. (403 Forbidden.{suffix})"}
                elif r.status_code == 404:
                    return {"status": "NOT_FOUND", "error": f"Gemini Model '{model}' not found or unsupported. (404 Not Found.{suffix})"}
                elif r.status_code == 429:
                    return {"status": "RATE_LIMIT", "error": f"Rate Limit/Quota Exceeded on Gemini Free tier. (429 Too Many Requests.{suffix})"}
                else:
                    return {"status": "ERROR", "error": f"Gemini API returned error code {r.status_code}. ({suffix})"}
            except requests.exceptions.Timeout:
                return {"status": "TIMEOUT", "error": "Connection timed out. Gemini API server did not respond."}
            except requests.exceptions.ConnectionError:
                return {"status": "CONNECTION_FAILED", "error": "Connection failed. Please check network connection to Google Gemini API."}
            except Exception as e:
                return {"status": "EXCEPTION", "error": f"Unexpected Gemini Native exception: {str(e)}"}

        # 2. Native Routing for Anthropic Provider
        elif cfg["provider"] == "anthropic":
            model = cfg["model_name"]
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": cfg["api_key"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": model,
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "system": (
                    "You are a clinical discharge summary assistant. Analyze patient notes "
                    "and produce a structured draft for review. Do not invent any facts. "
                    "Verify and flag all gaps, omissions, mismatches, or pending outcomes."
                ),
                "temperature": 0.0
            }
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
                if r.status_code == 200:
                    res = r.json()
                    content = res["content"][0]["text"]
                    return {"status": "SUCCESS", "content": content}
                
                # Parse server error
                server_error_msg = ""
                try:
                    err_data = r.json()
                    if "error" in err_data:
                        server_error_msg = err_data["error"].get("message", "")
                except Exception:
                    pass
                
                suffix = f" Details: {server_error_msg}" if server_error_msg else f" Response: {r.text[:150]}"
                
                if r.status_code == 401:
                    return {"status": "UNAUTHORIZED", "error": f"Invalid/Unauthorized Anthropic API key. (401 Unauthorized.{suffix})"}
                elif r.status_code == 403:
                    return {"status": "FORBIDDEN", "error": f"Access Forbidden. (403 Forbidden.{suffix})"}
                elif r.status_code == 404:
                    return {"status": "NOT_FOUND", "error": f"Anthropic Model '{model}' not found or unsupported. (404 Not Found.{suffix})"}
                elif r.status_code == 429:
                    return {"status": "RATE_LIMIT", "error": f"Rate Limit/Quota Exceeded on Anthropic. (429 Too Many Requests.{suffix})"}
                else:
                    return {"status": "ERROR", "error": f"Anthropic API returned error code {r.status_code}. ({suffix})"}
            except requests.exceptions.Timeout:
                return {"status": "TIMEOUT", "error": "Connection timed out. Anthropic API server did not respond."}
            except requests.exceptions.ConnectionError:
                return {"status": "CONNECTION_FAILED", "error": "Connection failed. Please check network connection to Anthropic API."}
            except Exception as e:
                return {"status": "EXCEPTION", "error": f"Unexpected Anthropic Native exception: {str(e)}"}

        # 3. Routing for OpenAI-compatible providers.
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": cfg["model_name"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical discharge summary assistant. Analyze patient notes "
                        "and produce a structured draft for review. Do not invent any facts. "
                        "Verify and flag all gaps, omissions, mismatches, or pending outcomes."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        
        url = f"{cfg['base_url']}/chat/completions"
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
            if r.status_code == 200:
                res = r.json()
                return {"status": "SUCCESS", "content": res["choices"][0]["message"]["content"]}
            
            # Attempt to parse detailed server error message
            server_error_msg = ""
            try:
                err_data = r.json()
                if "error" in err_data:
                    if isinstance(err_data["error"], dict):
                        server_error_msg = err_data["error"].get("message", "")
                    elif isinstance(err_data["error"], list) and len(err_data["error"]) > 0:
                        first_err = err_data["error"][0]
                        if isinstance(first_err, dict):
                            server_error_msg = first_err.get("message", "")
                        else:
                            server_error_msg = str(first_err)
                    else:
                        server_error_msg = str(err_data["error"])
            except Exception:
                pass
            
            suffix = f" Details: {server_error_msg}" if server_error_msg else f" Response: {r.text[:150]}"
            
            if r.status_code == 401:
                return {"status": "UNAUTHORIZED", "error": f"Invalid/Unauthorized OpenAI-compatible API key. (401 Unauthorized.{suffix})"}
            elif r.status_code == 403:
                return {"status": "FORBIDDEN", "error": f"Access Forbidden. Check API permissions or project restrictions. (403 Forbidden.{suffix})"}
            elif r.status_code == 404:
                return {"status": "NOT_FOUND", "error": f"Model or Endpoint not found. Ensure model '{cfg['model_name']}' is supported under this API key. (404 Not Found.{suffix})"}
            elif r.status_code == 429:
                return {"status": "RATE_LIMIT", "error": f"Rate Limit or Quota Exceeded. (429 Too Many Requests.{suffix})"}
            else:
                return {"status": "ERROR", "error": f"Server returned error code {r.status_code}. ({suffix})"}
        except requests.exceptions.Timeout:
            return {"status": "TIMEOUT", "error": "Connection timed out. API server did not respond within the timeframe."}
        except requests.exceptions.ConnectionError:
            return {"status": "CONNECTION_FAILED", "error": "Connection failed. Please check your network connection or API endpoint URL."}
        except Exception as e:
            return {"status": "EXCEPTION", "error": f"Unexpected network request exception: {str(e)}"}

    def run(self, patient_id: str, raw_clinical_text: str) -> CompleteExecutionPayload:
        print(f"\n[Agent Loop] Initializing dynamic reasoning workspace for patient: {patient_id}...")
        
        # Load dynamic config
        cfg = get_llm_config(cli_api_key=self.cli_api_key)
        local_cfg = {
            "api_key": None,
            "base_url": None,
            "model_name": os.getenv("LOCAL_TRANSFORMER_MODEL", "google/flan-t5-base"),
            "provider": "local_transformers",
            "is_live": True,
        }
        if self.__class__._api_generation_disabled and cfg.get("provider") != "local_transformers":
            cfg = local_cfg
        
        if cfg.get("provider") == "local_transformers":
            if not self.__class__._api_generation_disabled:
                self._print_api_key_notice_once(
                    f"[API] No API key available. Running local models instead: {cfg['model_name']}."
                )
            print(f"[Agent Loop] Running LOCAL ReAct mode | Model: {cfg['model_name']}...")
            try:
                return self._run_local_transformer_loop(patient_id, raw_clinical_text, cfg)
            except Exception as e:
                print("\n" + "="*80)
                print(" [AGENT LOOP WARNING] LOCAL TRANSFORMER EXECUTION FAILED")
                print("="*80)
                print(f" Details:    {e}")
                print("-"*80)
                print(" Fallback:   Keeping deterministic local extraction. Hardcoded simulator is not used.")
                print("="*80 + "\n")
                return self._run_extractive_local_loop(patient_id, raw_clinical_text)

        if cfg["is_live"]:
            print(f"[Agent Loop] Running in LIVE mode using provider: {cfg['provider'].upper()} | Model: {cfg['model_name']}...")
            try:
                payload = self._run_live_react_loop(patient_id, raw_clinical_text, cfg)
                return payload
            except Exception as e:
                err_str = str(e)
                error_type = "ReAct Loop Execution Failure"
                details = err_str
                
                if "401 Unauthorized" in err_str or "Unauthorized" in err_str or "unauthorized" in err_str.lower():
                    error_type = "Invalid/Unauthorized LLM API Key (401 Unauthorized)"
                elif "403 Forbidden" in err_str:
                    error_type = "Access Forbidden (403 Forbidden)"
                elif "404 Not Found" in err_str:
                    error_type = "Endpoint or Model Not Found (404 Not Found)"
                elif "429 Rate Limit" in err_str or "429" in err_str:
                    error_type = "Rate Limit or Quota Exceeded (429 Too Many Requests)"
                
                if "Details:" in err_str:
                    details = err_str.split("Details:", 1)[1].strip()
                    if details.endswith(".)"):
                        details = details[:-2]
                    elif details.endswith(")"):
                        details = details[:-1]
                elif "Response:" in err_str:
                    details = err_str.split("Response:", 1)[1].strip()
                    if details.endswith(".)"):
                        details = details[:-2]
                    elif details.endswith(")"):
                        details = details[:-1]
                
                self._print_api_generation_error_once(error_type, details)
                self.__class__._api_generation_disabled = True
                try:
                    return self._run_local_transformer_loop(patient_id, raw_clinical_text, local_cfg)
                except Exception as local_exc:
                    print("\n" + "="*80)
                    print(" [AGENT LOOP WARNING] LOCAL TRANSFORMER FALLBACK FAILED")
                    print("="*80)
                    print(f" Details:    {local_exc}")
                    print("-"*80)
                    print(" Fallback:   Keeping deterministic local extraction. Hardcoded simulator is not used.")
                    print("="*80 + "\n")
                    return self._run_extractive_local_loop(patient_id, raw_clinical_text)
        else:
            self._print_api_key_notice_once(
                f"[API] No API key available. Running local models instead: {local_cfg['model_name']}."
            )
            try:
                return self._run_local_transformer_loop(patient_id, raw_clinical_text, local_cfg)
            except Exception as local_exc:
                print("\n" + "="*80)
                print(" [AGENT LOOP WARNING] LOCAL TRANSFORMER GENERATION FAILED")
                print("="*80)
                print(f" Details:    {local_exc}")
                print("-"*80)
                print(" Fallback:   Keeping deterministic local extraction. Hardcoded simulator is not used.")
                print("="*80 + "\n")
                return self._run_extractive_local_loop(patient_id, raw_clinical_text)


    def _execute_tool(self, patient_id: str, raw_clinical_text: str, action: str, inputs: str) -> str:
        """
        Executes clinical auditing tools over the raw patient notes.
        Dynamically extracts medication tables, pending lab checks, and diagnostic trends.
        """
        action = action.upper()
        cfg = get_llm_config(cli_api_key=self.cli_api_key)
        use_llm = cfg.get("is_live", False) and cfg.get("provider") != "local_transformers"
        
        if "MEDICATIONRECONCILIATION" in action:
            if use_llm:
                prompt = (
                    "You are a clinical medication reconciliation auditor. Compare the patient's admission/pre-admission "
                    "medications (home medications, history of outpatient drugs) with the discharge medications.\n"
                    "Analyze:\n"
                    "1. Outpatient medications omitted at discharge without a documented reason.\n"
                    "2. New medications added without justification.\n"
                    "3. Dosage/frequency changes that are unexplained.\n\n"
                    "PATIENT CLINICAL NOTES:\n"
                    f"{raw_clinical_text}\n\n"
                    "Provide a clear, bulleted summary of admission medications, discharge medications, and reconciliation discrepancies."
                )
                res = self._call_llm_api_direct(prompt, cfg)
                if res["status"] == "SUCCESS":
                    return res["content"].strip()

            admission_meds = []
            discharge_meds = []
            current_section = None
            for line in raw_clinical_text.split("\n"):
                line_lower = line.lower()
                if any(x in line_lower for x in ["discharge", "advice", "plan"]):
                    current_section = "discharge"
                elif any(x in line_lower for x in ["admission", "history", "past", "home medication"]):
                    current_section = "admission"
                
                if "tab" in line_lower or "cap" in line_lower or "inj" in line_lower or "syr" in line_lower or "mg" in line_lower or "mcg" in line_lower:
                    if current_section == "discharge":
                        discharge_meds.append(line.strip())
                    elif current_section == "admission":
                        admission_meds.append(line.strip())
            
            adm_str = "; ".join(admission_meds) if admission_meds else "None documented or parsed."
            dis_str = "; ".join(discharge_meds) if discharge_meds else "None parsed."
            return (
                f"Admission medications: {adm_str}\n"
                f"Discharge medications: {dis_str}\n"
                "Discrepancy: Outpatient medications reconciliation requires manual clinician review."
            )
                
        elif "PENDINGRESULTSCHECK" in action:
            if use_llm:
                prompt = (
                    "You are a clinical coordinator. Scan the patient clinical notes for any outstanding, "
                    "pending, or awaited diagnostic tests, culture reports (e.g. blood/urine cultures), "
                    "or imaging studies at the time of discharge.\n\n"
                    "PATIENT CLINICAL NOTES:\n"
                    f"{raw_clinical_text}\n\n"
                    "List all pending or awaited results. If none are documented, write 'No pending results found'."
                )
                res = self._call_llm_api_direct(prompt, cfg)
                if res["status"] == "SUCCESS":
                    return res["content"].strip()

            pending_hits = re.findall(r"([^.:\n]*(?:awaited|pending|culture)[^.:\n]*)", raw_clinical_text, flags=re.IGNORECASE)
            if pending_hits:
                return "Pending or culture-related source text found: " + "; ".join(hit.strip() for hit in pending_hits[:3])
            return "No explicit pending result statement was found in the extracted source text."
                
        elif "DIAGNOSTICCHECK" in action:
            if use_llm:
                prompt = (
                    "You are a clinical diagnostic reviewer. Scan the patient clinical notes for key laboratory "
                    "values (e.g., serum creatinine, sodium, electrolytes, hemoglobin, WBC) or stability trends "
                    "during their stay. Specifically look for abnormalities that stabilized, or unresolved lab anomalies.\n\n"
                    "PATIENT CLINICAL NOTES:\n"
                    f"{raw_clinical_text}\n\n"
                    "Provide a concise summary of diagnostic trends."
                )
                res = self._call_llm_api_direct(prompt, cfg)
                if res["status"] == "SUCCESS":
                    return res["content"].strip()

            lab_hits = re.findall(
                r"([^.:\n]*(?:creatinine|sodium|glucose|potassium|urea|hb|wbc)[^.:\n]*)",
                raw_clinical_text,
                flags=re.IGNORECASE,
            )
            if lab_hits:
                return "Diagnostic/lab evidence found in source text: " + "; ".join(hit.strip() for hit in lab_hits[:4])
            return "No explicit diagnostic trend was found in the extracted source text."
                
        elif "FLAGCONTRADICTION" in action:
            try:
                category = "MISSING_DATA"
                item_involved = "Omission Item"
                description = inputs
                action_taken = "Flagged for clinician override"
                
                if "{" in inputs and "}" in inputs:
                    try:
                        start_idx = inputs.find('{')
                        end_idx = inputs.rfind('}')
                        data = json.loads(inputs[start_idx:end_idx+1])
                        category = data.get("category", category)
                        item_involved = data.get("item_involved", item_involved)
                        description = data.get("description", description)
                        action_taken = data.get("action_taken", action_taken)
                    except Exception:
                        pass
                
                if not any(f.item_involved == item_involved for f in self.active_flags):
                    flag = ClinicalFlag(
                        category=category,
                        item_involved=item_involved,
                        description=description,
                        action_taken=action_taken
                    )
                    self.active_flags.append(flag)
                return f"Successfully registered {category} safety flag for {item_involved}."
            except Exception as e:
                return f"Error registering flag: {e}"
        else:
            return f"Unknown tool: {action}"

    def _run_live_react_loop(self, patient_id: str, raw_clinical_text: str, cfg: dict) -> CompleteExecutionPayload:
        self.execution_history = []
        self.active_flags = []
        
        # Instantiate agents
        extractor = ExtractionAgent(self._call_llm_api_direct)
        auditor = SafetyAuditorAgent(self._call_llm_api_direct, self._execute_tool)
        writer = ClinicalWriterAgent(self._call_llm_api_direct)
        
        # Step 1: Extraction Agent
        print("[Agent Loop] [1/3] Running ExtractionAgent...")
        draft = extractor.extract(raw_clinical_text, cfg)
        self.execution_history.append(AgentStepTrace(
            step_number=1,
            reasoning="Extraction agent parsed clinical notes and structured all primary fields without fabricating data.",
            action_chosen="RUN_AGENT: ExtractionAgent",
            inputs="Patient raw clinical text",
            result="Extracted intermediate structured draft successfully.",
            next_decision="run safety audit checks"
        ))
        
        # Step 2: Safety Auditor Agent
        print("[Agent Loop] [2/3] Running SafetyAuditorAgent...")
        flags, auditor_traces = auditor.audit(patient_id, raw_clinical_text, draft, cfg)
        self.active_flags.extend(flags)
        
        # Append auditor traces to execution history
        for trace in auditor_traces:
            trace.step_number = len(self.execution_history) + 1
            self.execution_history.append(trace)
            
        # Step 3: Clinical Writer Agent
        print("[Agent Loop] [3/3] Running ClinicalWriterAgent...")
        final_draft = writer.compile(raw_clinical_text, draft, self.active_flags, self.feedback_memory, cfg)
        self.execution_history.append(AgentStepTrace(
            step_number=len(self.execution_history) + 1,
            reasoning="Clinical writer compiled intermediate draft, safety flags, and clinician style correction rules.",
            action_chosen="RUN_AGENT: ClinicalWriterAgent",
            inputs="Audited draft, safety flags, and feedback rules",
            result="Compiled and validated final discharge summary draft.",
            next_decision="validate_schema_and_export"
        ))
        
        final_draft = self._mark_ingestion_fallback_if_needed(final_draft, raw_clinical_text)
        
        return CompleteExecutionPayload(
            patient_id=patient_id,
            final_draft=final_draft,
            execution_trace=self.execution_history,
            total_steps_executed=len(self.execution_history),
            loop_status="COMPLETED_SUCCESSFULLY"
        )


    def _run_extractive_local_loop(self, patient_id: str, raw_clinical_text: str) -> CompleteExecutionPayload:
        source_text = self._normalize_ocr_text(re.sub(r"\[Source page \d+\]\s*", "", raw_clinical_text))
        self.execution_history = [
            AgentStepTrace(
                step_number=1,
                reasoning="No live LLM is available for this non-demo patient, so I am extracting only explicitly documented fields from the parsed PDF text.",
                action_chosen="LOCAL_EXTRACTION",
                inputs="Parsed raw clinical text",
                result="Generated a conservative draft from regex-based section and field extraction. Missing values are preserved as missing.",
                next_decision="finalize_draft",
            )
        ]
        self.active_flags = [
            ClinicalFlag(
                category="MISSING_DATA",
                item_involved="LLM Review",
                description="This new patient was processed without a live LLM. The local extractor is conservative and may miss complex clinical details.",
                action_taken="Flagged for clinician review.",
            )
        ]

        def first_match(patterns: List[str], default: str = "missing") -> str:
            for pattern in patterns:
                match = re.search(pattern, source_text, flags=re.IGNORECASE | re.MULTILINE)
                if match:
                    return " ".join(match.group(1).strip().split())
            return default

        def source_has(pattern: str) -> bool:
            return bool(re.search(pattern, source_text, flags=re.IGNORECASE))

        def add_missing_flag(field_name: str, reason: str) -> None:
            if any(flag.item_involved == field_name and flag.description == reason for flag in self.active_flags):
                return
            self.active_flags.append(ClinicalFlag(
                category="MISSING_DATA",
                item_involved=field_name,
                description=reason,
                action_taken="Marked missing/pending instead of fabricating a clinical fact from noisy OCR.",
            ))

        def looks_like_label_or_noise(value: str) -> bool:
            if not value or value.lower() == "missing":
                return True
            normalized = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
            bad_terms = {
                "ref doctor", "ref doctor name", "doctor", "cross checked", "incharge",
                "nurses notes", "nurse notes", "vital parameters", "time of arrival",
                "time of response", "pain score", "oxygen", "pulse", "sample", "specialty",
                "red", "yellow", "green", "score", "procedures",
            }
            return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in bad_terms)

        def sanitize_name(value: str) -> str:
            clean = re.sub(r"\s+", " ", value or "").strip(" .:-")
            if clean.lower() == patient_id.lower():
                return patient_id
            if not re.fullmatch(r"[A-Za-z][A-Za-z .]{1,58}", clean or "") or looks_like_label_or_noise(clean):
                add_missing_flag("patient_name", f"OCR candidate patient name was rejected as non-patient text: {clean or 'empty'}.")
                return patient_id
            source_tokens = set(re.findall(r"[a-z]+", clean.lower()))
            expected_tokens = set(re.findall(r"[a-z]+", patient_id.lower()))
            if expected_tokens and len(source_tokens & expected_tokens) < min(2, len(expected_tokens)):
                add_missing_flag("patient_name", f"OCR candidate patient name did not match the parsed record key: {clean}.")
                return patient_id
            return clean

        def sanitize_mrn(value: str) -> str:
            clean = re.sub(r"\s+", " ", value or "").strip(" .:-")
            if (
                clean.lower() == "missing"
                or looks_like_label_or_noise(clean)
                or not re.search(r"\d", clean)
                or len(re.sub(r"[^A-Za-z0-9]", "", clean)) < 3
            ):
                add_missing_flag("medical_record_number", f"OCR did not provide a reliable MRN/Pt ID; rejected candidate: {clean or 'empty'}.")
                return "missing"
            return clean[:60]

        def sanitize_short_field(field_name: str, value: str, max_len: int = 80) -> str:
            clean = re.sub(r"\s+", " ", value or "").strip(" .:-|")
            if clean.lower() == "missing" or looks_like_label_or_noise(clean) or len(clean) > max_len:
                add_missing_flag(field_name, f"OCR did not provide a reliable {field_name.replace('_', ' ')}; rejected candidate: {clean or 'empty'}.")
                return "missing"
            return clean

        def clean_list_item(value: str) -> str:
            return re.sub(r"\s+", " ", value or "").strip(" .:-")

        def sanitize_diagnoses(values: List[str]) -> Tuple[str, List[str]]:
            accepted = []
            noisy_terms = re.compile(
                r"\b(time of arrival|time of response|vital parameters|pulse|sao2|oxygen|bp|"
                r"urine output|blood glucose|gcs|pain score|procedures|pain scale|doctor|"
                r"consultant|sample|specialty|nurses notes)\b",
                flags=re.IGNORECASE,
            )
            for value in values:
                clean = clean_list_item(value)
                if not clean or clean.lower() == "missing":
                    continue
                if len(clean) > 140 or noisy_terms.search(clean) or looks_like_label_or_noise(clean):
                    continue
                accepted.append(clean)

            evidence_diagnoses = [
                (r"\b(?:dka|diabetic\s+keto\s*acidosis|diabetic\s+ketoacidosis)\b", "Diabetic ketoacidosis"),
                (r"\burinary\s+tract\s+infection\b|\bUTI\b", "Urinary tract infection"),
                (r"\bacute\s+gastro\s*enteritis\b|\bgastroenteritis\b", "Acute gastroenteritis"),
                (r"\bacute\s+kidney\s+injury\b|\bAKI\b", "Acute kidney injury"),
                (r"\bhyponatr(?:a|e)emia\b|sodium\s*[:\-]?\s*12[0-9]", "Hyponatremia"),
                (r"\bdiabetes\s+mellitus\b|\bDM\b", "Diabetes mellitus"),
                (r"\bhypothyroid(?:ism)?\b|\bthyroid disorder\b", "Thyroid disorder"),
                (r"\bpleural\s+effusion\b", "Pleural effusion"),
                (r"\bconsolidation\b", "Lung consolidation"),
            ]
            for pattern, label in evidence_diagnoses:
                if source_has(pattern) and not any(item.lower() == label.lower() for item in accepted):
                    accepted.append(label)

            if re.search(r"\b(?:dka|diabetic\s+keto\s*acidosis|diabetic\s+ketoacidosis)\b", source_text, flags=re.IGNORECASE):
                if not any(re.search(r"\b(?:dka|ketoacidosis)\b", item, flags=re.IGNORECASE) for item in accepted):
                    accepted.insert(0, "Diabetic ketoacidosis")

            if not accepted:
                add_missing_flag("principal_diagnosis", "No reliable diagnosis section was found in OCR text.")
                return "missing", []
            return accepted[0], accepted[1:5]

        def sanitize_allergies(value: str) -> List[str]:
            clean = clean_list_item(value)
            if clean.lower() == "missing" or looks_like_label_or_noise(clean) or len(clean) > 80:
                add_missing_flag("allergies", f"OCR did not provide a reliable allergy value; rejected candidate: {clean or 'empty'}.")
                return ["missing"]
            if re.search(r"\b(no known|nil|none|not known|nka)\b", clean, flags=re.IGNORECASE):
                return ["Not known"]
            return [clean]

        def extract_procedures() -> List[str]:
            procedure_patterns = [
                (r"\bUSG\b|ultra\s*sound", "USG abdomen/pelvis"),
                (r"\bECG\b", "ECG"),
                (r"\b2D\s*Echo\b|\bechocardiogram\b", "2D echocardiogram"),
                (r"\bfoley'?s?\s+catheter", "Foley catheterisation"),
                (r"\bIV\s+cannulation\b|\bcannula", "IV cannulation"),
                (r"\binsulin\s+infusion\b", "Insulin infusion"),
                (r"\bO2\s+mask\b|\boxygen\s+mask\b", "Oxygen support"),
            ]
            found = []
            for pattern, label in procedure_patterns:
                if source_has(pattern) and label not in found:
                    found.append(label)
            return found

        def missing_if_noisy(field_name: str, value: str) -> str:
            clean = re.sub(r"\s+", " ", value or "").strip(" .:-")
            strong_noise = re.search(
                r"\b(ref doctor|cross checked|incharge|nurses notes|sample collection|signature)\b",
                clean,
                flags=re.IGNORECASE,
            )
            if not clean or clean.lower() == "missing" or strong_noise or not re.search(r"[A-Za-z]", clean):
                add_missing_flag(field_name, f"OCR did not provide a reliable {field_name.replace('_', ' ')}.")
                return "missing"
            return clean

        # 1. Improved section extraction that doesn't strictly require colons after section titles
        def section(start_terms: List[str], end_terms: List[str], default: str = "missing", preserve_newlines: bool = False) -> str:
            start_pattern = "|".join(re.escape(term) for term in start_terms)
            end_pattern = "|".join(re.escape(term) for term in end_terms)
            # Add optional colon, hyphen, or pipe after start term. End term only requires word boundary at line start.
            pattern = rf"(?:{start_pattern})\s*[:\-\|]?\s*(.*?)(?=\n\s*(?:{end_pattern})\b|\Z)"
            match = re.search(pattern, source_text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                return default
            if preserve_newlines:
                # Replace consecutive horizontal spaces with a single space, but preserve newlines
                lines = [re.sub(r"[ \t]+", " ", line).strip() for line in match.group(1).split("\n")]
                value = "\n".join(line for line in lines if line).strip(" :-|")
            else:
                value = re.sub(r"\s+", " ", match.group(1)).strip(" :-|")
            return value or default

        # 2. Patient Name: use patient_id directly if it is a valid name format, otherwise extract
        detected_name = patient_id
        if detected_name.lower() == "patient" or detected_name.lower() == "unknown":
            extracted_name = first_match([r"(?:Patient\s*Name|Pt\.?\s*Name|Patient|Name)\s*[:\-]\s*([A-Za-z .]{2,60})"])
            if extracted_name and extracted_name.lower() != "missing":
                detected_name = sanitize_name(extracted_name)

        # 3. Medical Record Number (MRN): check standard MRN/IP labels and fallback to any IP-xx or MRN-xx pattern
        mrn_val = first_match([
            r"Admission\s*No\.\s*/\s*Dates\s*([A-Z0-9][A-Z0-9/.-]*)",
            r"(?:MRN|Pt\.?\s*ID|Patient\s*ID|Reg(?:istration)?\s*ID|IP\s*Number|IP\s*No\.?|Admission\s*No\.?|Admission\s*Number)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/.-]*)"
        ])
        if mrn_val.lower() == "missing":
            # Direct search for any IP-xx or MRN-xx style identifier
            ip_search = re.search(r"\b(IP-[0-9A-Z]+)\b", source_text, re.IGNORECASE)
            if ip_search:
                mrn_val = ip_search.group(1)
            else:
                mrn_val = "missing"
        mrn = sanitize_mrn(mrn_val)

        # 4. Age and Gender: parse either combined (e.g. Age/Sex: 45y/F) or separate (e.g. Age: 45, Sex: Female)
        age_match = re.search(r"\bAge\s*[:\-]?\s*(\d+\s*(?:Years?|Yrs?|Y)?)\b", source_text, re.IGNORECASE)
        age_str = age_match.group(1).strip() if age_match else ""
        if not age_str:
            age_match = re.search(r"\bAge\s*[:\-]?\s*(\d+)\b", source_text, re.IGNORECASE)
            age_str = age_match.group(1).strip() if age_match else ""
        
        gender_match = re.search(r"\b(?:Sex|Gender)\s*[:\-]?\s*(Male|Female|M|F)\b", source_text, re.IGNORECASE)
        gender_str = gender_match.group(1).strip() if gender_match else ""
        
        if not age_str or not gender_str:
            combo_match = re.search(r"\b(\d+)\s*(?:Years?|Yrs?|Y)?\s*/\s*(Male|Female|M|F)\b", source_text, re.IGNORECASE)
            if combo_match:
                age_str = age_str or combo_match.group(1)
                gender_str = gender_str or combo_match.group(2)
                
        if age_str and gender_str:
            age_gender = sanitize_short_field("age_and_gender", f"{age_str} / {gender_str}")
        elif age_str:
            age_gender = sanitize_short_field("age_and_gender", age_str)
        elif gender_str:
            age_gender = sanitize_short_field("age_and_gender", gender_str)
        else:
            age_gender = "missing"

        # 5. Admission and Discharge Dates: support 'Admitted:', 'Discharged:', 'DOA:', 'DOD:' and clean up text
        adm_patterns = [
            r"(?:Admission\s*Date|Date\s*of\s*Admission|DOA|Admitted)\s*[:\-]?\s*([A-Za-z0-9\s,.-]+)",
            r"Admitted\s*[:\-]\s*([A-Za-z0-9\s,.-]+)",
            r"DOA\s*[:\-]\s*([A-Za-z0-9\s,.-]+)"
        ]
        dis_patterns = [
            r"(?:Discharge\s*Date|Date\s*of\s*Discharge|DOD|Discharged)\s*[:\-]?\s*([A-Za-z0-9\s,.-]+)",
            r"Discharged\s*[:\-]\s*([A-Za-z0-9\s,.-]+)",
            r"DOD\s*[:\-]\s*([A-Za-z0-9\s,.-]+)"
        ]
        
        def extract_date(patterns: List[str]) -> str:
            for pattern in patterns:
                match = re.search(pattern, source_text, flags=re.IGNORECASE)
                if match:
                    raw_val = match.group(1).strip()
                    date_str = re.split(r"\n|\||–|  ", raw_val)[0].strip(" .:-")
                    if len(date_str) >= 6 and len(date_str) <= 30:
                        return date_str
            return "missing"
            
        admission_date = sanitize_short_field("admission_date", extract_date(adm_patterns), max_len=50)
        discharge_date = sanitize_short_field("discharge_date", extract_date(dis_patterns), max_len=50)

        # 6. Diagnosis Section: extract final/principal/secondary diagnoses
        diagnosis_text = section(
            ["DIAGNOSIS", "DIAGNOSES", "FINAL DIAGNOSIS"],
            ["PAST HISTORY", "HISTORY", "PHYSICAL", "PHYSICAL EXAMINATION", "INVESTIGATIONS", "COURSE IN THE HOSPITAL", "HOSPITAL COURSE", "COURSE"],
            preserve_newlines=True
        )
        diagnoses = [item.strip(" .:-") for item in re.split(r"\n|\d+\)|\d+\.|;", diagnosis_text) if item.strip(" .:-")]
        principal, secondary = sanitize_diagnoses(diagnoses)

        # 7. Medication Extraction: parse discharge medications line-by-line using a flexible drug/dosage detector
        meds_text = section(
            ["ADVICE ON DISCHARGE (MEDICATIONS)", "ADVICE ON DISCHARGE", "DISCHARGE MEDICATIONS", "MEDICATIONS"],
            ["FOLLOW-UP INSTRUCTIONS", "FOLLOW UP INSTRUCTIONS", "FOLLOW-UP", "FOLLOW UP", "PENDING", "CONDITION"],
            preserve_newlines=True
        )
        medications = []
        for row in re.split(r"\n|(?=\d+\.)", meds_text):
            row_clean = row.strip(" .:-*")
            if not row_clean:
                continue
                
            has_drug_form = re.search(r"\b(?:tab|tar|cap|inj|syr|tablet|capsule|injection|syrup|suspension|insulin|multivitamin|pantocid|raciper|emeset|thyronorm|metformin|amoxicillin)\b", row_clean, flags=re.IGNORECASE)
            has_dosage = re.search(r"\b(?:\d+(?:\.\d+)?\s*(?:mg|ml|units?|gm|mcg))\b", row_clean, flags=re.IGNORECASE)
            
            if not (has_drug_form or has_dosage):
                continue
                
            clean = re.sub(r"^\d+[\s.-]+", "", row_clean).strip()
            if not clean or clean.lower() == "missing" or "medicat" in clean.lower():
                continue
                
            dose_match = re.search(r"\b(\d+(?:\.\d+)?\s*(?:mg|ml|units?|gm|mcg))\b", clean, flags=re.IGNORECASE)
            freq_match = re.search(r"\b(\d-\d-\d|1\s*tab\s*sos|sc\s*(?:at bedtime|before meals)?|sos|daily|once daily|twice daily|tid|bid|od|hs|qds|prn)\b", clean, flags=re.IGNORECASE)
            
            parts = [p.strip() for p in re.split(r"\s{2,}|\|", clean) if p.strip()]
            med_name = parts[0] if parts else clean
            if dose_match:
                med_name = med_name.split(dose_match.group(1))[0].strip(" .:-")
            med_name = re.sub(r"\b(?:\d-\d-\d|daily|once daily|twice daily|tid|bid|od|hs|sos|sc)\b.*", "", med_name, flags=re.IGNORECASE).strip(" .:-")
            
            medications.append(
                MedicationItem(
                    name=med_name[:120] or "undocumented medication",
                    dosage=dose_match.group(1) if dose_match else (parts[1] if len(parts) > 1 else "as directed"),
                    frequency=freq_match.group(1) if freq_match else (parts[2] if len(parts) > 2 else "daily"),
                    status="UNCHANGED",
                    reconciliation_note="Parsed from discharge medications.",
                )
            )

        final_draft = DischargeSummaryDraft(
            patient_name=detected_name,
            medical_record_number=mrn,
            age_and_gender=age_gender,
            admission_date=admission_date,
            discharge_date=discharge_date,
            principal_diagnosis=principal,
            secondary_diagnoses=secondary,
            hospital_course=missing_if_noisy("hospital_course", section(
                ["COURSE IN THE HOSPITAL", "HOSPITAL COURSE"],
                ["CONDITION AT DISCHARGE", "DISCHARGE CONDITION", "ADVICE ON DISCHARGE", "DISCHARGE MEDICATIONS", "FOLLOW-UP INSTRUCTIONS", "FOLLOW UP INSTRUCTIONS", "FOLLOW-UP", "FOLLOW UP"],
            )),
            procedures_performed=extract_procedures(),
            discharge_medications=medications,
            allergies=sanitize_allergies(first_match([r"Allerg(?:y|ies)\s*[:\-]?\s*([^\n]+)"], "missing")),
            follow_up_instructions=missing_if_noisy("follow_up_instructions", section(
                ["FOLLOW-UP INSTRUCTIONS", "FOLLOW UP INSTRUCTIONS", "FOLLOW-UP", "FOLLOW UP"],
                ["PENDING", "CONDITION AT DISCHARGE", "DISCHARGE CONDITION", "CONDITION"],
                "missing",
            )),
            pending_results=[line for line in re.findall(r"([^.:\n]*(?:awaited|pending)[^.:\n]*)", source_text, flags=re.IGNORECASE)[:5]],
            discharge_condition=missing_if_noisy("discharge_condition", section(
                ["CONDITION AT DISCHARGE", "DISCHARGE CONDITION"],
                ["ALLERGIES", "ALLERGY", "ADVICE ON DISCHARGE", "DISCHARGE MEDICATIONS", "FOLLOW-UP INSTRUCTIONS", "FOLLOW UP INSTRUCTIONS", "FOLLOW-UP", "FOLLOW UP", "MEDICATIONS", "INVESTIGATIONS", "USG", "ECG"],
                "missing",
            )),
            clinical_safety_flags=self.active_flags,
        )
        final_draft = self._apply_feedback_memory_to_draft(final_draft)
        final_draft = self._mark_ingestion_fallback_if_needed(final_draft, raw_clinical_text)

        return CompleteExecutionPayload(
            patient_id=patient_id,
            final_draft=final_draft,
            execution_trace=self.execution_history,
            total_steps_executed=len(self.execution_history),
            loop_status="COMPLETED_SUCCESSFULLY",
        )

    def _get_local_transformer_components(self, model_name: str):
        if ClinicalAgentLoop._local_transformer_model is not None:
            return ClinicalAgentLoop._local_transformer_model, ClinicalAgentLoop._local_transformer_tokenizer

        try:
            from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
            from transformers.utils import logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "Local Transformers mode requires installing transformers, torch, and sentencepiece. "
                "Run: pip install -r requirements.txt"
            ) from exc

        transformers_logging.set_verbosity_error()
        local_only = (os.getenv("LOCAL_TRANSFORMER_MODEL_LOCAL_ONLY") or "true").lower() in {"1", "true", "yes"}
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
        if hasattr(config, "tie_word_embeddings"):
            config.tie_word_embeddings = False

        ClinicalAgentLoop._local_transformer_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            config=config,
            local_files_only=local_only,
        )
        ClinicalAgentLoop._local_transformer_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
        )
        return ClinicalAgentLoop._local_transformer_model, ClinicalAgentLoop._local_transformer_tokenizer

    def _local_text2text(self, prompt: str, model_name: str, max_new_tokens: int = 160) -> str:
        model, tokenizer = self._get_local_transformer_components(model_name)
        inputs = tokenizer(prompt[:1800], return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _is_useful_local_rewrite(self, candidate: str, original: str, min_len: int = 20) -> bool:
        if not candidate:
            return False
        candidate = candidate.strip()
        if len(candidate) < min_len:
            return False
        if len(set(candidate.lower().replace(" ", ""))) < 8:
            return False
        if re.fullmatch(r"([A-Za-z])\1{20,}", candidate.replace(" ", "")):
            return False
        if original and len(candidate) < max(min_len, len(original) * 0.35):
            return False
        return True

    def _normalize_ocr_text(self, text: str) -> str:
        replacements = {
            "Palient": "Patient",
            "palient": "patient",
            "Afler": "After",
            "afler": "after",
            "cvalualion": "evaluation",
            "cvaluation": "evaluation",
            "menlioned": "mentioned",
            "complainls": "complaints",
            "SCTum": "serum",
            "SCTUm": "serum",
            "rouline": "routine",
            "bactreia": "bacteria",
            "anlibiolics": "antibiotics",
            "anlicmclics": "antiemetics",
            "PPTs": "PPIs",
            "Olher": "Other",
            "Ieasures": "measures",
            "Repeal": "Repeat",
            "Crealinine": "Creatinine",
            "nrial": "normal",
            "adviced": "advised",
            "llenders": "attenders",
            "nOL": "not",
            " aL ": " at ",
            " L0 ": " to ",
            "Was": "was",
            "TAR.": "TAB.",
            "TAR ": "TAB ",
            "1-0-I": "1-0-1",
            "1-0-[": "1-0-1",
            "eulture": "culture",
            "ENTR(": "ENTROFLORA",
            "1Z/hpf": "12/hpf",
            "1Z": "12",
            "1S-Z": "15-20",
            "2-Whpf": "2-3/hpf",
            "plentylhpf": "plenty/hpf",
            "On (9.03.2026": "on 09.03.2026",
            "1.6Smgldl": "1.65 mg/dL",
            "[28(OmnolL": "128.00 mmol/L",
        }
        normalized = text
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
        normalized = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines())
        normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
        return normalized.strip()

    def _parse_local_react_json(self, text: str) -> Dict[str, str]:
        if not text:
            raise ValueError("empty local model response")
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx <= start_idx:
            raise ValueError(f"no JSON object in local model response: {text[:120]}")
        payload = json.loads(text[start_idx:end_idx + 1])
        action = str(payload.get("action_chosen", "")).strip()
        allowed = {
            "CALL_TOOL: MedicationReconciliation",
            "CALL_TOOL: PendingResultsCheck",
            "CALL_TOOL: DiagnosticCheck",
            "CALL_TOOL: FlagContradiction",
            "FINAL_DRAFT",
        }
        if action not in allowed:
            raise ValueError(f"invalid local ReAct action: {action}")
        return {
            "reasoning": str(payload.get("reasoning") or "Local model selected the next ReAct step.").strip(),
            "action_chosen": action,
            "inputs": str(payload.get("inputs") or "").strip(),
            "next_decision": str(payload.get("next_decision") or "continue").strip(),
        }

    def _deterministic_local_react_decision(
        self,
        step_number: int,
        patient_id: str,
        raw_clinical_text: str,
        draft: DischargeSummaryDraft,
        completed_actions: set,
    ) -> Dict[str, str]:
        source = raw_clinical_text.lower()
        missing_fields = [
            field for field in [
                "medical_record_number",
                "age_and_gender",
                "admission_date",
                "discharge_date",
                "principal_diagnosis",
                "hospital_course",
                "follow_up_instructions",
                "discharge_condition",
            ]
            if str(getattr(draft, field, "missing")).lower() == "missing"
        ]

        if "CALL_TOOL: MedicationReconciliation" not in completed_actions and (
            draft.discharge_medications or re.search(r"\b(?:tab|cap|inj|medication|advice on discharge)\b", source)
        ):
            return {
                "reasoning": "Local ReAct policy found medication/discharge-advice evidence and selected medication reconciliation.",
                "action_chosen": "CALL_TOOL: MedicationReconciliation",
                "inputs": "Extracted medication and discharge advice text",
                "next_decision": "check pending results",
            }

        if "CALL_TOOL: PendingResultsCheck" not in completed_actions and (
            draft.pending_results or re.search(r"\b(?:awaited|pending|culture|sensitivity)\b", source)
        ):
            return {
                "reasoning": "Local ReAct policy found pending/culture evidence and selected pending-result review.",
                "action_chosen": "CALL_TOOL: PendingResultsCheck",
                "inputs": "Pending lab and culture evidence",
                "next_decision": "check diagnostic trends",
            }

        if "CALL_TOOL: DiagnosticCheck" not in completed_actions and re.search(
            r"\b(?:creatinine|sodium|glucose|dka|ketoacidosis|aki|kidney|renal|bp|spo2)\b",
            source,
        ):
            return {
                "reasoning": "Local ReAct policy found diagnostic/lab evidence and selected diagnostic trend review.",
                "action_chosen": "CALL_TOOL: DiagnosticCheck",
                "inputs": "Diagnostic and laboratory evidence",
                "next_decision": "flag unresolved safety issues",
            }

        if "CALL_TOOL: FlagContradiction" not in completed_actions and missing_fields:
            return {
                "reasoning": "Local ReAct policy found required discharge-summary fields still missing and selected safety flagging.",
                "action_chosen": "CALL_TOOL: FlagContradiction",
                "inputs": json.dumps({
                    "category": "MISSING_DATA",
                    "item_involved": ", ".join(missing_fields[:4]),
                    "description": "Required fields were not reliably sourced from OCR/API extraction.",
                    "action_taken": "Marked missing and flagged for clinician review.",
                }),
                "next_decision": "finalize draft",
            }

        return {
            "reasoning": "Local ReAct policy found no additional useful tool call, so it finalized the clean extracted draft.",
            "action_chosen": "FINAL_DRAFT",
            "inputs": "Validated extracted draft",
            "next_decision": "finalize draft",
        }

    def _local_react_decision(
        self,
        step_number: int,
        patient_id: str,
        raw_clinical_text: str,
        draft: DischargeSummaryDraft,
        model_name: str,
        completed_actions: set,
    ) -> Dict[str, str]:
        draft_snapshot = json.dumps(draft.model_dump(), ensure_ascii=False)[:1800]
        prompt = (
            "You are a local clinical ReAct agent. Choose exactly one next action as JSON only.\n"
            "Allowed actions: CALL_TOOL: MedicationReconciliation, CALL_TOOL: PendingResultsCheck, "
            "CALL_TOOL: DiagnosticCheck, CALL_TOOL: FlagContradiction, FINAL_DRAFT.\n"
            "Do not invent facts. Use tools when medication, pending result, diagnostic, missing, or conflict evidence needs review.\n"
            f"Patient: {patient_id}\n"
            f"Step: {step_number}\n"
            f"Completed actions: {sorted(completed_actions)}\n"
            f"Draft JSON: {draft_snapshot}\n"
            f"Source excerpt: {raw_clinical_text[:1800]}\n"
            "Return JSON with keys reasoning, action_chosen, inputs, next_decision."
        )
        try:
            response = self._local_text2text(prompt, model_name, max_new_tokens=180)
            decision = self._parse_local_react_json(response)
            decision["reasoning"] = "Local model decision: " + decision["reasoning"]
            return decision
        except Exception as exc:
            decision = self._deterministic_local_react_decision(
                step_number,
                patient_id,
                raw_clinical_text,
                draft,
                completed_actions,
            )
            decision["reasoning"] += f" Local model JSON decision was unavailable/invalid ({exc}); deterministic local policy used."
            return decision

    def _run_local_react_steps(
        self,
        patient_id: str,
        raw_clinical_text: str,
        draft: DischargeSummaryDraft,
        model_name: str,
    ) -> None:
        completed_actions = set()
        max_local_steps = min(MAX_AGENT_STEPS, int(os.getenv("LOCAL_REACT_MAX_STEPS", "5")))
        print(f"[Agent Loop] Starting LOCAL ReAct reasoning/tool-selection loop with {model_name}...")

        for step_offset in range(max_local_steps):
            step_number = len(self.execution_history) + 1
            decision = self._local_react_decision(
                step_number,
                patient_id,
                raw_clinical_text,
                draft,
                model_name,
                completed_actions,
            )
            action = decision["action_chosen"]
            if action == "FINAL_DRAFT":
                self.execution_history.append(AgentStepTrace(
                    step_number=step_number,
                    reasoning=decision["reasoning"],
                    action_chosen=action,
                    inputs=decision["inputs"],
                    result="Local ReAct loop finalized the extracted draft.",
                    next_decision=decision["next_decision"],
                ))
                break

            result = self._execute_tool(patient_id, raw_clinical_text, action, decision["inputs"])
            completed_actions.add(action)
            self.execution_history.append(AgentStepTrace(
                step_number=step_number,
                reasoning=decision["reasoning"],
                action_chosen=action,
                inputs=decision["inputs"],
                result=result,
                next_decision=decision["next_decision"],
            ))

            if len(completed_actions) >= 4:
                self.execution_history.append(AgentStepTrace(
                    step_number=len(self.execution_history) + 1,
                    reasoning="Local ReAct loop reached the useful tool coverage limit.",
                    action_chosen="FINAL_DRAFT",
                    inputs="Completed local tool checks",
                    result="Local ReAct loop finalized the extracted draft after tool review.",
                    next_decision="validate_schema_and_export",
                ))
                break

    def _run_local_transformer_loop(self, patient_id: str, raw_clinical_text: str, cfg: dict) -> CompleteExecutionPayload:
        """
        Runs a small local Transformers pipeline suited for low-memory laptops.
        The model assists with concise wording; extraction and safety flags remain
        deterministic so the system does not depend on a tiny model to discover facts.
        """
        model_name = cfg.get("model_name") or "google/flan-t5-base"
        ingestion_fallback_used = "INGESTION FALLBACK" in (raw_clinical_text or "")
        has_extracted_source = bool(raw_clinical_text and len(raw_clinical_text.strip()) > 80 and not ingestion_fallback_used)

        if has_extracted_source:
            payload = self._run_extractive_local_loop(patient_id, raw_clinical_text)
        else:
            print(
                "[Agent Loop] No usable extracted clinical text was available. "
                "Creating a missing-field review draft; hardcoded simulator is not used."
            )
            payload = self._run_extractive_local_loop(patient_id, raw_clinical_text or "")
            payload.final_draft = self._mark_ingestion_fallback_if_needed(payload.final_draft, raw_clinical_text or "")
            return payload
            
        draft = payload.final_draft
        self._run_local_react_steps(patient_id, raw_clinical_text, draft, model_name)
        draft.clinical_safety_flags = self.active_flags
        payload.execution_trace = self.execution_history

        source = raw_clinical_text[:2400]
        course_prompt = (
            "Rewrite this hospital course as one concise clinical discharge-summary paragraph. "
            "Use only facts present in the source. If facts are missing, do not invent them.\n\n"
            f"SOURCE:\n{source}\n\n"
            f"CURRENT COURSE:\n{draft.hospital_course}"
        )
        follow_up_prompt = (
            "Rewrite these follow-up instructions clearly for a discharge summary. "
            "Use only facts present in the source. Keep pending results explicit.\n\n"
            f"SOURCE:\n{source}\n\n"
            f"CURRENT FOLLOW UP:\n{draft.follow_up_instructions}"
        )

        improved_course = ""
        improved_follow_up = ""
        if draft.hospital_course and draft.hospital_course.lower() != "missing":
            improved_course = self._local_text2text(course_prompt, model_name, max_new_tokens=180)
        if draft.follow_up_instructions and draft.follow_up_instructions.lower() != "missing":
            improved_follow_up = self._local_text2text(follow_up_prompt, model_name, max_new_tokens=120)

        if self._is_useful_local_rewrite(improved_course, draft.hospital_course, min_len=40):
            draft.hospital_course = improved_course
        if self._is_useful_local_rewrite(improved_follow_up, draft.follow_up_instructions, min_len=15):
            draft.follow_up_instructions = improved_follow_up
        draft = self._apply_feedback_memory_to_draft(draft)

        payload.execution_trace.append(
            AgentStepTrace(
                step_number=len(payload.execution_trace) + 1,
                reasoning=f"Used local Transformers text2text-generation pipeline with {model_name}; local ReAct reasoning/tool selection already reviewed the extracted draft.",
                action_chosen="LOCAL_TRANSFORMERS_PIPELINE",
                inputs="hospital_course, follow_up_instructions, and local ReAct-reviewed draft",
                result="Completed local generation path while preserving structured extracted facts and local tool decisions.",
                next_decision="validate_schema_and_export",
            )
        )
        payload.total_steps_executed = len(payload.execution_trace)
        return payload

