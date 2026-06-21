# Clinical Discharge Summary Agent

Read a patient's clinical PDF records and produce a structured, clinician-ready discharge summary — with medication reconciliation, pending-result flags, and a self-improving feedback loop. The agent never invents clinical facts; any field it cannot source from the documents is explicitly marked `missing` and flagged for clinician review.

## Base URL
https://clinical-discharge-summary-agent-production.up.railway.app

## Endpoints

---

### POST /api/set-api-key
Inject a cloud LLM API key for this session so the agent uses a live model instead of the offline local transformer. Supports OpenAI (`sk-`), Anthropic (`sk-ant-`), Google Gemini (`AIzaSy`), and Groq (`gsk_`). The key is stored only in the server process — not persisted to disk.

Example:
  curl -X POST "https://clinical-discharge-summary-agent-production.up.railway.app/api/set-api-key" \
    -H "Content-Type: application/json" \
    -d '{"api_key": "AIzaSy..."}'

Response:
  { "status": "success", "message": "API key saved for this session." }

---

### GET /api/config-status
Check which LLM provider and model are currently active, and whether a live API key is configured.

Example:
  curl "https://clinical-discharge-summary-agent-production.up.railway.app/api/config-status"

Response:
  { "has_key": false, "provider": "local_transformers", "model": "google/flan-t5-base", "is_live": true }

---

### POST /api/upload-pdf
Upload a clinical PDF (text-layer or scanned/handwritten). The server automatically detects patients in the document, segments pages by patient, runs multi-engine OCR on image pages, and returns the extracted text per patient ready for the agent.

Example:
  curl -X POST "https://clinical-discharge-summary-agent-production.up.railway.app/api/upload-pdf" \
    -F "file=@patient_record.pdf"

Response:
  {
    "status": "success",
    "patients": {
      "John Smith": {
        "preview": "Admitted: 12-Jan-2025. Diagnosis: Acute...",
        "full_length": 4821
      }
    }
  }

---

### GET /api/run-full-pipeline?patient_name={name}
Run the complete 3-iteration pipeline for one extracted patient:
  1. ReAct agent loop -> structured discharge draft
  2. Simulated doctor review -> edit distance measured
  3. Correction rules extracted -> injected into next iteration

The patient must have been uploaded via /api/upload-pdf first. URL-encode spaces (%20).

Example:
  curl "https://clinical-discharge-summary-agent-production.up.railway.app/api/run-full-pipeline?patient_name=John%20Smith"

Response (abbreviated):
  {
    "patient_name": "John Smith",
    "iterations": [
      {
        "iteration": 1,
        "draft": {
          "patient_name": "John Smith",
          "principal_diagnosis": "Acute Kidney Injury (Stage 2)",
          "medications_on_discharge": ["Furosemide 40mg OD", "Amlodipine 5mg OD"],
          "pending_results": ["Urine culture — result pending at discharge"],
          "clinical_safety_flags": ["Creatinine trending up — monitor closely"],
          "follow_up_instructions": "Nephrology review in 1 week. Repeat renal panel.",
          "missing_fields": []
        },
        "edit_distance": 0.38,
        "total_steps": 8,
        "correction_memory": ["Always include ICD-10 code after diagnosis"]
      },
      { "iteration": 2, "edit_distance": 0.0 },
      { "iteration": 3, "edit_distance": 0.0 }
    ],
    "final_correction_memory": ["Always include ICD-10 code after diagnosis"]
  }

---

### POST /api/run-agent
Run a single pass of the ReAct agent loop for one patient (no doctor review or learning). Useful for testing extraction on raw text.

Example:
  curl -X POST "https://clinical-discharge-summary-agent-production.up.railway.app/api/run-agent" \
    -H "Content-Type: application/json" \
    -d '{
      "patient_name": "John Smith",
      "raw_text": "Admitted 12-Jan-2025 with AKI. BP 145/90. Creatinine 3.2 mg/dL...",
      "feedback_memory": []
    }'

Response:
  {
    "patient_id": "John Smith",
    "final_draft": { "principal_diagnosis": "Acute Kidney Injury", "..." : "..." },
    "execution_trace": [
      { "step": 1, "reasoning": "Check medication reconciliation first.", "action": "MedicationReconciliation", "result": "No omissions detected." }
    ],
    "total_steps_executed": 6
  }

---

### POST /api/run-doctor-review
Apply the simulated clinician review policy to an existing draft. Returns the edited draft with style corrections applied.

Example:
  curl -X POST "https://clinical-discharge-summary-agent-production.up.railway.app/api/run-doctor-review" \
    -H "Content-Type: application/json" \
    -d '{"draft": { "patient_name": "John Smith", "principal_diagnosis": "AKI", "..." : "..." }}'

Response: the edited DischargeSummaryDraft object.

---

### POST /api/run-learning
Register one iteration's edit distance and extract correction rules from the draft vs. doctor-edited diff.

Example:
  curl -X POST "https://clinical-discharge-summary-agent-production.up.railway.app/api/run-learning" \
    -H "Content-Type: application/json" \
    -d '{
      "patient_name": "John Smith",
      "draft_diagnosis": "AKI",
      "draft_followup": "Follow up in 1 week.",
      "edited_diagnosis": "Acute Kidney Injury (N17.9)",
      "edited_followup": "Nephrology review in 7 days. Repeat renal function panel.",
      "draft": {},
      "edited": {}
    }'

Response:
  {
    "edit_distance": 0.385,
    "new_rules": ["Always append ICD-10 code in parentheses after the diagnosis name."],
    "correction_memory": ["Always append ICD-10 code in parentheses after the diagnosis name."],
    "all_distances": { "John Smith": [0.385] }
  }

---

### GET /api/drafts
List all discharge summary drafts saved to disk from previous full-pipeline runs.

Example:
  curl "https://clinical-discharge-summary-agent-production.up.railway.app/api/drafts"

Response:
  {
    "drafts": [
      { "filename": "John_Smith_draft.json", "data": { "principal_diagnosis": "..." } }
    ]
  }

---

### GET /api/traces
List all saved ReAct agent execution traces (full step-by-step reasoning logs).

Example:
  curl "https://clinical-discharge-summary-agent-production.up.railway.app/api/traces"

Response:
  { "traces": [ { "filename": "John_Smith_trace.json", "data": [ ... ] } ] }

---

### GET /api/learning-curve
Return the edit-distance learning curve data for all patients — ready to render on a chart.

Example:
  curl "https://clinical-discharge-summary-agent-production.up.railway.app/api/learning-curve"

Response:
  { "learning_data": { "John Smith": [0.385, 0.0, 0.0] } }

---

## How the agent should use this

1. Configure (optional): Call POST /api/set-api-key with a Gemini or OpenAI key to enable the live reasoning model. Without a key the system runs fully offline using google/flan-t5-base.

2. Ingest the PDF: Call POST /api/upload-pdf with the clinical PDF. The system handles text PDFs, scanned documents, and handwritten notes. The returned "patients" object maps each detected patient name to a text preview and character count.

3. Run the full pipeline: For each patient name returned, call GET /api/run-full-pipeline?patient_name={name}. The agent will:
   - Run a bounded ReAct reasoning loop (up to 10 steps) calling MedicationReconciliation, PendingResultsCheck, DiagnosticCheck, and FlagContradiction tools
   - Produce a structured discharge draft with all missing fields marked explicitly as "missing"
   - Simulate doctor review and measure the Normalized Levenshtein edit distance
   - Extract correction rules and re-run for 3 iterations until the draft converges

4. Read the result: From the pipeline response, read iterations[-1].draft for the final discharge summary. Check clinical_safety_flags for safety concerns. Check missing_fields for data the agent could not find in the source documents — these must be completed by the clinician.

5. Track improvement: Call GET /api/learning-curve to get edit-distance values across iterations. A curve converging 0.38 -> 0.0 -> 0.0 confirms the agent learned the clinician's style.

6. Retrieve saved outputs: Call GET /api/drafts and GET /api/traces to retrieve persisted discharge summaries and full reasoning traces for audit or compliance purposes.

No-fabrication guarantee: Every field the agent cannot source from the uploaded documents is set to "missing" or "undocumented" and listed in missing_fields. The agent never invents clinical facts. The output is always a draft for clinician review, never an auto-finalized document.
