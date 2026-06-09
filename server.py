# server.py  FastAPI Backend for Clinical Discharge Summary Agent UI
import os
import sys
import json
import shutil
import base64
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.parser import ClinicalTextParser
from src.agent_loop import ClinicalAgentLoop
from src.doctor_sim import DoctorSimulator
from src.learning_engine import FeedbackLearningEngine
from src.models import DischargeSummaryDraft, AgentStepTrace, CompleteExecutionPayload

# App init
app = FastAPI(title="Clinical Discharge Summary Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state — persists for the lifetime of the server process
extracted_patients: Dict[str, str] = {}          # patient_name -> raw_text
pipeline_results: Dict[str, dict] = {}           # patient_name -> full pipeline result
learning_engine = FeedbackLearningEngine()
doctor = DoctorSimulator()

# Ensure output dirs exist
os.makedirs("output/drafts", exist_ok=True)
os.makedirs("output/traces", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)

# Request / response models

class RunAgentRequest(BaseModel):
    patient_name: str
    raw_text: str
    feedback_memory: List[str] = []

class RunDoctorRequest(BaseModel):
    draft: dict

class RunLearningRequest(BaseModel):
    patient_name: str
    draft_diagnosis: str
    draft_followup: str
    edited_diagnosis: str
    edited_followup: str
    draft: dict
    edited: dict

class SetApiKeyRequest(BaseModel):
    api_key: str

# API endpoints

@app.post("/api/set-api-key")
async def set_api_key(req: SetApiKeyRequest):
    """Store the user-supplied API key in the process environment for this session."""
    key = (req.api_key or "").strip()
    if not key or len(key) < 8:
        raise HTTPException(status_code=400, detail="API key is too short or empty.")
    os.environ["LLM_API_KEY"] = key
    # Also set the canonical aliases so any provider check picks it up
    if key.startswith("sk-ant-"):
        os.environ["ANTHROPIC_API_KEY"] = key
    elif key.startswith("AIzaSy") or key.startswith("AQ"):
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GOOGLE_API_KEY"] = key
    elif key.startswith("sk-"):
        os.environ["OPENAI_API_KEY"] = key
    elif key.startswith("gsk_"):
        os.environ["GROQ_API_KEY"] = key
    return {"status": "success", "message": "API key saved for this session."}


@app.get("/api/config-status")
async def get_config_status():
    """Return current LLM provider configuration status."""
    from config.settings import get_llm_config
    try:
        cfg = get_llm_config()
        provider = cfg.get("provider", "local_transformers")
        model    = cfg.get("model_name", "unknown")
        has_key  = cfg.get("api_key") is not None
        return {
            "has_key":  has_key,
            "provider": provider,
            "model":    model,
            "is_live":  cfg.get("is_live", True),
        }
    except Exception as e:
        return {"has_key": False, "provider": "unknown", "model": "unknown", "is_live": False}


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a clinical PDF and extract patient records."""
    global extracted_patients

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save uploaded file temporarily
    temp_path = f"data/raw_patients/_uploaded_{file.filename}"
    os.makedirs("data/raw_patients", exist_ok=True)
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        parser = ClinicalTextParser()
        from starlette.concurrency import run_in_threadpool
        patient_records = await run_in_threadpool(parser.parse_patient_pdf, temp_path)
        extracted_patients = patient_records

        # Build response with preview
        patients_out = {}
        for name, text in patient_records.items():
            patients_out[name] = {
                "preview": text[:500] + ("..." if len(text) > 500 else ""),
                "full_length": len(text)
            }

        return {"status": "success", "patients": patients_out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/run-agent")
async def run_agent(req: RunAgentRequest):
    """Run the ReAct agent loop for a single patient."""
    try:
        agent = ClinicalAgentLoop(feedback_memory=req.feedback_memory)
        from starlette.concurrency import run_in_threadpool
        payload = await run_in_threadpool(agent.run, req.patient_name, req.raw_text)
        return payload.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.post("/api/run-doctor-review")
async def run_doctor_review(req: RunDoctorRequest):
    """Apply the simulated doctor review policy to a draft."""
    try:
        from starlette.concurrency import run_in_threadpool
        draft = DischargeSummaryDraft.model_validate(req.draft)
        edited = await run_in_threadpool(doctor.apply_hidden_doctor_policy, draft)
        return edited.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Doctor review failed: {str(e)}")


@app.post("/api/run-learning")
async def run_learning(req: RunLearningRequest):
    """Register iteration performance and extract feedback rules."""
    try:
        def execute_learning_sync():
            draft_text = f"{req.draft_diagnosis} | {req.draft_followup}"
            edited_text = f"{req.edited_diagnosis} | {req.edited_followup}"

            learning_engine.register_iteration_performance(req.patient_name, draft_text, edited_text)

            draft_obj = DischargeSummaryDraft.model_validate(req.draft)
            edited_obj = DischargeSummaryDraft.model_validate(req.edited)
            new_rules = learning_engine.extract_feedback_rules(draft_obj, edited_obj)

            # Get the latest distance for this patient
            distances = learning_engine.performance_history.get(req.patient_name, [])
            latest_distance = distances[-1] if distances else None

            return {
                "edit_distance": latest_distance,
                "new_rules": new_rules,
                "correction_memory": learning_engine.correction_memory,
                "all_distances": dict(learning_engine.performance_history)
            }

        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(execute_learning_sync)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning step failed: {str(e)}")


@app.get("/api/run-full-pipeline")
async def run_full_pipeline(patient_name: str = Query(...)):
    """Run the complete 3-iteration pipeline for one patient."""
    global pipeline_results

    raw_text = extracted_patients.get(patient_name)
    if not raw_text:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_name}' not found. Upload a PDF first.")

    try:
        def execute_pipeline_sync():
            local_engine = FeedbackLearningEngine()
            local_doctor = DoctorSimulator()
            iterations = []

            for iteration_num in range(1, 4):
                agent = ClinicalAgentLoop(feedback_memory=local_engine.correction_memory)
                payload = agent.run(patient_id=patient_name, raw_clinical_text=raw_text)
                draft = payload.final_draft
                edited = local_doctor.apply_hidden_doctor_policy(draft)

                str_d = f"{draft.principal_diagnosis} | {draft.follow_up_instructions}"
                str_e = f"{edited.principal_diagnosis} | {edited.follow_up_instructions}"
                local_engine.register_iteration_performance(patient_name, str_d, str_e)

                distances = local_engine.performance_history.get(patient_name, [])

                if iteration_num == 1:
                    local_engine.extract_feedback_rules(draft, edited)

                iterations.append({
                    "iteration": iteration_num,
                    "draft": draft.model_dump(),
                    "edited": edited.model_dump(),
                    "edit_distance": distances[-1] if distances else 0,
                    "trace": [t.model_dump() for t in payload.execution_trace],
                    "total_steps": payload.total_steps_executed,
                    "correction_memory": list(local_engine.correction_memory)
                })

            # Save outputs
            patient_slug = patient_name.replace(" ", "_")
            draft_path = f"output/drafts/{patient_slug}_draft.json"
            with open(draft_path, "w") as f:
                json.dump(iterations[-1]["draft"], f, indent=4)

            trace_path = f"output/traces/{patient_slug}_trace.json"
            with open(trace_path, "w") as f:
                json.dump(iterations[-1]["trace"], f, indent=4)

            return {
                "patient_name": patient_name,
                "iterations": iterations,
                "final_correction_memory": list(local_engine.correction_memory),
                "learning_data": dict(local_engine.performance_history)
            }

        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(execute_pipeline_sync)
        pipeline_results[patient_name] = result
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@app.get("/api/drafts")
async def list_drafts():
    """List all saved discharge summary drafts."""
    drafts_dir = "output/drafts"
    results = []
    if os.path.isdir(drafts_dir):
        for fname in sorted(os.listdir(drafts_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(drafts_dir, fname)
                with open(fpath, "r") as f:
                    data = json.load(f)
                results.append({"filename": fname, "data": data})
    return {"drafts": results}


@app.get("/api/traces")
async def list_traces():
    """List all saved execution traces."""
    traces_dir = "output/traces"
    results = []
    if os.path.isdir(traces_dir):
        for fname in sorted(os.listdir(traces_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(traces_dir, fname)
                with open(fpath, "r") as f:
                    data = json.load(f)
                results.append({"filename": fname, "data": data})
    return {"traces": results}


@app.get("/api/learning-curve")
async def get_learning_curve():
    """Return learning curve data for Chart.js rendering."""
    all_data = {}
    for name, result in pipeline_results.items():
        all_data[name] = [it["edit_distance"] for it in result["iterations"]]

    if not all_data and learning_engine.performance_history:
        all_data = dict(learning_engine.performance_history)

    return {"learning_data": all_data}


# Serve frontend — must be last so it doesn't intercept API routes
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

# Run directly
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  CLINICAL DISCHARGE SUMMARY AGENT  WEB INTERFACE")
    print("  Open your browser at: http://localhost:8000")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
