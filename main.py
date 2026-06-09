# main.py — CLI orchestrator for the Clinical Discharge Summary Agent
# Use this to run the full pipeline without the web server.
# For the web interface, run: python -m uvicorn server:app --port 8000 --reload

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Load .env before any other imports so API keys are available immediately
load_dotenv(override=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parser import ClinicalTextParser
from src.agent_loop import ClinicalAgentLoop
from src.doctor_sim import DoctorSimulator
from src.learning_engine import FeedbackLearningEngine


def main():
    # ── CLI Arguments ─────────────────────────────────────────────────
    arg_parser = argparse.ArgumentParser(
        description="Clinical Discharge Summary Agent — CLI Orchestrator"
    )
    arg_parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="LLM API key. Overrides the key in .env. Supports OpenAI, Gemini, Anthropic, Groq, OpenRouter.",
    )
    args = arg_parser.parse_args()
    cli_key = args.api_key

    print("=" * 70)
    print("   ClinicalAI — Discharge Summary Agent")
    print("=" * 70)

    # ── Create output directories ──────────────────────────────────────
    os.makedirs("output/drafts", exist_ok=True)
    os.makedirs("output/traces", exist_ok=True)
    os.makedirs("output/plots",  exist_ok=True)

    # ── PDF path — update this to point to your clinical PDF ──────────
    pdf_path = "data/raw_patients/patient 2.pdf"

    # ── Step 1: Parse patient records from the PDF ────────────────────
    print(f"\n[1/4] Parsing patient records from: {pdf_path}")
    parser = ClinicalTextParser()
    patient_records = parser.parse_patient_pdf(pdf_path)

    if not patient_records:
        print(
            "\n[Error] No patient records were extracted from the PDF.\n"
            "        Check that the file exists and is readable."
        )
        return

    print(f"      Extracted {len(patient_records)} patient record(s): {list(patient_records.keys())}")

    # ── Step 2: Initialise shared components ──────────────────────────
    doctor          = DoctorSimulator()
    learning_engine = FeedbackLearningEngine()

    # ── Step 3: Run the 3-iteration agent-doctor loop per patient ─────
    print("\n[2/4] Running 3-iteration agent pipeline for each patient...\n")

    for patient_name, raw_text in patient_records.items():
        print(f"\n{'=' * 70}")
        print(f"  Patient: {patient_name.upper()}")
        print(f"{'=' * 70}")

        # --- Iteration 1: Baseline (no learned rules) ---
        print("\n  [Iteration 1/3] Baseline generation — no feedback yet")
        agent_1  = ClinicalAgentLoop(feedback_memory=[], cli_api_key=cli_key)
        payload_1 = agent_1.run(patient_id=patient_name, raw_clinical_text=raw_text)
        draft_1   = payload_1.final_draft
        edited_1  = doctor.apply_hidden_doctor_policy(draft_1)

        str_d1 = f"{draft_1.principal_diagnosis} | {draft_1.follow_up_instructions}"
        str_e1 = f"{edited_1.principal_diagnosis} | {edited_1.follow_up_instructions}"
        learning_engine.register_iteration_performance(patient_name, str_d1, str_e1)
        learning_engine.extract_feedback_rules(draft_1, edited_1)
        print(f"  Correction rules in memory: {len(learning_engine.correction_memory)}")

        # --- Iteration 2: Feedback-injected ---
        print("\n  [Iteration 2/3] Feedback-injected — applying learned rules")
        agent_2   = ClinicalAgentLoop(feedback_memory=learning_engine.correction_memory, cli_api_key=cli_key)
        payload_2 = agent_2.run(patient_id=patient_name, raw_clinical_text=raw_text)
        draft_2   = payload_2.final_draft
        edited_2  = doctor.apply_hidden_doctor_policy(draft_2)

        str_d2 = f"{draft_2.principal_diagnosis} | {draft_2.follow_up_instructions}"
        str_e2 = f"{edited_2.principal_diagnosis} | {edited_2.follow_up_instructions}"
        learning_engine.register_iteration_performance(patient_name, str_d2, str_e2)
        learning_engine.extract_feedback_rules(draft_2, edited_2)

        # --- Iteration 3: Final aligned run ---
        print("\n  [Iteration 3/3] Final alignment run")
        agent_3   = ClinicalAgentLoop(feedback_memory=learning_engine.correction_memory, cli_api_key=cli_key)
        payload_3 = agent_3.run(patient_id=patient_name, raw_clinical_text=raw_text)
        draft_3   = payload_3.final_draft
        edited_3  = doctor.apply_hidden_doctor_policy(draft_3)

        str_d3 = f"{draft_3.principal_diagnosis} | {draft_3.follow_up_instructions}"
        str_e3 = f"{edited_3.principal_diagnosis} | {edited_3.follow_up_instructions}"
        learning_engine.register_iteration_performance(patient_name, str_d3, str_e3)

        # --- Save outputs ---
        slug = patient_name.replace(" ", "_")

        draft_path = f"output/drafts/{slug}_draft.json"
        with open(draft_path, "w") as f:
            json.dump(payload_3.final_draft.model_dump(), f, indent=4)
        print(f"\n  [Saved] Discharge draft  → {draft_path}")

        trace_path = f"output/traces/{slug}_trace.json"
        with open(trace_path, "w") as f:
            json.dump([t.model_dump() for t in payload_3.execution_trace], f, indent=4)
        print(f"  [Saved] Execution trace  → {trace_path}")

    # ── Step 4: Generate learning curve plot ──────────────────────────
    print("\n[3/4] Generating learning curve chart...")
    plot_path = "output/plots/learning_curve.png"
    learning_engine.generate_and_save_learning_curve(plot_path)

    print("\n[4/4] All done.")
    print("=" * 70)
    print("  Discharge drafts : output/drafts/")
    print("  Execution traces : output/traces/")
    print("  Learning curve   : output/plots/learning_curve.png")
    print("=" * 70)
    print("\n  Tip: Run the web server to view results interactively:")
    print("       python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload")
    print()


if __name__ == "__main__":
    main()
