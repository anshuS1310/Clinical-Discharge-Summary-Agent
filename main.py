# main.py
import os
import sys
import json
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

# Ensure dotenv override runs at very startup
load_dotenv(override=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parser import ClinicalTextParser
from src.agent_loop import ClinicalAgentLoop
from src.doctor_sim import DoctorSimulator
from src.learning_engine import FeedbackLearningEngine

def main():
    # Parse CLI Arguments for flexible configuration
    parser_arg = argparse.ArgumentParser(description="Clinical Discharge Summary Agent Orchestrator")
    parser_arg.add_argument(
        "--api-key", "-k",
        type=str,
        default=None,
        help="Optional LLM API Key to run in live mode. Overrides environmental keys."
    )
    args = parser_arg.parse_args()
    cli_key = args.api_key

    print("======================================================================")
    print("      LAUNCHING CLINICAL DISCHARGE SUMMARY AGENT WORKSPACE            ")
    print("======================================================================")
    
    # 1. Initialize Directories
    os.makedirs("output/drafts", exist_ok=True)
    os.makedirs("output/traces", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)
    
    pdf_path = "data/raw_patients/patient 2.pdf"
    
    # 2. Parsing and Ingestion Phase
    parser = ClinicalTextParser()
    patient_records = parser.parse_patient_pdf(pdf_path)
    
    doctor = DoctorSimulator()
    learning_engine = FeedbackLearningEngine()
    
    # 3. Process Patients through the Agent-Doctor Loop
    for patient_name, raw_text in patient_records.items():
        print(f"\n==============================================================")
        print(f" PROCESSING PATIENT: {patient_name.upper()} ")
        print(f"==============================================================")
        
        # Iteration 1: Raw Agent Draft Generation (Baseline)
        print("\n--- [Optimization Run 1: Baseline Generation] ---")
        agent_run_1 = ClinicalAgentLoop(feedback_memory=[], cli_api_key=cli_key)
        payload_1 = agent_run_1.run(patient_id=patient_name, raw_clinical_text=raw_text)
        draft_1 = payload_1.final_draft
        
        # Clinician Review
        edited_1 = doctor.apply_hidden_doctor_policy(draft_1)
        
        # Calculate Edit Distance
        str_d1 = f"{draft_1.principal_diagnosis} | {draft_1.follow_up_instructions}"
        str_e1 = f"{edited_1.principal_diagnosis} | {edited_1.follow_up_instructions}"
        learning_engine.register_iteration_performance(patient_name, str_d1, str_e1)
        
        # Extract compliance adjustments into structured correction memory
        new_rules = learning_engine.extract_feedback_rules(draft_1, edited_1)
        print(f"Rules currently in correction memory: {learning_engine.correction_memory}")
        
        # Iteration 2: Learning Applied (Feedback Injected)
        print("\n--- [Optimization Run 2: Feedback-Injected Generation] ---")
        agent_run_2 = ClinicalAgentLoop(feedback_memory=learning_engine.correction_memory, cli_api_key=cli_key)
        payload_2 = agent_run_2.run(patient_id=patient_name, raw_clinical_text=raw_text)
        draft_2 = payload_2.final_draft
        
        # Clinician Review (2nd Round)
        edited_2 = doctor.apply_hidden_doctor_policy(draft_2)
        
        # Calculate Edit Distance
        str_d2 = f"{draft_2.principal_diagnosis} | {draft_2.follow_up_instructions}"
        str_e2 = f"{edited_2.principal_diagnosis} | {edited_2.follow_up_instructions}"
        learning_engine.register_iteration_performance(patient_name, str_d2, str_e2)
        
        # Iteration 3: Full Alignment Run
        print("\n--- [Optimization Run 3: Fully Aligned State] ---")
        agent_run_3 = ClinicalAgentLoop(feedback_memory=learning_engine.correction_memory, cli_api_key=cli_key)
        payload_3 = agent_run_3.run(patient_id=patient_name, raw_clinical_text=raw_text)
        draft_3 = payload_3.final_draft
        
        # Clinician Review (3rd Round)
        edited_3 = doctor.apply_hidden_doctor_policy(draft_3)
        
        # Calculate Edit Distance
        str_d3 = f"{draft_3.principal_diagnosis} | {draft_3.follow_up_instructions}"
        str_e3 = f"{edited_3.principal_diagnosis} | {edited_3.follow_up_instructions}"
        learning_engine.register_iteration_performance(patient_name, str_d3, str_e3)
        
        # Save structured outputs for this patient
        patient_slug = patient_name.replace(" ", "_")
        
        # Save final draft
        draft_out_path = f"output/drafts/{patient_slug}_draft.json"
        with open(draft_out_path, "w") as f:
            json.dump(payload_3.final_draft.model_dump(), f, indent=4)
        print(f"[Exporter] Saved final discharge draft to: {draft_out_path}")
        
        # Save execution trace
        trace_out_path = f"output/traces/{patient_slug}_trace.json"
        # Convert traces list of objects to list of dicts
        traces_dict = [trace.model_dump() for trace in payload_3.execution_trace]
        with open(trace_out_path, "w") as f:
            json.dump(traces_dict, f, indent=4)
        print(f"[Exporter] Saved step execution trace to: {trace_out_path}")

    # 4. Generate and save learning curve plot
    plot_output_path = "output/plots/learning_curve.png"
    learning_engine.generate_and_save_learning_curve(plot_output_path)
    
    print("\n======================================================================")
    print("      EXECUTION COMPLETED SUCCESSFULLY - ALL ARTIFACTS GENERATED     ")
    print("======================================================================")

if __name__ == "__main__":
    main()