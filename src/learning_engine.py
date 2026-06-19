# src/learning_engine.py
# Tracks how well the agent's drafts align with the doctor's edits over
# successive iterations. Uses Levenshtein edit distance as the friction metric.
# Extracts structured correction rules from the diff and injects them into the
# agent's prompt for the next run. Also generates the learning curve chart.

import json
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from config.settings import get_llm_config

class FeedbackLearningEngine:
    """
    Measures and drives the iterative improvement loop.

    After each agent-doctor iteration, this engine:
      1. Computes the normalized edit distance between the AI draft and
         the doctor-edited version (0 = identical, 1 = completely different).
      2. Extracts structured correction rules from the observed edits.
      3. Stores those rules in memory so they can be injected into the
         agent's prompt on the next iteration.
      4. Generates a learning curve chart showing friction decreasing over runs.
    """

    def __init__(self):
        # patient_id -> list of normalized edit distances, one per iteration
        self.performance_history: Dict[str, List[float]] = {}
        # Structured rules extracted from clinician edits
        self.correction_memory: List[str] = []

    def calculate_normalized_edit_distance(self, draft_text: str, edited_text: str) -> float:
        """
        Computes normalized Levenshtein distance between the AI draft and
        the doctor's edited version. The result is in [0, 1]:
          0.0 = no changes at all (perfect alignment)
          1.0 = completely rewritten
        """
        max_len = max(len(draft_text), len(edited_text))
        if max_len == 0:
            return 0.0
        return float(Levenshtein.distance(draft_text, edited_text)) / max_len

    def extract_feedback_rules(self, draft_summary: Any, edited_summary: Any) -> List[str]:
        """
        Compares the agent's draft with the doctor's edits and extracts
        structured rules to remember. Uses the LLM when online to do a generalized
        rule extraction, and falls back to deterministic check when offline.
        """
        cfg = get_llm_config()
        if cfg.get("is_live", False) and cfg.get("provider") != "local_transformers":
            try:
                # Prompt the LLM to compare draft vs edited and output style rules
                prompt = (
                    "You are a clinical machine learning metadata extractor.\n"
                    "Compare the original AI discharge draft and the doctor's edited version below.\n"
                    "Extract the style, format, and clinical prefix/suffix guidelines the doctor consistently applied.\n"
                    "Guidelines must be written as actionable instructions for future drafts, e.g.:\n"
                    "- 'For principal_diagnosis: Always append [suffix]'\n"
                    "- 'For follow_up_instructions: Always prepend [prefix]'\n\n"
                    "ORIGINAL AI DRAFT:\n"
                    f"{draft_summary.model_dump_json(indent=2)}\n\n"
                    "DOCTOR EDITED DRAFT:\n"
                    f"{edited_summary.model_dump_json(indent=2)}\n\n"
                    "Return the list of instructions as a raw JSON array of strings (no other text, no backticks):"
                )
                
                res = self._call_llm_api_direct(prompt, cfg)
                if res["status"] == "SUCCESS":
                    content = res["content"].strip()
                    start_idx = content.find('[')
                    end_idx = content.rfind(']')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        rules = json.loads(content[start_idx:end_idx + 1])
                        new_rules = []
                        for rule in rules:
                            if rule not in self.correction_memory:
                                new_rules.append(rule)
                                self.correction_memory.append(rule)
                        if new_rules:
                            print(f"[Learning Engine] Extracted {len(new_rules)} new dynamic rule(s): {new_rules}")
                        return new_rules
            except Exception as e:
                print(f"[Learning Engine] LLM rule extraction failed: {e}. Falling back to rule-based parser.")

        # Fallback / Offline rule extraction
        new_rules = []
        
        # Check if diagnosis was modified
        if draft_summary.principal_diagnosis != edited_summary.principal_diagnosis:
            d_orig = draft_summary.principal_diagnosis
            d_edit = edited_summary.principal_diagnosis
            if d_edit.startswith(d_orig) and len(d_edit) > len(d_orig):
                suffix = d_edit[len(d_orig):]
                rule = f"For principal_diagnosis: Always append '{suffix}'"
                if rule not in self.correction_memory:
                    new_rules.append(rule)
                    self.correction_memory.append(rule)
                    
        # Check if follow-up was modified
        if draft_summary.follow_up_instructions != edited_summary.follow_up_instructions:
            f_orig = draft_summary.follow_up_instructions
            f_edit = edited_summary.follow_up_instructions
            if f_edit.endswith(f_orig) and len(f_edit) > len(f_orig):
                prefix = f_edit[:-len(f_orig)]
                rule = f"For follow_up_instructions: Always prepend '{prefix}'"
                if rule not in self.correction_memory:
                    new_rules.append(rule)
                    self.correction_memory.append(rule)

        if new_rules:
            print(f"[Learning Engine] Extracted {len(new_rules)} rule(s) via offline parser.")
        return new_rules

    def register_iteration_performance(self, patient_id: str, draft_text: str, edited_text: str):
        """
        Records the edit distance for this iteration. Call this once per
        agent-doctor round, in order (iteration 1, 2, 3).
        """
        distance = self.calculate_normalized_edit_distance(draft_text, edited_text)
        if patient_id not in self.performance_history:
            self.performance_history[patient_id] = []
        self.performance_history[patient_id].append(distance)
        print(
            f"[Learning Engine] {patient_id} | "
            f"Iteration {len(self.performance_history[patient_id])} | "
            f"Edit distance: {distance:.4f}"
            + ("  Perfect alignment" if distance == 0.0 else "")
        )

    def generate_and_save_learning_curve(self, output_image_path: str):
        """
        Generates and saves the learning curve chart showing how edit distance
        decreases across iterations for each patient.
        """
        if not self.performance_history:
            print("[Learning Engine] No iteration data recorded  skipping chart generation.")
            return

        plt.figure(figsize=(9, 5.5))
        colors = ["#1A6FBA", "#0D9488", "#D97706", "#DC2626"]

        for idx, (patient_id, distances) in enumerate(self.performance_history.items()):
            iterations = np.arange(1, len(distances) + 1)
            color = colors[idx % len(colors)]

            plt.plot(
                iterations,
                distances,
                marker="o",
                color=color,
                linestyle="-",
                linewidth=2.5,
                markersize=8,
                label=f"Patient: {patient_id}",
            )

            for i, val in enumerate(distances):
                plt.text(
                    iterations[i], val + 0.04, f"{val:.4f}",
                    ha="center", fontsize=9, fontweight="semibold", color=color,
                )

        plt.title("Clinician Feedback Loop  Edit Distance Over Iterations", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Optimization Iteration", fontsize=10, labelpad=8)
        plt.ylabel("Normalized Edit Distance (0 = perfect alignment)", fontsize=10, labelpad=8)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        print(f"[Learning Engine] Learning curve saved to: {output_image_path}")

    def _call_llm_api_direct(self, prompt: str, cfg: dict) -> dict:
        from src.agent_loop import ClinicalAgentLoop
        agent = ClinicalAgentLoop()
        return agent._call_llm_api_direct(prompt, cfg)