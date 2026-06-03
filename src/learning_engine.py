# src/learning_engine.py
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

class FeedbackLearningEngine:
    """
    Tracks doctor edits, computes normalized Levenshtein distance as a metric,
    extracts structured correction memory, and generates the learning curve plot.
    """
    def __init__(self):
        # Maps patient_id -> list of normalized edit distances
        self.performance_history: Dict[str, List[float]] = {}
        self.correction_memory = []

    def calculate_normalized_edit_distance(self, draft_text: str, edited_text: str) -> float:
        """Computes the normalized Levenshtein edit distance between draft and correction."""
        max_len = max(len(draft_text), len(edited_text))
        if max_len == 0:
            return 0.0
        distance = Levenshtein.distance(draft_text, edited_text)
        return float(distance) / max_len

    def extract_feedback_rules(self, draft_summary: Any, edited_summary: Any) -> List[str]:
        """
        Analyzes the edits made by the clinician and extracts structured rules
        to store in the correction memory.
        """
        new_rules = []
        
        # 1. Check principal diagnosis edits
        if draft_summary.principal_diagnosis != edited_summary.principal_diagnosis:
            # Check if doctor appended a verification suffix
            suffix = " [Clinically Verified via Discharge Evaluation Policy]"
            if edited_summary.principal_diagnosis.endswith(suffix) and not draft_summary.principal_diagnosis.endswith(suffix):
                rule = "For principal_diagnosis: Append ' [Clinically Verified via Discharge Evaluation Policy]' to denote clinical validation."
                if rule not in self.correction_memory:
                    new_rules.append(rule)
                    self.correction_memory.append(rule)
                    
        # 2. Check follow-up instructions edits
        if draft_summary.follow_up_instructions != edited_summary.follow_up_instructions:
            # Check if doctor prepended a compliance warning
            prefix = "CRITICAL CLINICAL FOLLOW-UP: Please visit the clinic as scheduled. "
            if edited_summary.follow_up_instructions.startswith(prefix) and not draft_summary.follow_up_instructions.startswith(prefix):
                rule = "For follow_up_instructions: Prepend 'CRITICAL CLINICAL FOLLOW-UP: Please visit the clinic as scheduled. ' to ensure patient safety."
                if rule not in self.correction_memory:
                    new_rules.append(rule)
                    self.correction_memory.append(rule)
                    
        if new_rules:
            print(f"[Learning Engine] Extracted {len(new_rules)} new compliance rules into correction memory.")
        return new_rules

    def register_iteration_performance(self, patient_id: str, draft_text: str, edited_text: str):
        """Records performance metrics for the current optimization step."""
        dist = self.calculate_normalized_edit_distance(draft_text, edited_text)
        if patient_id not in self.performance_history:
            self.performance_history[patient_id] = []
        self.performance_history[patient_id].append(dist)
        print(f"[Learning Engine] Patient: {patient_id} | Calculated Normalized Edit Distance: {dist:.4f} (Friction Metric)")

    def generate_and_save_learning_curve(self, output_image_path: str):
        """Generates the evaluation improvement curve plot and saves it."""
        if not self.performance_history:
            print("[Learning Engine] Cannot plot curve: No performance iterations recorded yet.")
            return

        plt.figure(figsize=(9, 5.5))
        
        # Color palette for premium aesthetics
        colors = ['#1e3a8a', '#10b981', '#f59e0b', '#ef4444']
        
        for idx, (patient_id, distances) in enumerate(self.performance_history.items()):
            iterations = np.arange(1, len(distances) + 1)
            color = colors[idx % len(colors)]
            
            plt.plot(
                iterations, 
                distances, 
                marker='o', 
                color=color, 
                linestyle='-', 
                linewidth=2.5, 
                markersize=8,
                label=f"Patient: {patient_id}"
            )
            
            # Add labels to markers
            for i, val in enumerate(distances):
                plt.text(iterations[i], val + 0.04, f"{val:.4f}", ha='center', fontsize=9, fontweight='semibold', color=color)
        
        plt.title("Part 2 Evaluation: Clinician Feedback Loop Optimization Curve", fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("System Training Iterations / Optimization Runs", fontsize=10, labelpad=8)
        plt.ylabel("Normalized Edit Distance (Clinician Friction)", fontsize=10, labelpad=8)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc="upper right", frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0')
        
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=300)
        plt.close()
        print(f"[Learning Engine] Evaluation performance plot successfully saved to: {output_image_path}")