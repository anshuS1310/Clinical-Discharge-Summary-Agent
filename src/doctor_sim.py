# src/doctor_sim.py
# Simulates a clinician reviewing and editing the AI-generated discharge draft.
# The "hidden policy" here mimics the formatting preference a real doctor might apply.
# The edits are what the learning engine measures and learns from across iterations.

import os
import json
from src.models import DischargeSummaryDraft
from config.settings import get_llm_config

class DoctorSimulator:
    """
    Simulates a clinician reviewer who applies a hidden editing policy
    to every discharge draft the agent produces.
    Uses an LLM to dynamically apply the style rules when online, keeping it completely generalizable.
    """
    def __init__(self):
        # The clinician's hidden rules
        self.rules = [
            "Always append ' [Clinically Verified via Chief of Staff]' to the principal_diagnosis.",
            "Always prepend 'ATTENTION PATIENT: ' to the follow_up_instructions."
        ]

    def apply_hidden_doctor_policy(self, draft: DischargeSummaryDraft) -> DischargeSummaryDraft:
        """
        Applies the hidden clinician style rules to the draft.
        Uses the LLM if a live key is present, otherwise falls back to deterministic rule execution.
        """
        cfg = get_llm_config()
        if cfg.get("is_live", False) and cfg.get("provider") != "local_transformers":
            try:
                # Ask LLM to act as Doctor and apply the style rules
                rules_str = "\n".join(f"- {r}" for r in self.rules)
                prompt = (
                    "You are a senior supervising physician reviewing this clinical discharge summary draft.\n"
                    "Apply the following style and policy guidelines to edit the draft:\n"
                    f"{rules_str}\n\n"
                    "DRAFT SUMMARY:\n"
                    f"{draft.model_dump_json(indent=2)}\n\n"
                    "Return the edited draft as a raw JSON matching the DischargeSummaryDraft schema (no backticks, no other text):"
                )
                
                res = self._call_llm_api_direct(prompt, cfg)
                if res["status"] == "SUCCESS":
                    start_idx = res["content"].find('{')
                    end_idx = res["content"].rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        return DischargeSummaryDraft.model_validate_json(res["content"][start_idx:end_idx + 1])
            except Exception as e:
                print(f"[Doctor Simulator] LLM editing failed: {e}. Falling back to rule-based edits.")

        # Fallback / Offline path: Deterministically apply the rules to the draft
        edited = draft.model_copy(deep=True)
        suffix = " [Clinically Verified via Chief of Staff]"
        if (
            edited.principal_diagnosis
            and edited.principal_diagnosis.lower() != "missing"
            and not edited.principal_diagnosis.endswith(suffix)
        ):
            edited.principal_diagnosis += suffix

        prefix = "ATTENTION PATIENT: "
        if (
            edited.follow_up_instructions
            and edited.follow_up_instructions.lower() != "missing"
            and not edited.follow_up_instructions.startswith(prefix)
        ):
            edited.follow_up_instructions = prefix + edited.follow_up_instructions

        return edited

    def _call_llm_api_direct(self, prompt: str, cfg: dict) -> dict:
        from src.agent_loop import ClinicalAgentLoop
        agent = ClinicalAgentLoop()
        return agent._call_llm_api_direct(prompt, cfg)
