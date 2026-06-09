# Simulates a clinician reviewing and editing the AI-generated discharge draft.
# The "hidden policy" here mimics the kind of consistent formatting preference
# a real doctor might apply  a verification suffix on diagnoses and a
# compliance prefix on follow-up instructions. These edits are what the learning
# engine measures and learns from across iterations.

from src.models import DischargeSummaryDraft


class DoctorSimulator:
    """
    Simulates a clinician reviewer who applies a fixed, hidden editing policy
    to every discharge draft the agent produces.

    The agent doesn't know these rules upfront. It discovers them by observing
    the diff between its draft and the doctor's edited version, then learns to
    apply them autonomously by iteration 2.
    """

    def __init__(self):
        # The doctor always wants the principal diagnosis stamped with this
        # verification suffix to confirm it has been reviewed and approved.
        self.verification_suffix = " [Clinically Verified via Discharge Evaluation Policy]"

    def apply_hidden_doctor_policy(self, draft: DischargeSummaryDraft) -> DischargeSummaryDraft:
        """
        Takes the agent's draft and returns an edited copy reflecting the
        doctor's formatting preferences. Both the original and the edited
        version are passed to the learning engine to compute the edit distance.
        """
        # Deep-copy so the original draft is preserved for comparison
        edited = draft.model_copy(deep=True)

        # Policy 1  Principal diagnosis must carry the verification stamp
        if (
            edited.principal_diagnosis
            and edited.principal_diagnosis.lower() != "missing"
            and not edited.principal_diagnosis.endswith(self.verification_suffix)
        ):
            edited.principal_diagnosis += self.verification_suffix

        # Policy 2  Follow-up instructions must open with the compliance warning
        compliance_prefix = "CRITICAL CLINICAL FOLLOW-UP: Please visit the clinic as scheduled. "
        if (
            edited.follow_up_instructions
            and edited.follow_up_instructions.lower() != "missing"
            and not edited.follow_up_instructions.startswith(compliance_prefix)
        ):
            edited.follow_up_instructions = compliance_prefix + edited.follow_up_instructions

        return edited
