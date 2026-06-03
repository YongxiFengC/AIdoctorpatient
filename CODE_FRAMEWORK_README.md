# CADRE Code Framework README

This document explains the current CADRE computational MVP for professor-facing
discussion. The code is not a trained clinical model. It is a synthetic,
transparent prototype that verifies whether the latest CADRE mechanism can be
implemented as a runnable pipeline.

## Core Idea

The MVP implements:

```text
shared objective patient latent state z_t
        ↓
AI objective branch → R_AI

doctor-context profile u_t^d
        ↓
latent calibration h_t^d = z_t + C(z_t, u_t^d, c_t)
        ↓
doctor ranking head → R_MD

compare R_AI and R_MD
        ↓
discrepancy attribution
        ↓
constraint-aware revision
        ↓
final output
```

The important point is that the AI branch and doctor-context branch are not two
unrelated models. They start from the same shared patient representation
`z_t`. The doctor branch applies context calibration to that same latent state.

## Main Entry Point

The main pipeline is in:

```text
src/agent.py
```

The core execution order is:

```python
state = self.encoder.encode(case)

safety = self.safety.screen(state, case.candidate_actions)

ai_ranking = self.ai_branch.rank(
    state, safety.safe_actions, self.knowledge_base
)

profile = self.profile_builder.build(case, safety.safe_actions)

calibrated_state = self.calibration.calibrate(state, profile)

doctor_ranking = self.doctor_head.rank(
    calibrated_state, safety.safe_actions
)

attributions = self.attribution.attribute(
    safety.safe_actions, ai_ranking, doctor_ranking, profile
)

final_ranking = self.reviser.revise(
    case, safety.safe_actions, ai_ranking, profile
)
```

This corresponds to the ten-step methodology in the Overleaf notes.

## Directory Structure

```text
src/
  data/
    patient_state.py
    synthetic_cases.py

  knowledge/
    medical_knowledge_base.py

  models/
    shared_state_encoder.py
    ai_objective_branch.py
    doctor_context_profile.py
    latent_calibration.py
    doctor_ranking_head.py
    discrepancy_attribution.py

  safety/
    clinical_rules.py

  planning/
    constraint_aware_reviser.py

  output/
    output_controller.py

  training/
    losses.py
    trainer.py

  agent.py
  demo.py
  evaluate.py
  train.py

tests/
  test_pipeline.py
```

## Module Explanation

### `data/patient_state.py`

Defines the shared data structures used across the MVP:

- `PatientCase`
- `CandidateAction`
- `LatentState`
- `ContextProfile`
- `CalibratedState`
- `RankedAction`
- `Attribution`
- `ReviewOutput`

This file is the code-level counterpart of the variables in the method notes.

### `data/synthetic_cases.py`

Defines synthetic visit-level cases. These are not real patients and not time
series. They are one-shot cases designed to test whether the CADRE mechanism
responds correctly to different discrepancy sources.

Current cases:

- `aligned_views`
- `access_constraint`
- `unconfirmed_access`
- `patient_goal_difference`
- `missing_safety_information`
- `known_contraindication`
- `unexplained_discrepancy`

### `models/shared_state_encoder.py`

Implements Step 1: shared patient representation.

It converts patient observations into the shared objective latent state:

```text
z_t = E(h_t, o_t)
```

In the MVP, this is a transparent synthetic representation containing:

- severity
- progression risk
- treatment tolerance
- adherence risk
- uncertainty

It is not yet a learned neural embedding.

### `safety/clinical_rules.py`

Implements Step 2: knowledge-based safety filtering.

It checks whether:

- safety-critical fields are missing;
- an action has a known synthetic contraindication flag.

If safety-critical information is missing, the pipeline can defer instead of
generating an overconfident recommendation.

### `knowledge/medical_knowledge_base.py`

Represents the current synthetic medical knowledge base.

At this stage it only returns a placeholder `knowledge_score`. In a real
disease-specific implementation, this module would be replaced with guideline
rules, contraindication rules, drug-interaction logic, monitoring rules, and
other medical knowledge.

### `models/ai_objective_branch.py`

Implements Step 3: AI objective ranking.

It computes:

```text
AI score = outcome score + knowledge alignment - uncertainty penalty
```

and returns:

```text
R_AI
```

For example:

```text
A > B > C
```

### `models/doctor_context_profile.py`

Implements Step 4: doctor-context profile.

It builds:

```text
u_t^d
```

from context variables such as:

- action availability;
- insurance coverage;
- patient goals;
- physician or institution preference placeholders.

It also records:

- confirmed constraints;
- unconfirmed constraints;
- information to verify;
- action-specific context biases.

### `models/latent_calibration.py`

Implements Step 5: doctor-context latent calibration.

It computes:

```text
h_t^d = z_t + C(z_t, u_t^d, c_t)
```

In the MVP, this is transparent and rule-based. The output explicitly contains:

- the original patient latent dimensions;
- context calibration dimensions;
- action-specific calibration biases.

### `models/doctor_ranking_head.py`

Implements Step 6: predicted doctor-context ranking.

It uses the calibrated state `h_t^d` to generate:

```text
R_MD
```

For example, in the `access_constraint` case:

```text
AI ranking:                  A > B > C
doctor-calibrated ranking:   B > C > A
```

because Action A is unavailable in the context.

### `models/discrepancy_attribution.py`

Implements Step 8: discrepancy attribution.

It compares:

```text
R_AI vs R_MD
```

and assigns rule-anchored attribution hypotheses:

- `real_world_feasibility_constraint`
- `patient_goal_difference`
- `physician_or_institutional_preference`
- `missing_or_unobserved_information`
- `knowledge_related_or_evidence_alignment_mismatch`
- `unresolved_discrepancy`

This is not clustering and not a supervised classifier. It is an interpretable,
rule-anchored MVP because we do not currently assume true labels explaining why
historical physicians made decisions.

### `planning/constraint_aware_reviser.py`

Implements Step 9: action revision.

It does not blindly accept `R_MD`. Instead, it:

- keeps the AI objective score as the main ranking signal;
- removes actions with confirmed feasibility constraints;
- adds patient-goal adjustment when available.

Example:

```text
AI ranking:      A > B > C
A unavailable
final ranking:  B > C
```

### `output/output_controller.py`

Implements Step 10: output selection.

It returns one of:

- `ranked_recommendation`
- `conditional_recommendation`
- `information_request_defer_to_clinician`

The final JSON output includes:

- shared latent state;
- doctor-context profile;
- doctor-calibrated state;
- AI ranking;
- doctor-calibrated ranking;
- discrepancy attribution;
- final ranking;
- information to verify.

### `training/losses.py`

Defines the replaceable multi-task objective:

```text
L = L_outcome
  + lambda_MD * L_physician_action
  + lambda_K * L_knowledge
  + lambda_cal * L_calibration
```

The current implementation is framework-agnostic and uses plain Python numbers.
It is not real clinical training. It exists to show how future empirical code
would connect outcome labels, historical physician actions, knowledge
constraints, and calibration regularization.

### `training/trainer.py`

Runs one synthetic training-objective smoke example. This gives an inspectable
loss breakdown without pretending that we have trained a real model.

## How to Run

From the project root:

```bash
python3 -m src.demo
```

Run one case:

```bash
python3 -m src.demo --case access_constraint
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

Generate JSON Lines output:

```bash
python3 -m src.demo --jsonl demo_outputs.jsonl
```

Inspect the training-objective skeleton:

```bash
python3 -m src.train
```

## What Has Been Verified

The tests currently verify:

1. Shared latent state can produce both AI ranking and doctor-calibrated
   ranking.
2. Doctor-context calibration can change predicted doctor ranking.
3. Discrepancy attribution can identify rule-anchored possible causes.
4. Final recommendation can change under confirmed constraints.
5. Patient goals can modify final ranking.
6. Missing safety information triggers defer output.
7. Known contraindications are excluded before ranking.
8. The training objective skeleton returns named loss components.

Current test result:

```text
Ran 8 tests
OK
```

## What This MVP Does Not Prove

This MVP does not prove:

- clinical validity;
- true treatment effect estimation;
- accurate physician behavior prediction;
- ground-truth discrepancy causes;
- real-world deployment safety.

It only verifies computational feasibility of the latest CADRE mechanism.

## Professor-Facing Summary

The current code is a synthetic computational MVP for CADRE. It demonstrates
that the framework can start from a shared objective patient latent state,
generate both an AI objective ranking and a doctor-context calibrated ranking,
compare the two rankings, infer rule-anchored discrepancy hypotheses, and
revise the final recommendation under credible constraints or patient goals.

All scores and rules are placeholders. After selecting the disease setting and
dataset, the placeholder modules can be replaced by learned components:

- learned shared state encoder;
- learned AI objective/outcome branch;
- learned doctor-context calibration branch;
- disease-specific medical knowledge base;
- weakly supervised or expert-reviewed attribution module.
- empirical training loop using outcome, physician-action, knowledge, and
  calibration losses.
