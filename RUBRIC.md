# SWE-PRBench Evaluation Rubrics

This document is the canonical reference for how we classify agent comments, what counts as ground-truth human feedback, and how PR difficulty types are defined. 
**Version:** v0.4.1


---

## 1. Judge classification rubric (CONFIRMED / PLAUSIBLE / FABRICATED)

These rules apply **per agent comment**, comparing it to human review comments and the code in context. They align with the operational definitions in the judge prompt (`eval_harness/judge.py`).

### CONFIRMED

An agent comment is **CONFIRMED** if **all** of the following hold:

1. The agent identifies a specific issue in the code.
2. A human reviewer identified the **same underlying issue** (exact wording need not match).
3. The issue concerns the same file or functional area.
4. The agent’s concern would lead to the **same kind of code change** as the human reviewer’s concern.

### PLAUSIBLE

An agent comment is **PLAUSIBLE** if:

1. The comment is grounded in code visible in the context.
2. The observation is **factually correct** about the code.
3. A reasonable engineer might raise the concern.
4. **No** human reviewer raised this specific concern in the ground-truth set.

### FABRICATED

An agent comment is **FABRICATED** if **any** of the following hold:

1. References code, functions, or behavior **not present** in the provided context.
2. Makes **factually incorrect** claims about the code.
3. Describes a bug that **does not exist** in the shown code.
4. Invents method signatures, variable names, or behavior.

---

## 2. Ground-truth inclusion rubric (human comments)

A human review comment **is included** in ground truth if:

1. Written by a human (not a bot or AI tool).
2. Is an **initiating** comment (not a reply).
3. Contains **≥10 words**.
4. References specific code behavior, correctness, or engineering concern.
5. Is not pure praise or acknowledgment.

A comment **is not** included if:

1. Author matches known bot patterns.
2. Is a reply to another comment.
3. Is style-only feedback.
4. Is a question without substantive concern.

*(Implementation details may live in the data collection pipeline; this rubric states the intent.)*

---

## 3. Difficulty classification rubric (PR types)

### Type1_Direct

**All** ground-truth comments reference lines present in the diff hunk (`is_in_diff=True`).

### Type2_Contextual

**At least one** ground-truth comment references unchanged code in the same files as the diff, **or** references how the changed code interacts with existing code (`is_in_diff=False`, file in changed files).

### Type3_Latent_Candidate

**At least one** ground-truth comment references a file **not** in the diff (`comment_file` not in `changed_files`).

---
