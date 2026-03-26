from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalInput:
    task_id: str
    pr_number: int
    repo: str
    config_name: str
    rendered_context: str
    total_tokens: int
    pipeline_version: str
    difficulty: str
    pr_type: str
    language: str
    diff_patch: str
    human_comments: list[dict]
    has_severity_annotations: bool


@dataclass
class AgentComment:
    comment_id: str
    body: str
    file_reference: str | None
    line_reference: int | None
    severity_claim: str | None
    is_outside_diff: bool


@dataclass
class AgentOutput:
    task_id: str
    config_name: str
    model: str
    raw_response: str
    comments: list[AgentComment]
    parse_success: bool
    parse_error: str | None


@dataclass
class JudgeClassification:
    comment_id: str
    classification: str
    matched_human_comment_id: str | None
    actionability_score: int
    reasoning: str | None


@dataclass
class HumanCommentStatus:
    comment_id: str
    status: str
    matched_agent_comment_id: str | None


@dataclass
class JudgeOutput:
    task_id: str
    config_name: str
    agent_classifications: list[JudgeClassification]
    human_comment_statuses: list[HumanCommentStatus]
    judge_model: str
    judge_prompt_version: str


@dataclass
class EvalResult:
    task_id: str
    pr_number: int
    config_name: str
    model: str
    pipeline_version: str
    difficulty: str
    language: str
    detection_rate: float
    false_positive_rate: float
    severity_accuracy: float
    actionability_score: float
    semantic_alignment: float
    adjacent_detection_rate: float
    overall_score: float
    total_agent_comments: int
    confirmed_count: int
    plausible_count: int
    fabricated_count: int
    caught_human_comments: int
    missed_human_comments: int
    total_human_comments: int
    matched_pairs: list[dict]
    missed_comments: list[dict]
    fabricated_comments: list[dict]
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    hallucination_rate: float = 0.0
    redundancy_penalty: float = 0.0
    plausible_penalty: float = 0.0
    fallback_penalty: float = 0.0
    judge_alignment_agreement: float = 0.0
    attempt_rate: float = 0.0
    coverage: float = 0.0 
    difficulty_weight: float = 0.0
    failure_type: str | None = None
    pr_difficulty_tag: str = "easy"
    agent_parse_success: bool = True
    agent_parse_error: str | None = None
    judge_parse_fallback_used: bool = False

