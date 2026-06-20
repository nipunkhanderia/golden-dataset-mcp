"""
MCP-facing input/output models for golden-dataset-studio-mcp.

These wrap golden_dataset's internal Pydantic models (GoldenEntry,
DatasetVersion, EvalSummary, etc.) with explicit dataset_path parameters,
since every tool call must be self-contained (no server-side session
state — see README for rationale).
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class InitDatasetInput(BaseModel):
    dataset_path: str = Field(..., description="Directory to initialise the dataset in")
    name: str = Field(..., description="Name of the dataset")
    description: str = Field("", description="Short dataset description")


class InitDatasetOutput(BaseModel):
    name: str
    description: str
    created_at: str


class AddEntryInput(BaseModel):
    dataset_path: str = Field(..., description="Directory containing the dataset")
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    ground_truth: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddEntryOutput(BaseModel):
    id: str
    status: str


class UpdateEntryInput(BaseModel):
    dataset_path: str
    entry_id: str
    question: Optional[str] = None
    answer: Optional[str] = None
    contexts: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class EntryOutput(BaseModel):
    id: str
    question: str
    answer: str
    contexts: list[str]
    tags: list[str]
    metadata: dict[str, Any]
    updated_at: str


class DeleteEntryInput(BaseModel):
    dataset_path: str
    entry_id: str


class DeleteEntryOutput(BaseModel):
    id: str
    status: str


class ListEntriesInput(BaseModel):
    dataset_path: str
    version: Optional[str] = Field(
        None, description="Specific committed version to list; omit for working tree"
    )


class ListEntriesOutput(BaseModel):
    entries: list[EntryOutput]
    count: int


class CommitInput(BaseModel):
    dataset_path: str
    description: str = ""


class CommitOutput(BaseModel):
    version: str
    entry_count: int
    sha256: str
    parent_version: Optional[str]


class DiffInput(BaseModel):
    dataset_path: str
    v1: str = Field(..., description="Earlier version, e.g. '1.0'")
    v2: str = Field(..., description="Later version, e.g. '1.1'")


class DiffOutput(BaseModel):
    added: list[EntryOutput]
    removed: list[EntryOutput]
    changed: list[EntryOutput]


class EvaluateInput(BaseModel):
    dataset_path: str
    actual_answers: list[str] = Field(
        ..., description="Answers to score, in the same order as entries in the target version"
    )
    version: Optional[str] = Field(
        None, description="Version to evaluate against; omit for current committed version"
    )


class EvalResultOutput(BaseModel):
    entry_id: str
    question: str
    expected_answer: str
    actual_answer: str
    semantic_similarity: Optional[float]
    overall_score: Optional[float]


class EvaluateOutput(BaseModel):
    dataset_name: str
    version: str
    total_entries: int
    avg_semantic_similarity: Optional[float]
    passed: bool
    results: list[EvalResultOutput]


class StatusInput(BaseModel):
    dataset_path: str


class StatusOutput(BaseModel):
    name: str
    description: str
    current_version: Optional[str]
    working_tree_entry_count: int
    versions: list[str]
