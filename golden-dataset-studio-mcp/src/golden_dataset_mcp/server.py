"""
golden-dataset-studio-mcp: an MCP server wrapping the golden-dataset-studio
library (pip install golden-dataset-studio) for version-controlled golden
dataset management and semantic evaluation of RAG/LLM pipelines.

This server does NOT call any LLM itself — evaluation uses TF-IDF cosine
similarity (scikit-learn), so no API key is required. All data is stored
on the local filesystem under <dataset_path>/.golden_dataset/.

Every tool takes an explicit `dataset_path` parameter rather than relying
on server-side session state or the process's current working directory.
This keeps the server stateless between calls and safe for concurrent use
across multiple datasets/clients. See README.md for rationale.
"""

from pathlib import Path

from fastmcp import FastMCP

from golden_dataset.evaluator import Evaluator
from golden_dataset.models import GoldenEntry
from golden_dataset.store import DatasetStore

from golden_dataset_mcp.models import (
    AddEntryInput,
    AddEntryOutput,
    CommitInput,
    CommitOutput,
    DeleteEntryInput,
    DeleteEntryOutput,
    DiffInput,
    DiffOutput,
    EntryOutput,
    EvalResultOutput,
    EvaluateInput,
    EvaluateOutput,
    InitDatasetInput,
    InitDatasetOutput,
    ListEntriesInput,
    ListEntriesOutput,
    StatusInput,
    StatusOutput,
    UpdateEntryInput,
)

mcp = FastMCP(
    name="golden-dataset-studio",
    instructions=(
        "Version-controlled golden dataset management and semantic evaluation "
        "for RAG/LLM pipelines. Every tool requires an explicit dataset_path "
        "pointing at (or to be initialised as) a golden dataset directory. "
        "Typical flow: init_dataset -> add_entry (repeat) -> commit -> "
        "evaluate_answers against a committed version. Use diff_versions to "
        "see what changed between two committed versions, and status to "
        "check current state. No LLM is called by this server; evaluation "
        "uses TF-IDF cosine similarity, not semantic embeddings."
    ),
)


def _entry_to_output(entry: GoldenEntry) -> EntryOutput:
    return EntryOutput(
        id=entry.id,
        question=entry.question,
        answer=entry.answer,
        contexts=entry.contexts,
        tags=entry.tags,
        metadata=entry.metadata,
        updated_at=str(entry.updated_at),
    )


@mcp.tool()
def init_dataset(input: InitDatasetInput) -> InitDatasetOutput:
    """
    Initialise a new version-controlled golden dataset at dataset_path.

    Creates a .golden_dataset/ directory there. Fails if one already exists
    at that path — delete .golden_dataset/ manually to start fresh.
    """
    store = DatasetStore(root=Path(input.dataset_path))
    manifest = store.init(input.name, input.description)
    return InitDatasetOutput(
        name=manifest.name,
        description=manifest.description,
        created_at=str(manifest.created_at),
    )


@mcp.tool()
def add_entry(input: AddEntryInput) -> AddEntryOutput:
    """
    Add a question-answer pair to the working tree of a golden dataset.

    Entries added here are NOT yet versioned — call commit_version to
    snapshot them. dataset_path must already be initialised.
    """
    store = DatasetStore(root=Path(input.dataset_path))
    entry = GoldenEntry(
        question=input.question,
        answer=input.answer,
        contexts=input.contexts,
        ground_truth=input.ground_truth,
        tags=input.tags,
        metadata=input.metadata,
    )
    store.add_entry(entry)
    return AddEntryOutput(id=entry.id, status="added")


@mcp.tool()
def update_entry(input: UpdateEntryInput) -> EntryOutput:
    """
    Update fields of an existing working-tree entry by its id.

    Only fields you provide are changed; omitted fields are left as-is.
    Raises an error if entry_id is not found in the working tree.
    """
    store = DatasetStore(root=Path(input.dataset_path))
    kwargs = {}
    for field in ("question", "answer", "contexts", "tags", "metadata"):
        value = getattr(input, field)
        if value is not None:
            kwargs[field] = value
    updated = store.update_entry(input.entry_id, **kwargs)
    return _entry_to_output(updated)


@mcp.tool()
def delete_entry(input: DeleteEntryInput) -> DeleteEntryOutput:
    """Remove an entry from the working tree by its id. Does not affect already-committed versions."""
    store = DatasetStore(root=Path(input.dataset_path))
    store.delete_entry(input.entry_id)
    return DeleteEntryOutput(id=input.entry_id, status="deleted")


@mcp.tool()
def list_entries(input: ListEntriesInput) -> ListEntriesOutput:
    """
    List entries in a dataset. Omit version to see the uncommitted working
    tree; pass a version (e.g. '1.0') to see a committed snapshot.
    """
    store = DatasetStore(root=Path(input.dataset_path))
    if input.version is None:
        entries = store.get_working_entries()
    else:
        entries = store.load_version(input.version)
    outputs = [_entry_to_output(e) for e in entries]
    return ListEntriesOutput(entries=outputs, count=len(outputs))


@mcp.tool()
def commit_version(input: CommitInput) -> CommitOutput:
    """
    Snapshot the current working tree as a new immutable dataset version.

    Versions auto-increment (1.0 -> 1.1 -> 1.2...). Fails if the working
    tree is empty.
    """
    store = DatasetStore(root=Path(input.dataset_path))
    version = store.commit(input.description)
    return CommitOutput(
        version=version.version,
        entry_count=version.entry_count,
        sha256=version.sha256,
        parent_version=version.parent_version,
    )


@mcp.tool()
def diff_versions(input: DiffInput) -> DiffOutput:
    """Show entries added, removed, or changed between two committed versions."""
    store = DatasetStore(root=Path(input.dataset_path))
    changes = store.diff(input.v1, input.v2)
    return DiffOutput(
        added=[_entry_to_output(e) for e in changes["added"]],
        removed=[_entry_to_output(e) for e in changes["removed"]],
        changed=[_entry_to_output(e) for e in changes["changed"]],
    )


@mcp.tool()
def evaluate_answers(input: EvaluateInput) -> EvaluateOutput:
    """
    Score actual LLM/RAG-generated answers against the golden dataset using
    TF-IDF cosine similarity (no LLM call, no API key needed).

    actual_answers must be supplied in the same order as the entries in
    the target version. Omit `version` to evaluate against the current
    committed version.
    """
    store = DatasetStore(root=Path(input.dataset_path))
    manifest = store.load_manifest()
    version = input.version or manifest.current_version
    if not version:
        raise ValueError("No committed version exists yet. Call commit_version first.")

    entries = store.load_version(version)
    evaluator = Evaluator()
    summary = evaluator.evaluate_dataset(
        entries, input.actual_answers, dataset_name=manifest.name, version=version
    )
    store.save_eval(summary)

    return EvaluateOutput(
        dataset_name=summary.dataset_name,
        version=summary.version,
        total_entries=summary.total_entries,
        avg_semantic_similarity=summary.avg_semantic_similarity,
        passed=summary.passed(),
        results=[
            EvalResultOutput(
                entry_id=r.entry_id,
                question=r.question,
                expected_answer=r.expected_answer,
                actual_answer=r.actual_answer,
                semantic_similarity=r.semantic_similarity,
                overall_score=r.overall_score,
            )
            for r in summary.results
        ],
    )


@mcp.tool()
def dataset_status(input: StatusInput) -> StatusOutput:
    """Show the current state of a golden dataset: name, current version, and working tree size."""
    store = DatasetStore(root=Path(input.dataset_path))
    manifest = store.load_manifest()
    working = store.get_working_entries()
    return StatusOutput(
        name=manifest.name,
        description=manifest.description,
        current_version=manifest.current_version,
        working_tree_entry_count=len(working),
        versions=[v.version for v in manifest.versions],
    )


def main() -> None:
    """Entry point for the `golden-dataset-mcp` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
