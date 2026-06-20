"""
Lifecycle tests for golden_dataset_mcp.server tools.

Each test uses a real temp directory (no mocking of the filesystem) since
this server's whole job is filesystem-backed versioning — mocking it away
would test nothing meaningful.
"""

import pytest

from golden_dataset_mcp.models import (
    AddEntryInput,
    CommitInput,
    DeleteEntryInput,
    DiffInput,
    EvaluateInput,
    InitDatasetInput,
    ListEntriesInput,
    StatusInput,
    UpdateEntryInput,
)
from golden_dataset_mcp.server import (
    add_entry,
    commit_version,
    dataset_status,
    delete_entry,
    diff_versions,
    evaluate_answers,
    init_dataset,
    list_entries,
    update_entry,
)


class TestInitDataset:
    def test_creates_dataset(self, dataset_path):
        result = init_dataset(
            InitDatasetInput(dataset_path=dataset_path, name="my-ds", description="desc")
        )
        assert result.name == "my-ds"
        assert result.description == "desc"

    def test_double_init_raises(self, initialised_dataset):
        with pytest.raises(FileExistsError):
            init_dataset(InitDatasetInput(dataset_path=initialised_dataset, name="again"))


class TestAddEntry:
    def test_adds_entry_to_working_tree(self, initialised_dataset):
        result = add_entry(
            AddEntryInput(
                dataset_path=initialised_dataset,
                question="What is BA's main hub?",
                answer="Heathrow Terminal 5",
                tags=["geography"],
            )
        )
        assert result.status == "added"
        assert result.id

        listing = list_entries(ListEntriesInput(dataset_path=initialised_dataset))
        assert listing.count == 1
        assert listing.entries[0].question == "What is BA's main hub?"

    def test_multiple_entries_accumulate(self, initialised_dataset):
        for i in range(3):
            add_entry(
                AddEntryInput(
                    dataset_path=initialised_dataset,
                    question=f"Q{i}",
                    answer=f"A{i}",
                )
            )
        listing = list_entries(ListEntriesInput(dataset_path=initialised_dataset))
        assert listing.count == 3


class TestUpdateAndDeleteEntry:
    def test_update_changes_fields(self, initialised_dataset):
        added = add_entry(
            AddEntryInput(dataset_path=initialised_dataset, question="Q", answer="old answer")
        )
        updated = update_entry(
            UpdateEntryInput(
                dataset_path=initialised_dataset, entry_id=added.id, answer="new answer"
            )
        )
        assert updated.answer == "new answer"
        assert updated.question == "Q"  # untouched field preserved

    def test_update_missing_entry_raises(self, initialised_dataset):
        with pytest.raises(KeyError):
            update_entry(
                UpdateEntryInput(
                    dataset_path=initialised_dataset, entry_id="nonexistent", answer="x"
                )
            )

    def test_delete_removes_entry(self, initialised_dataset):
        added = add_entry(
            AddEntryInput(dataset_path=initialised_dataset, question="Q", answer="A")
        )
        delete_entry(DeleteEntryInput(dataset_path=initialised_dataset, entry_id=added.id))
        listing = list_entries(ListEntriesInput(dataset_path=initialised_dataset))
        assert listing.count == 0

    def test_delete_missing_entry_raises(self, initialised_dataset):
        with pytest.raises(KeyError):
            delete_entry(DeleteEntryInput(dataset_path=initialised_dataset, entry_id="nope"))


class TestCommitVersion:
    def test_commit_creates_version_1_0(self, initialised_dataset):
        add_entry(AddEntryInput(dataset_path=initialised_dataset, question="Q", answer="A"))
        result = commit_version(CommitInput(dataset_path=initialised_dataset, description="first"))
        assert result.version == "1.0"
        assert result.entry_count == 1
        assert result.parent_version is None

    def test_second_commit_increments_version(self, initialised_dataset):
        add_entry(AddEntryInput(dataset_path=initialised_dataset, question="Q1", answer="A1"))
        commit_version(CommitInput(dataset_path=initialised_dataset))

        add_entry(AddEntryInput(dataset_path=initialised_dataset, question="Q2", answer="A2"))
        second = commit_version(CommitInput(dataset_path=initialised_dataset))
        assert second.version == "1.1"
        assert second.parent_version == "1.0"

    def test_commit_empty_working_tree_raises(self, initialised_dataset):
        with pytest.raises(ValueError, match="empty"):
            commit_version(CommitInput(dataset_path=initialised_dataset))


class TestDiffVersions:
    def test_diff_detects_added_entry(self, initialised_dataset):
        add_entry(AddEntryInput(dataset_path=initialised_dataset, question="Q1", answer="A1"))
        commit_version(CommitInput(dataset_path=initialised_dataset))

        add_entry(AddEntryInput(dataset_path=initialised_dataset, question="Q2", answer="A2"))
        commit_version(CommitInput(dataset_path=initialised_dataset))

        diff = diff_versions(DiffInput(dataset_path=initialised_dataset, v1="1.0", v2="1.1"))
        assert len(diff.added) == 1
        assert diff.added[0].question == "Q2"
        assert len(diff.removed) == 0

    def test_diff_detects_changed_answer(self, initialised_dataset):
        added = add_entry(
            AddEntryInput(dataset_path=initialised_dataset, question="Q", answer="original")
        )
        commit_version(CommitInput(dataset_path=initialised_dataset))

        update_entry(
            UpdateEntryInput(dataset_path=initialised_dataset, entry_id=added.id, answer="revised")
        )
        commit_version(CommitInput(dataset_path=initialised_dataset))

        diff = diff_versions(DiffInput(dataset_path=initialised_dataset, v1="1.0", v2="1.1"))
        assert len(diff.changed) == 1
        assert diff.changed[0].answer == "revised"


class TestEvaluateAnswers:
    def test_perfect_match_scores_high(self, initialised_dataset):
        add_entry(
            AddEntryInput(
                dataset_path=initialised_dataset,
                question="What is BA's main hub?",
                answer="Heathrow Terminal 5",
            )
        )
        commit_version(CommitInput(dataset_path=initialised_dataset))

        result = evaluate_answers(
            EvaluateInput(
                dataset_path=initialised_dataset,
                actual_answers=["Heathrow Terminal 5"],
            )
        )
        assert result.total_entries == 1
        assert result.avg_semantic_similarity == pytest.approx(1.0, abs=0.01)
        assert result.passed is True

    def test_mismatched_answer_count_raises(self, initialised_dataset):
        add_entry(AddEntryInput(dataset_path=initialised_dataset, question="Q", answer="A"))
        commit_version(CommitInput(dataset_path=initialised_dataset))

        with pytest.raises(ValueError, match="Mismatch"):
            evaluate_answers(
                EvaluateInput(
                    dataset_path=initialised_dataset,
                    actual_answers=["one", "two"],  # 2 answers, 1 entry
                )
            )

    def test_evaluate_with_no_committed_version_raises(self, initialised_dataset):
        with pytest.raises(ValueError, match="No committed version"):
            evaluate_answers(
                EvaluateInput(dataset_path=initialised_dataset, actual_answers=[])
            )

    def test_very_short_or_stopword_only_answers_raise_sklearn_error(self, initialised_dataset):
        """
        Documents a real upstream limitation: golden-dataset-studio's
        TF-IDF-based similarity scorer raises ValueError('empty vocabulary')
        when an answer is too short or consists only of English stop words
        (e.g. a bare number like '4', or 'the a an'). This is not a bug in
        this wrapper — it surfaces the underlying library's behaviour as-is.
        Callers should expect evaluate_answers to fail on such inputs and
        should avoid single-token/stopword-only golden answers, or catch
        and handle this case explicitly.
        """
        add_entry(
            AddEntryInput(dataset_path=initialised_dataset, question="Sum?", answer="4")
        )
        commit_version(CommitInput(dataset_path=initialised_dataset))

        with pytest.raises(ValueError, match="empty vocabulary"):
            evaluate_answers(
                EvaluateInput(dataset_path=initialised_dataset, actual_answers=["4"])
            )



class TestDatasetStatus:
    def test_status_reflects_state(self, initialised_dataset):
        add_entry(AddEntryInput(dataset_path=initialised_dataset, question="Q", answer="A"))
        status = dataset_status(StatusInput(dataset_path=initialised_dataset))
        assert status.name == "test-dataset"
        assert status.working_tree_entry_count == 1
        assert status.versions == []

        commit_version(CommitInput(dataset_path=initialised_dataset))
        status = dataset_status(StatusInput(dataset_path=initialised_dataset))
        assert status.current_version == "1.0"
        assert status.versions == ["1.0"]
