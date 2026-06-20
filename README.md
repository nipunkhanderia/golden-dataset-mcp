# golden-dataset-mcp

An [MCP](https://modelcontextprotocol.io) server wrapping [`golden-dataset-studio`](https://pypi.org/project/golden-dataset-studio/) — version-controlled golden dataset management and semantic evaluation for RAG/LLM pipelines.

This is a thin protocol layer over the existing `golden_dataset` library (`DatasetStore`, `Evaluator`). It does not reimplement any logic — it exposes the library's existing Python API as MCP tools so an agent (Claude Desktop, Claude Code, or any MCP client) can manage golden datasets conversationally.

**No LLM API key required.** Evaluation uses TF-IDF cosine similarity (scikit-learn), not an LLM call.

## Why a separate package from `golden-dataset-studio`?

`golden-dataset-studio` is a CLI tool — designed for a human typing `golden add`, `golden commit`, etc. in a terminal. `golden-dataset-mcp` exposes the same underlying operations as MCP tools so an LLM agent can drive them programmatically, e.g. as part of an automated RAG evaluation pipeline. Keeping them as separate PyPI packages means CLI users aren't forced to pull in `fastmcp` as a dependency, and MCP users get a clean, protocol-focused package.

## Tools

| Tool | What it does |
|---|---|
| `init_dataset` | Initialise a new dataset at a given path |
| `add_entry` | Add a question/answer pair to the working tree |
| `update_entry` | Edit fields of an existing working-tree entry |
| `delete_entry` | Remove an entry from the working tree |
| `list_entries` | List working-tree or committed-version entries |
| `commit_version` | Snapshot the working tree as a new immutable version |
| `diff_versions` | Show entries added/removed/changed between two versions |
| `evaluate_answers` | Score actual answers against a version via TF-IDF cosine similarity |
| `dataset_status` | Show current version, working tree size, and version history |

## Design: every tool takes an explicit `dataset_path`

Unlike the CLI (which operates on the current working directory), every tool here requires an explicit `dataset_path` parameter. This keeps the server fully stateless between calls — no hidden "current dataset" session state to lose track of, and safe for one server instance to manage multiple datasets or serve multiple concurrent clients.

## Installation

```bash
pip install golden-dataset-mcp
```

This pulls in `golden-dataset-studio` automatically as a dependency.

## Usage with Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "golden-dataset": {
      "command": "golden-dataset-mcp"
    }
  }
}
```

No environment variables needed — no API key, no config.

## Example flow

```
1. init_dataset(dataset_path="./my-rag-eval", name="support-bot-eval")
2. add_entry(dataset_path="./my-rag-eval", question="...", answer="...")
   [repeat for each golden Q&A pair]
3. commit_version(dataset_path="./my-rag-eval", description="initial 50 questions")
4. [run your RAG pipeline, collect actual answers]
5. evaluate_answers(dataset_path="./my-rag-eval", actual_answers=[...])
   -> avg_semantic_similarity, per-entry scores, pass/fail
```

As your RAG pipeline changes over time, `commit_version` again after edits and use `diff_versions` to see exactly what changed in your golden set between releases.

## Relationship to the underlying library

| | `golden-dataset-studio` | `golden-dataset-mcp` |
|---|---|---|
| **Interface** | CLI (`golden ...`) | MCP tools |
| **Driven by** | A human typing commands | An LLM agent / MCP client |
| **Path handling** | Current working directory | Explicit `dataset_path` per call |
| **Dependency direction** | — | Depends on `golden-dataset-studio` |

If you want the human-driven CLI, use `golden-dataset-studio` directly. If you want an agent to drive it, use this package.

## Development

```bash
git clone https://github.com/nipunkhanderia/golden-dataset-mcp
cd golden-dataset-mcp
pip install -e ".[dev]" --break-system-packages
pytest -v
```

Validate the MCP-facing contract:

```bash
npx @modelcontextprotocol/inspector golden-dataset-mcp
```

## Limitations

- `evaluate_answers` uses TF-IDF cosine similarity, which captures lexical overlap better than deep semantic meaning. For embedding-based or RAGAS-style metrics, call the underlying library's `Evaluator.ragas_evaluate()` directly (requires `pip install ragas datasets` — not exposed as an MCP tool in this version).
- **Very short or stop-word-only answers will raise an error.** scikit-learn's TF-IDF vectorizer raises `ValueError: empty vocabulary` on inputs like a bare `"4"` or `"the a an"`. Avoid single-token golden answers, or expect `evaluate_answers` to fail on them.
- All state is filesystem-backed JSON/JSONL under `<dataset_path>/.golden_dataset/`; this server does no remote storage or syncing.
- `golden-dataset-studio` (as of 0.1.1) uses `scikit-learn` at runtime but does not declare it in its own dependencies — this package pins `scikit-learn` explicitly so `evaluate_answers` works correctly out of the box regardless of upstream's declared requirements.

## License

MIT
