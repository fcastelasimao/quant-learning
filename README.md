# personal_projects

Personal quantitative-finance R&D — one repo per strategy/idea, plus planning and notes.
Part of the `QuantFinance` workspace (see the workspace root `README.md` for the shared
data engine, conda env layout, and how this repo relates to its sibling repo
`quantcore`).

> **Disclaimer:** Educational and research software, not financial advice. Past
> performance does not guarantee future results.

## Structure

```
personal_projects/
├── projects/     Individual strategy/research projects — see projects/README.md
│                 for the full project index, status, and quick-start commands
├── archive/      Concluded projects, kept for reference (see projects/README.md
│                 "Archived" table for why each one was shelved)
├── roadmaps/     Interactive HTML/JSX learning & planning roadmaps
└── notes/        Obsidian vault (Concepts, Daily-Log, Quick-Reference, Snippets)
```

## Where to start

- **Project index, status, and per-project quick-start:** [`projects/README.md`](projects/README.md).
- **Shared market data:** provided by the workspace-level `quantcore` package; see the
  `QuantFinance/README.md` at the workspace root for the ingestion CLI and setup.
- **Environments:** each project under `projects/` keeps its own conda environment —
  see that project's own README/`environment.yml` for the env name and entry point.

## License

This repository is for personal research and education.
