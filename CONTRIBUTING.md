# Contributing & Workflow

## Branch Strategy

```
main          ← stable, tested, documented
  └── dev     ← integration branch
        └── feature/step-N-description   ← one branch per build step
```

**Rules:**
- `main` only receives merges from `dev` — never commit directly
- `dev` only receives merges from `feature/*` branches
- Every feature branch maps to one step in `DEVLOG.md`
- Branch is deleted after merge

## Branch Naming

```
feature/step-03-embedding-verification
feature/step-04-pdf-extraction
feature/step-05-chunking-pipeline
feature/step-06-duckdb-schema
feature/step-07-faiss-index
feature/step-08-networkx-graph
feature/step-09-ingestion-single-paper
feature/step-10-layer3-extraction
feature/step-11-query-engine
fix/short-description-of-bug
```

## Commit Message Convention

```
<type>(<scope>): <short description>

[optional body — what and why, not how]
```

**Types:**
| Type | When to use |
|---|---|
| `feat` | New working capability added |
| `fix` | Bug fix |
| `chore` | Setup, config, dependencies |
| `docs` | DEVLOG, README, comments |
| `test` | Verification scripts |
| `refactor` | Code restructure, no behaviour change |
| `wip` | Work in progress — not yet working |

**Examples:**
```
chore(env): add .gitignore and requirements.txt

feat(embedding): verify SPECTER2 runs on RTX 5070 Ti

fix(torch): switch cu124 to cu130 nightly for sm_120 support

docs(devlog): fill in step 3 embedding verification output

feat(extraction): implement PDF chunking with section context
```

## Workflow Per Step

```
1. Create feature branch
   git checkout dev
   git checkout -b feature/step-N-description

2. Write code / run / iterate

3. When working — stage and commit
   git add .
   git commit -m "feat(scope): description"

4. Update DEVLOG.md
   git add DEVLOG.md
   git commit -m "docs(devlog): fill in step N output and decisions"

5. Merge to dev
   git checkout dev
   git merge feature/step-N-description
   git branch -d feature/step-N-description

6. Push
   git push origin dev

7. Merge dev → main only at stable milestones
   git checkout main
   git merge dev
   git push origin main
   git tag v0.N-step-description
```
