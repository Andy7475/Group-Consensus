# Group Consensus

An AI-assisted deliberation tool for groups that need to find common ground before a decision. It runs two complementary tracks from the same set of participant opinions, producing outputs that serve different purposes in a facilitated process.

Based on two research traditions:
- **Habermas Machine** (Google DeepMind, *Science* 2024) — LLM-mediated deliberation that generates and refines group consensus statements iteratively, selecting winners via Schulze social-choice voting
- **Polis** (Computational Democracy Project) — opinion clustering that maps where participants stand by collecting votes on statements and running K-means to reveal opinion groups and cross-group consensus

---

## How It Works

```
                    OPINIONS COLLECTED
                    (free-text, one per participant)
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
  TRACK A: MEDIATOR               TRACK B: ATOMIC VOTING
  03_mediator.ipynb               01_mediate.ipynb
  (~10 minutes)                   (~1 week to collect votes)
                                  + 02_analyse.ipynb
  LLM generates candidate         ─────────────────────────
  group statements from           Atomic statements extracted
  raw opinions. Schulze           from opinions → compressed
  voting picks the one            → participants vote agree/
  with broadest predicted         pass/disagree → opinion
  support.                        matrix → K-means clustering
                                  → bridging and divisive
  Use for: advance planning,      statements identified →
  narrowing real-world            group statement synthesised
  options, briefing a             from agreed points
  facilitator before the
  vote closes.                    Use for: democratic anchor,
                                  opinion map, surfacing
                                  hidden disagreements.
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                  FACILITATED WORKSHOP
                  ─────────────────────
                  Facilitator arrives with:
                  - Mediator statement (quick hypothesis)
                  - Atomic statement (democratic anchor)
                  - Opinion clusters (who agrees with whom)
                  - Divisive statements (where to focus)

                  Group finalises the decision.
```

---

## The Two Tracks

### Track A — Mediator (`03_mediator.ipynb`)

Runs immediately once opinions are in. The LLM reads all opinions, generates 5 candidate group statements from different angles, predicts how each participant would rank them based on what they wrote, and Schulze voting picks the winner. Optional: share the winner with participants and ask for critiques to run a second refinement round.

The output is a polished, holistic statement — good for advance planning. It doesn't require a voting round, and the turnaround is minutes. The trade-off: the Schulze ranking is an AI prediction of preference, not an actual vote. Participants never saw or chose between the candidates.

### Track B — Atomic Voting (`01_mediate.ipynb` + `02_analyse.ipynb`)

Takes longer because it requires real votes. The LLM extracts short atomic statements from opinions (one idea each, 5–15 words), compresses duplicates, flags genuine contradictions as CONTESTED. These go out as a voting form — MS Forms, Excel Online, or a shared spreadsheet. Participants vote Agree / Pass / Disagree on each statement.

Votes are loaded into an opinion matrix, clustered by similarity of voting pattern (UMAP + K-means), and analysed for bridging statements (≥60% approval across all clusters) and divisive ones (sharp splits between clusters). The bridging statements feed a final synthesis step that produces the group statement.

The output is grounded in what people actually agreed on. It also surfaces group structure — who clusters with whom, and where opinion divides — which the Mediator misses entirely. The democratic legitimacy matters when the group needs to own the outcome.

---

## Notebooks

| Notebook | Purpose | Time |
|---|---|---|
| `03_mediator.ipynb` | Run Mediator track, optional critique round, compare outputs | ~10 min |
| `01_mediate.ipynb` | Extract atomic statements, review, save voting form | ~15 min |
| `02_analyse.ipynb` | Load votes, cluster, detect bridging/divisive statements | ~5 min after votes in |

Run `03_mediator.ipynb` immediately after collecting opinions. Run `01_mediate.ipynb` at the same time to generate the voting form. Run `02_analyse.ipynb` once votes are back.

---

## Architecture

```
group_consensus/
├── models/
│   └── types.py              # Pydantic data models
├── mediation/
│   ├── async_atomic_model.py # Extract, compress, synthesise atomic statements
│   ├── async_mediator.py     # Async Mediator + StatementModel + RewardModel
│   ├── statement_model.py    # LLM candidate statement generation
│   ├── reward_model.py       # LLM-as-judge preference prediction
│   ├── social_choice.py      # Schulze voting method
│   └── mediator.py           # Synchronous deliberation loop
├── clustering/
│   ├── opinion_matrix.py     # Vote matrix
│   ├── engine.py             # UMAP + K-means clustering
│   └── consensus.py          # Bridging/divisive statement detection
├── pipeline/
│   └── session.py            # End-to-end session orchestration
└── api/
    ├── main.py               # FastAPI app
    └── routes/               # REST endpoints
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Anthropic API key

### Setup

```bash
git clone https://github.com/andy7475/group-consensus
cd group-consensus
uv sync
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
uv run pytest
```

### Run the notebooks

```bash
cd notebooks
uv run jupyter notebook
```

Open `03_mediator.ipynb` for the fast track, or `01_mediate.ipynb` to start the atomic voting track.

---

## Configuration

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key | required |
| `CLAUDE_MODEL` | Model for generation and ranking | `claude-sonnet-4-6` |
| `NUM_CANDIDATE_STATEMENTS` | Candidates per Mediator round | `5` |
| `MAX_DELIBERATION_ROUNDS` | Max Mediator iterations | `5` |
| `MIN_CLUSTER_APPROVAL` | Threshold for bridging statements | `0.6` |

---

## Roadmap

- [x] Schulze social-choice voting
- [x] LLM-based statement generation and candidate ranking
- [x] Deliberation loop with critique rounds (`03_mediator.ipynb`)
- [x] Atomic statement extraction and compression
- [x] Opinion matrix, UMAP + K-means clustering
- [x] Bridging/divisive statement detection
- [x] Group statement synthesis from agreed atomic points
- [x] Side-by-side comparison of both approaches
- [ ] Real world test

---

## References

- Tessler, M. H., et al. (2024). *AI can help humans find common ground in democratic deliberation*. **Science**, 386(6719). https://doi.org/10.1126/science.adq2852
- Google DeepMind Habermas Machine: https://github.com/google-deepmind/habermas_machine
- Computational Democracy Project — Polis: https://github.com/compdemocracy/polis
- Small, C., et al. (2021). *Polis: Scaling Deliberation by Mapping High Dimensional Opinion Spaces*. https://arxiv.org/abs/2005.12729

---

## Licence

MIT — see [LICENSE](LICENSE). If you incorporate code from the Habermas Machine repository directly, that code is Apache 2.0; Polis is AGPLv3. Check licences before production deployment.
