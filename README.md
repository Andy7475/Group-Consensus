# Group Consensus

An AI-mediated platform for scalable group deliberation, combining two complementary research approaches:

- **Habermas Machine** (Google DeepMind, *Science* 2024) — LLM-mediated caucus deliberation that generates and refines group consensus statements iteratively, selecting winners via social-choice voting
- **Polis** (Computational Democracy Project) — opinion clustering that maps where participants stand across a topic by collecting votes on statements and running K-means to reveal opinion groups and cross-group consensus

The idea is that the Habermas-style mediation does the hard work of surfacing what people actually need, fear, and value in small groups — then those refined statements feed into a Polis-style wide-distribution system where a much larger population can engage. Human mediators get a dashboard showing the consensus landscape so they can intervene precisely where it matters.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DEEP DELIBERATION                   │
│              (Habermas Machine approach, small groups)          │
│                                                                 │
│  Participants submit opinions → LLM generates N candidate       │
│  statements → reward model predicts preferences → Schulze       │
│  social-choice voting selects winner → participants critique    │
│  → repeat until convergence                                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │ refined consensus statements
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: WIDE DISTRIBUTION                   │
│              (Polis-style clustering, large groups)             │
│                                                                 │
│  Refined statements circulated broadly → participants vote      │
│  agree/disagree/pass → opinion matrix built → K-means          │
│  clustering reveals opinion groups → group-informed consensus   │
│  identified (statements resonating across ALL clusters)         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ consensus map + bridging statements
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: HUMAN FACILITATION                  │
│                                                                 │
│  Mediators view 2D opinion landscape, cluster breakdown,        │
│  group-informed consensus scores, and divisive statements.      │
│  AI has done the groundwork; facilitators intervene with        │
│  full situational awareness.                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Habermas Machine (Phase 1)

From [Tessler et al., 2024](https://www.science.org/doi/10.1126/science.adq2852), tested with 5,734 UK participants on contested topics. The original system uses a fine-tuned reward model; this implementation uses Claude as both generator and judge, which is a valid approximation for a new deployment.

Core loop:
1. Participants write free-text opinions on a topic
2. LLM reads all opinions and generates 16 candidate consensus statements
3. For each candidate, the LLM predicts how each participant would rank it
4. **Schulze method** (social-choice voting) selects the winning statement from predicted rankings
5. Participants can critique the statement
6. Critiques feed back into the next generation round
7. Repeat until the group is satisfied or a round limit is hit

### Polis-Style Clustering (Phase 2)

From [compdemocracy/polis](https://github.com/compdemocracy/polis). Participants vote Agree (+1), Disagree (-1), or Pass (0) on each statement. This builds an opinion matrix. K-means (k=2–5, chosen by silhouette score) clusters participants into opinion groups. A **group-informed consensus** statement is one where every cluster has above-threshold approval — it bridges divides rather than just winning a majority.

### Why Combine Them

The Habermas Machine produces a small number of high-quality, carefully refined statements. Polis handles scale and reveals the opinion landscape. Neither alone is sufficient: raw Polis without curation can flood participants with poor statements; the Habermas Machine without scale produces small-group outputs that may not generalise. Together they deliver quality and reach.

---

## Architecture

```
group_consensus/
├── models/
│   └── types.py          # Pydantic data models (Participant, Statement, Vote, etc.)
├── mediation/
│   ├── statement_model.py # LLM-based consensus statement generation (Claude API)
│   ├── reward_model.py    # LLM-as-judge preference prediction
│   ├── social_choice.py   # Schulze voting method
│   └── mediator.py        # Orchestrates the full deliberation loop
├── clustering/
│   ├── opinion_matrix.py  # Builds and manages the vote matrix
│   ├── engine.py          # K-means + PCA/dimensionality reduction
│   └── consensus.py       # Group-informed consensus detection
├── pipeline/
│   └── session.py         # End-to-end session: mediation → clustering
└── api/
    ├── main.py            # FastAPI app
    └── routes/            # REST endpoints
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

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run tests
uv run pytest

# Start the API server
uv run uvicorn group_consensus.api.main:app --reload
```

### Run a Demo Deliberation

```python
import asyncio
from group_consensus.pipeline.session import ConsensusSession

async def main():
    session = ConsensusSession(
        topic="How should our organisation approach remote working policy?",
        participants=[
            {"id": "p1", "name": "Alice"},
            {"id": "p2", "name": "Bob"},
            {"id": "p3", "name": "Carol"},
        ]
    )

    # Phase 1: gather opinions and mediate
    opinions = {
        "p1": "I value flexibility and work better without a commute.",
        "p2": "I find it hard to collaborate remotely and miss the office energy.",
        "p3": "A hybrid approach seems fair but we need clear guidelines.",
    }
    result = await session.run_mediation(opinions, max_rounds=3)
    print("Consensus statement:", result.consensus_statement.text)

    # Phase 2: circulate statements and cluster
    # (add more participants voting on the refined statements)
    cluster_result = await session.run_clustering()
    print("Group-informed consensus:", cluster_result.bridging_statements)

asyncio.run(main())
```

---

## Configuration

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key | required |
| `CLAUDE_MODEL` | Model for statement generation | `claude-sonnet-4-6` |
| `NUM_CANDIDATE_STATEMENTS` | Candidates generated per round | `8` |
| `MAX_DELIBERATION_ROUNDS` | Max mediation iterations | `5` |
| `MIN_CLUSTER_APPROVAL` | Threshold for group-informed consensus | `0.6` |
| `DATABASE_URL` | SQLAlchemy connection string | `sqlite:///consensus.db` |

---

## Roadmap

- [x] Core data models
- [x] Schulze social-choice voting
- [x] LLM-based statement generation (Claude)
- [x] LLM-as-judge reward model
- [x] Deliberation loop orchestration
- [x] Opinion matrix and K-means clustering
- [x] Group-informed consensus detection
- [x] Pipeline session combining both phases
- [x] FastAPI REST endpoints
- [ ] Mediator dashboard (React frontend)
- [ ] Real-time session management (WebSockets)
- [ ] Export to Polis-compatible format
- [ ] Fine-tuned reward model (replace LLM-as-judge with trained model)
- [ ] Multi-language support
- [ ] Persistent database backend (PostgreSQL)
- [ ] Authentication and multi-tenancy

---

## References

- Tessler, M. H., et al. (2024). *AI can help humans find common ground in democratic deliberation*. **Science**, 386(6719). https://doi.org/10.1126/science.adq2852
- Google DeepMind Habermas Machine: https://github.com/google-deepmind/habermas_machine
- Computational Democracy Project — Polis: https://github.com/compdemocracy/polis
- Small, C., et al. (2021). *Polis: Scaling Deliberation by Mapping High Dimensional Opinion Spaces*. https://arxiv.org/abs/2005.12729

---

## Licence

MIT — see [LICENSE](LICENSE). Note that if you incorporate code from the Habermas Machine repository directly, that code is Apache 2.0; Polis is AGPLv3. Check licences carefully before production deployment.
