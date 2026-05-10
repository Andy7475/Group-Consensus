# Walkthrough

This tool helps a group reach consensus on a topic. It has three phases: extract statements from opinions, collect votes, then synthesise a group statement from what everyone agreed on.

The examples below are from a real run on the topic: *"Where should we hold the combined summer social for engineering and marketing?"*

---

## Phase 1 — Extract atomic statements (`01_mediate.ipynb`)

Start by setting the topic and pointing to a CSV of opinions.

```python
TOPIC        = "Where should we hold the combined summer social for engineering and marketing?"
OPINIONS_CSV = "example_opinions.csv"
```

The CSV should have one row per person with a name column and a free-text opinion column. The notebook loads these into structured `Opinion` objects.

```python
for i, row in df.iterrows():
    pid  = f"p{i}"
    participants.append(Participant(id=pid, name=str(row[NAME_COL]).strip()))
    opinions.append(Opinion(participant_id=pid, text=str(row[OPINION_COL]).strip(), session_id=SESSION_ID))
```

Opinions are free-form. For example:

> *"I have two young kids so I need to be home by 8pm — this is a hard constraint not a preference. Beyond that I want somewhere we can actually have a conversation, not shout over noise."* — Sarah Chen
>
> *"Step-free access is non-negotiable for me. I'm also vegetarian so the food options matter. I'd prefer somewhere central — within easy distance of Waterloo."* — Priya Sharma
>
> *"An icebreaker activity would really help the two teams actually connect rather than just co-existing in the same room. Escape room, cooking class, cocktail making — something hands-on."* — Kwame Asante
>
> *"The most important thing for me is that it doesn't feel like a work meeting with nicer food. I'm less keen on structured activities where you're assigned to a group."* — Tom Walsh

The LLM reads all opinions and pulls out short, single-idea statements (5–15 words each). These are the raw building blocks.

```python
raw_statements = asyncio.run(
    model.extract(topic=TOPIC, opinions=opinions, participants=participants)
)
```

It then compresses duplicates. Semantic equivalents are merged into one. Genuine contradictions become `CONTESTED` statements and are kept — they represent real disagreements worth surfacing in the vote.

```python
compressed = asyncio.run(
    model.compress(topic=TOPIC, statements=raw_statements)
)

atomic    = [s for s in compressed.statements if s.type == StatementType.ATOMIC]
contested = [s for s in compressed.statements if s.type == StatementType.CONTESTED]
```

**Merging in practice.** Sarah said she needs to leave by 8pm; Yuki said she needs to leave by 9pm. These were extracted as two separate statements. During compression, the 9pm statement was absorbed into the 8pm one — an event ending at 8pm satisfies both constraints, so only the stricter version needs to be voted on.

Similarly, Priya's *"step-free access is non-negotiable"* and Yuki's *"step-free access throughout, not just the entrance"* were merged into a single, stronger statement: *"The venue must have comprehensive step-free access throughout, accommodating wheelchair users fully."*

**Contested pairs.** Kwame and Nina wanted a structured icebreaker to force cross-team mixing. Dan and Tom explicitly didn't want to be assigned to groups. The LLM recognised these as a genuine contradiction and produced two opposing statements rather than collapsing them:

> `CONTESTED` — *"There should be a structured activity at the start of the event to facilitate cross-team mixing and reduce departmental clustering"*
>
> `CONTESTED` — *"There should be no structured activity; a good restaurant with relaxed dining is sufficient"*

Both go into the voting pool so the data can show which groups hold which position.

**Stop here.** Review the list. If anything important from the original opinions didn't survive extraction, add it manually:

```python
manual_additions = [
    "The venue has some outdoor space",
]
additional_stmts = [
    Statement(text=t, type=StatementType.ATOMIC, session_id=SESSION_ID)
    for t in manual_additions
]
final_statements = compressed.statements + additional_stmts
```

The notebook saves `statements.json` and `label_map.json` (which maps `Statement 1`, `Statement 2`, etc. to the full text). It also prints a voting form template you can paste into your chosen collection tool.

---

## Collect votes (outside the notebooks)

Share the voting form with participants. Each person votes Agree / Pass / Disagree on every statement.

**In an Office 365 environment** the natural fit is **Microsoft Forms**. Create a new form, add one multiple-choice question per statement using the printed labels (`Statement 1`, `Statement 2`, etc.) as the question titles, and set the three options to `Agree`, `Pass`, `Disagree`. Send via Teams or Outlook. When responses are in, open the form in Excel Online (Forms → Open in Excel) and download as `.xlsx` or export as CSV.

You can also collect votes in a shared **Excel Online** or **SharePoint** spreadsheet — paste the template directly and have people fill in their column. Either way, export to CSV before loading into Phase 2.

The expected format is one row per person:

| Name  | Statement 1 | Statement 2 | Statement 3 |
|-------|-------------|-------------|-------------|
| Sarah | Agree       | Pass        | Disagree    |
| James | Pass        | Agree       | Agree       |

Column headers must match the labels from `label_map.json` exactly. Empty cells are treated as Pass.

---

## Phase 2 — Analyse votes (`02_analyse.ipynb`)

Set the path to the votes CSV, then load it alongside the `label_map.json` produced in Phase 1.

```python
VOTES_CSV   = "example_votes.csv"
SESSION_DIR = "session_data"
```

Votes are parsed to numeric values (`Agree` → 1, `Pass` → 0, `Disagree` → −1) and loaded into an opinion matrix.

```python
matrix = OpinionMatrix()
# ... registers statements and participants, then:
vote = Vote(
    participant_id=pid,
    statement_id=stmt.id,
    value=VoteValue(val),
    session_id=session_data["session_id"],
)
matrix.add_vote(vote)
```

The clustering engine groups participants by similarity of voting pattern using UMAP + k-means. The silhouette score tells you how cleanly separated the groups are (>0.5 is clear, <0.2 means no real structure).

```python
engine     = ClusteringEngine(max_clusters=5)
clustering = engine.run(matrix)
```

In this example, 19 participants split into two clear groups (silhouette score: 0.97):

> **Group 1** (12 people): Sarah, Priya, Dan, Lisa, Tom, Aisha, Yuki, Ben, Fatima, Alex, Sophie, Ravi
>
> **Group 2** (7 people): James, Emma, Kwame, Marcus, Ryan, Nina, Chioma

The 2D opinion map plots everyone as a dot — people close together voted similarly.

```python
fig = px.scatter(plot_df, x="x", y="y", color="cluster", text="name", ...)
fig.show()
```

The consensus detector finds **bridging statements** (approved by ≥60% of every group) and **divisive statements** (sharp splits between groups).

```python
detector  = ConsensusDetector(min_cluster_approval=0.6)
consensus = detector.detect(list(stmt_objects.values()), clustering)
```

**Bridging statements** — both groups agreed:

| Statement | G1 approval | G2 approval |
|-----------|-------------|-------------|
| A private or semi-private space should be used to aid conversation and hearing | 100% | 100% |
| The venue should have energy and atmosphere | 100% | 100% |
| Good quality food is a priority, with vegetarian options available | 100% | 100% |
| The company should cover the full cost of the evening including food and drinks | 80% | 100% |
| The venue should be quiet enough for conversation without needing to raise voices | 100% | 67% |

**Divisive statements** — where the groups split sharply:

| Statement | G1 approval | G2 approval |
|-----------|-------------|-------------|
| The event must end by 8pm | 100% | 0% |
| The venue should be in central London, within easy distance of Waterloo | 0% | 100% |
| The atmosphere should feel relaxed and celebratory, not formal or work-like | 100% | 0% |

Group 1 prioritised early finish, seating, and non-alcoholic options. Group 2 cared more about location, cross-team mixing, and a structured activity. These divisions are where facilitation energy is most needed.

A heatmap shows approval rate per statement per group at a glance. The mediator brief at the end is plain text you can copy into a facilitation document.

Results are saved to `consensus_result.json`.

---

## Phase 3 — Synthesise group statement (back to `01_mediate.ipynb`)

Once `consensus_result.json` exists, the final cell in `01_mediate.ipynb` auto-loads the bridging statements and synthesises a single coherent group statement from them.

```python
consensus_path = OUTPUT_DIR / "consensus_result.json"
if consensus_path.exists():
    saved = ConsensusResult.from_file(consensus_path)
    # resolves bridging statement labels automatically
```

```python
group_statement = asyncio.run(
    model.synthesise(topic=TOPIC, winning_statements=bridging_stmts)
)
```

The result is a plain-English recommendation built only from what the group agreed on. From this run:

> *The venue should strike a balance between energy and atmosphere on one hand, and a comfortable acoustic environment on the other. It must be quiet enough for easy conversation without guests needing to raise their voices, while still feeling lively and engaging. A private or semi-private dining space is strongly preferred — this protects the group from the noise of a general public crowd while preserving a sense of occasion.*
>
> *Good quality food is a clear priority. The venue must offer vegetarian options as standard, and a set menu format is well suited to the practical needs of a group booking of this size. The company will cover the full cost of the evening, including food and drinks, so guests can relax without any concern about the bill.*
>
> *There is a recognised tension in the group around structured activities... Both positions reflect genuine priorities — cross-team connection, and a relaxed, enjoyable atmosphere — and the final event design should seek to honour both.*

The result is saved to `session_data/group_statement.txt`. It represents only what everyone could agree on — the genuine common ground, not a watered-down compromise.

---

## Optional — Mediator comparison (`03_mediator.ipynb`)

This notebook runs an alternative approach on the same opinions: the Habermas Machine deliberation loop. No voting forms, no clustering — just raw opinions in, group statement out.

```python
config   = SessionConfig(session_id=SESSION_ID, topic=TOPIC, num_candidate_statements=5)
mediator = AsyncMediator(config)
result   = asyncio.run(mediator.run(topic=TOPIC, participants=participants, opinions=opinions))
```

The LLM generates 5 candidate group statements, then predicts how each participant would rank them based on their stated opinion. Schulze voting aggregates the predicted rankings to pick a winner.

```python
# Reconstruct full Schulze ranking for display
ranked_indices = schulze_ranking(index_rankings, len(candidates))
```

The notebook prints all candidates in rank order, saves the winner to `session_data/mediator_statement.txt`, then loads both outputs side by side for comparison.

**What the two approaches produce differently:**

The Mediator tends toward fluent, holistic prose — it reads like a document written by one thoughtful author. The atomic/voting statement is more granular and structurally tied to the specific points the group voted on. Both acknowledge tensions, but in different ways: the Mediator smooths them into narrative; the atomic approach preserves them as explicit contested statements in the vote record.

The deeper difference is legitimacy. In the Mediator flow, participants never see the candidate statements — the Schulze ranking is the LLM's prediction of what they'd prefer, not an actual choice. In the atomic flow, people voted. That distinction matters most when the group needs to own the outcome, not just receive it.
