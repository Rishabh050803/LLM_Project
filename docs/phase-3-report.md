# Phase 3 report: embeddings and retrieval

## Completed

- Added page-aware deterministic chunking.
- Added batched OpenAI `text-embedding-3-small` generation.
- Added a SQLite embedding cache to avoid duplicate embedding charges.
- Added a persistent Chroma collection at `data/index/vector_db`.
- Indexed 40 accepted text-bearing documents into 596 chunks.
- Added cosine-similarity retrieval with optional exact metadata filters.
- Added final grounded answer generation through the configured OpenAI chat model.
- Added source filename, URL, and page citations to every answer response.

## Deliberate design decision

There is no LLM reranking stage. Results are selected directly by Chroma cosine
similarity, with optional metadata filtering before the result set is returned.
This keeps latency, token use, and cost predictable.

## Commands

```bash
env -u OPENAI_API_KEY academic-rag index
env -u OPENAI_API_KEY academic-rag ask "What is the formula used for CGPA?"
```

The `env -u` is needed only because an older exported `OPENAI_API_KEY` can
override the key loaded from the shared parent `.env` file.

## Smoke test

The query `What is the formula used for CGPA?` retrieved three relevant chunks
and generated a grounded answer with citations from the indexed documents.

## Additional end-to-end query tests

All tests used the same pipeline: query embedding, Chroma cosine retrieval,
top-three evidence chunks, and one final `gpt-4o-mini` answer call. No LLM
reranking was used.

### 1. PhD admission criteria

**Question:** `What are the admission criteria for PhD applicants?`

**Generated answer:** The answer identified the regular admission process,
personal interview, GATE requirements for Engineering and Sciences, UGC NET for
Management/Humanities, and a 7.5 CGPA requirement for the MPDD pathway.

**Retrieved sources:**

- `3024_DualDegree.pdf`, pages 2–3, scores 0.464 and 0.412
- `1002.pdf`, page 1, score 0.427

**Result:** Relevant and well grounded. The answer correctly distinguishes the
general scholarship admission requirement from the MPDD-specific CGPA rule.

### 2. UG/PG internship rules

**Question:** `What are the main rules for undergraduate and postgraduate internships?`

**Generated answer:** The answer covered Dean (Academic) approval, supervisor
and committee recommendations, the required Office Order, UG semester and
backlog requirements, the 16-credit waiver, PG third-semester timing, and
evaluation requirements.

**Retrieved source:**

- `InternshipGuidelinesUGPG.pdf`, pages 1–3, scores 0.575, 0.570, and 0.535

**Result:** Strong retrieval and answer grounding. All three chunks came from
the same authoritative academic guideline document.

### 3. Academic integrity

**Question:** `What is the institute policy for academic integrity?`

**Generated answer:** The answer reported the 15% similarity threshold, the
conditional 20% threshold when each individual source remains below 1%, and
disciplinary consequences for conduct violations and unfair means.

**Retrieved sources:**

- `Guidelines_AcademicIntegrity.pdf`, page 1, score 0.600
- `UGCoursesStudy2022-2023.pdf`, page 16, scores 0.514 and 0.507

**Result:** Correctly grounded. The first source directly answers the question;
the other two provide related conduct and discipline context.

### 4. Fee refund policy

**Question:** `What is the fee refund policy for students?`

**Generated answer:** The answer covered caution-money refunds, medical and
non-medical semester withdrawal, registration state, No Dues requirements, and
forfeiture of semester fees after registration.

**Retrieved sources:**

- `UGCoursesStudy2022-2023.pdf`, pages 7–8, scores 0.678 and 0.593
- `UG_2019-20.pdf`, page 6, score 0.518

**Result:** Useful, but version sensitivity must be evaluated. Fee and refund
rules can change over time, so future answers should expose document dates and
prefer the newest applicable policy.

### 5. Out-of-scope hostel-menu question

**Question:** `What is the hostel mess menu for students?`

**Generated answer:** The answer explicitly stated that no menu was present in
the indexed documents and only related mess-advance fee references were found.

**Retrieved sources:** Three historical fee documents with scores 0.528, 0.471,
and 0.440.

**Result:** The answer did not invent a menu, which is good. However, the
retrieval scores are relatively weak and the system still supplied unrelated
fee chunks. This is a clear evaluation target for a minimum similarity
threshold or an explicit “no relevant evidence” classifier.

## Answer-generation readiness

The answer-generation stage is complete for the Phase 3 MVP:

- It receives only retrieved academic context.
- It is instructed not to invent unsupported facts.
- It cites retrieved sources inline.
- It returns source metadata and page numbers alongside the answer.
- It uses `store=False` for the Responses API request.
- It does not use LLM reranking.

The next phase should be evaluation rather than more generation features. The
evaluation set should include answerable questions, unanswerable questions,
conflicting document versions, metadata-filtered questions, and questions that
are outside the academic corpus. Recommended metrics are retrieval recall at
K, citation correctness, groundedness, answer completeness, and abstention
quality.

## Baseline evaluation

The baseline was tested without query expansion and without LLM reranking.
The test set contains seven answerable academic questions and three
unanswerable or out-of-scope questions. The raw dataset is stored in
[`baseline_queries.jsonl`](../data/evaluation/baseline_queries.jsonl), and the
full generated responses are stored in
[`baseline_results.json`](../data/evaluation/baseline_results.json).

### Measured results

| Metric | Result | Interpretation |
|---|---:|---|
| Retrieval recall@3 | 1.00 | All seven answerable cases retrieved an expected source keyword. |
| Citation format validity | 1.00 | Answerable responses cited valid retrieved-source indexes; abstentions were not required to cite. |
| Abstention accuracy | 1.00 | All three unanswerable cases explicitly declined to invent an answer. |
| Mean end-to-end latency | 2256.7 ms | Includes one query embedding call, Chroma retrieval, and one final answer call. |

### Baseline observations

- The baseline successfully retrieved the expected documents for CGPA
  conversion, PhD admission, internships, academic integrity, fee refunds,
  registration, and MPDD requirements.
- The out-of-scope questions about the hostel menu, library hours, and sports
  facilities caused the answerer to abstain correctly.
- Out-of-scope retrieval still returned three weakly related chunks because the
  current retriever always requests `top_k` results. This is the main baseline
  weakness and should be addressed with a similarity threshold or a calibrated
  no-evidence rule.
- Retrieval recall@3 here is keyword-based, not human-judged recall. Citation
  validity checks citation numbering, not whether every claim is factually
  supported by the cited text.
- Abstention accuracy uses deterministic phrase detection and should later be
  supplemented with human or model-judged groundedness.

### Evaluation command

```bash
env -u OPENAI_API_KEY academic-rag evaluate-baseline
```

The baseline harness is implemented in
[`app/evaluation/baseline.py`](../app/evaluation/baseline.py), and the dataset
is intentionally separate from the application so it can be reused when we
compare query expansion later.

## 100-query baseline results

The generated 100-query run produced the following aggregate report:

| Metric | Result |
|---|---:|
| Cases | 100 |
| Answerable cases | 80 |
| Unanswerable cases | 20 |
| Answerable retrieval recall@3 | 75% |
| Citation-format validity | 96% |
| Abstention accuracy | 95% |
| Mean latency | 3552 ms |
| Median latency | 2128 ms |
| 90th-percentile latency | 3075 ms |
| Slowest case | 65.2 seconds |

### Interpretation

The baseline is functional but not yet strong enough to call production-ready.
It found the expected source for 60 of 80 answerable questions. The 20 misses
are concentrated in historical fee-structure questions, where several similar
documents contain overlapping values. In those cases, the top three results
often came from a related year or programme rather than the exact expected
source. This is a document/version retrieval problem, not evidence that all
answers were factually wrong.

Five answerable questions caused the answerer to abstain because the exact
document was not retrieved. That is safer than inventing a value, but it lowers
answer coverage. Four citation-format failures also corresponded to answers
that abstained rather than fabricating citations.

The 95% abstention score means 19 of 20 out-of-scope cases were recognized by
the current heuristic. The remaining case was an evaluator phrase-matching
miss, not necessarily a model hallucination; the generated answer used a
different abstention wording.

Latency is usually around 2–4 seconds, but one case took 65.2 seconds. That
outlier needs investigation before production use, especially around provider
timeouts and retry behavior.

### Important dataset limitation

The 100 questions were generated from the indexed corpus, and many are about
historical fee documents. The results are therefore useful for comparing the
baseline against query expansion, but they are not a complete measure of user
quality. Before final evaluation, the dataset should be balanced across
admissions, regulations, internships, examinations, academic integrity,
programmes, fees, temporal conflicts, and out-of-scope questions.
