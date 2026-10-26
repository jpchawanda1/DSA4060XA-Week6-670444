# DSA 4060XA – Week 6: Evaluating Recommender Systems

**Name:** J. P. Chawanda
**Student ID:** 670444
**Submitted:** 26 October 2026

## Lab objective

Build three recommenders over the same movie-rating data and evaluate them on three
different families of metric, in order to show that "best model" is not a property of a
model but of a stated objective.

The three families are:

1. **Rating prediction accuracy** — MAE and RMSE on held-out ratings.
2. **Ranking quality** — Precision@K, Recall@K and NDCG@K over Top-3 lists.
3. **Beyond-accuracy behaviour** — catalogue coverage, novelty and intra-list diversity.

## Repository contents

```
DSA4060XA-Week6-670444/
├── DSA4060XA_Week6_Guided_Evaluation_Lab.ipynb   # the lab, executed top to bottom
├── DSA4060_Week6_Movies.csv                      # 20 movies, genre metadata
├── DSA4060_Week6_Ratings.csv                     # 120 ratings, 12 users
└── README.md
```

**Data:** 12 users × 20 movies = 120 explicit ratings on a 1–5 scale. Every user rated
exactly 10 movies, so the matrix is 50% dense. Ratings are centred rather than top-heavy
(mean 3.23; 49 of 120 are a 3; no 1s occur), and only 35% reach 4 or above.

**Split:** leave-one-last-out — each user's most recent rating by timestamp is held out,
giving 108 training and 12 test interactions.

## Models compared

| Model | How it scores an item |
|---|---|
| **Popularity baseline** | Mean training rating per movie, falling back to the global mean; ranked by mean rating then rating count |
| **Content-based** | Genre one-hot features; user profile is the rating-weighted mean of their rated items; score is cosine similarity mapped to 1–5 via `1 + 4·sim` |
| **Item-based CF** | Item–item cosine similarity on the training user–item matrix (unrated filled with 0); prediction is a similarity-weighted average of the user's own ratings |

## Results

| Model | MAE | RMSE | Precision@3 | Recall@3 | NDCG@3 | Coverage | Novelty | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity Baseline | 1.086 | 1.201 | 0.067 | 0.200 | 0.100 | 0.35 | 1.689 | 0.736 |
| Content-Based | 1.094 | 1.381 | 0.067 | 0.200 | **0.200** | 0.70 | 1.344 | 0.542 |
| Item-Based CF | **0.924** | **1.070** | 0.000 | 0.000 | 0.000 | **0.85** | **1.695** | **0.780** |

## Best model for rating prediction

**Item-based collaborative filtering** — lowest MAE (0.924 vs the baseline's 1.086) and
lowest RMSE (1.070 vs 1.201).

This win should be read carefully rather than at face value. CF's predictions span only
2.74–3.48, a range of 0.73 rating points on a 5-point scale. Because every cosine
similarity is positive, the similarity-weighted average is pulled towards each user's mean,
so the model is effectively predicting "about 3.2" with a small personalised nudge. On data
whose true mean is 3.23 and whose mode is 3, that caution minimises average error almost by
construction. The 0.16 MAE margin is also measured on **12 held-out ratings**, where one
prediction moving by two points would swing MAE by about 0.17 and erase the gap.

The content model is worst on RMSE (1.381) because `1 + 4·sim` is an uncalibrated mapping —
cosine similarity measures direction in genre space, not how much a user liked anything.
Its MAE should not be treated as evidence about the model's usefulness.

## Best model for ranking

**Content-based filtering** — tied on Precision@3 and Recall@3 (0.067 / 0.200) and the
outright winner on NDCG@3 (0.200 vs the baseline's 0.100).

The ranking picture inverts the rating picture. Item-based CF, the MAE and RMSE winner,
scores **zero on all three ranking metrics** — it never once placed a relevant held-out
movie in a user's Top-3. Its ranking is driven by differences within a near-constant score,
which is the same hedging that won it the rating metrics.

Popularity and content tie on Precision and Recall with one hit each, and NDCG is what
separates them: the content model's hit landed at rank 1, the baseline's at rank 3. Same
hits, better placement — which is exactly the question NDCG exists to answer.

Absolute values are low across the board. With 5 relevant items, 3 slots and 11+ candidates
per user, random guessing scores near 0.09 Precision@3, so the models are performing around
chance. The *ordering* between them is the interpretable result, not the magnitudes.

### Extension (Option A) – varying K

| K | Popularity P@K / R@K | Content P@K / R@K | CF P@K / R@K |
|---:|---:|---:|---:|
| 1 | 0.000 / 0.000 | **0.200 / 0.200** | 0.000 / 0.000 |
| 3 | 0.067 / 0.200 | 0.067 / 0.200 | 0.000 / 0.000 |
| 5 | 0.040 / 0.200 | **0.080 / 0.400** | 0.040 / 0.200 |

Recall rises monotonically with K — a longer list can only contain more relevant items.
Precision generally falls because it divides by K, so the baseline's single hit dilutes from
1/3 = 0.067 to 1/5 = 0.040. But precision *can* rise with K when a longer list adds hits
faster than slots: content-based goes 0.067 → 0.080 because it gains a second hit
(1/3 → 2/5).

K = 1 is the most discriminating setting and it reorders the models — content-based scores
0.200 while both others score zero, making it the only model whose single best guess was
ever right. The models rank differently at K = 1 than at K = 3, so choosing K is a modelling
decision that should mirror how many items the product actually shows, not a reporting
detail.

One structural note: each evaluated user has exactly one relevant held-out item, so per-user
Recall@K is either 0 or 1 and the mean is simply the fraction of the 5 users whose item was
found. Recall@K here is a hit-rate in disguise.

## Coverage, novelty and diversity findings

**Coverage** separates the models far more sharply than any accuracy metric: 0.35
(popularity), 0.70 (content), 0.85 (CF). The baseline shows only 7 distinct movies across
all 12 users — 13 of 20 titles are never recommended to anyone, and it only reaches 7 rather
than 3 because each user's already-seen movies get filtered out, forcing it down its fixed
ranking. CF reaches 17 of 20. This is the metric that directly answers "are the same titles
dominating", and the one on which the baseline fails outright.

**Novelty** is nearly tied between CF (1.695) and the baseline (1.689), with content-based
lowest (1.344). The baseline scoring well here is a quirk of the measure rather than a
virtue: novelty is computed from *rating frequency*, while the baseline ranks by *mean
rating* with count only as a tie-break, so its favourites (*Broken Hearts*, 6 ratings;
*The Promise*, 2 ratings) are well-liked rather than widely rated and therefore look novel.
Novelty-by-interaction-count and coverage are not measuring the same thing on this data, and
coverage is the more trustworthy signal of concentration — the baseline's 0.35 coverage shows
it really is showing everyone the same handful of films, even though novelty doesn't flag it.

**Diversity** is highest for CF (0.780), then popularity (0.736), with content-based clearly
lowest (0.542). Content-based scoring lowest is the model working as designed, not a defect:
it selects items *because* they match one genre profile, so its three winners resemble each
other by construction. That is the classic content-based filter bubble, and it is the price
of explainability. Diversity is also not aligned with relevance here — CF has the most varied
lists and the worst hit rate, because a list can be diverse by being indiscriminate.

**The cross-cutting finding** is that the content model's personalisation is weaker than its
NDCG win suggests. For User 1 it recommended the *same three films as the baseline*, merely
reordered. The cause is visible in the user profile: Drama tops it (1.67) not from
enthusiasm but from frequency — it appears in four of User 1's nine training movies, rated
3, 3, 3 and 2 — while Sci-Fi (1.00) reflects the genuinely high ratings (*Galaxy Mission* 5,
*Digital Dreams* 4). Because no rating in this data is negative, a genre seen often but
liked lukewarmly outranks one loved but seen twice, and the model's favourites drift towards
the catalogue's most common genres. Mean-centring ratings before weighting would correct
this.

## Final recommended model

For the stated objective — *"users should discover relevant movies, but the same popular
titles should not dominate every list"* — I recommend **content-based filtering, deployed
with a diversity re-rank**, and held as a hypothesis to test online rather than a settled
result.

The objective has two clauses and each eliminates a different model. The baseline fails the
anti-popularity clause (0.35 coverage, 7 distinct titles). CF fails the relevance clause
(0.000 on every ranking metric) despite leading coverage, novelty and diversity.
Content-based is the only model not disqualified by either: it wins NDCG@3 and is the best
model at both K = 1 and K = 5, while doubling the baseline's coverage. Clearing both bars
matters more here than leading any single column.

The metrics that drove the choice were **coverage** (the direct measure of the
anti-popularity clause), **NDCG@3** (the only metric separating popularity from content, and
the one matching a short ranked list where position matters), and **Precision@3 / Recall@3**
used as a floor rather than a ranking — CF's zero rules it out regardless of its other
scores. MAE and RMSE were deliberately *not* weighted heavily: they favour CF, which earns
them by predicting near the mean, and that behaviour is worthless for choosing three items.

**Trade-off accepted:** narrower lists for more reliable relevance. Content-based has the
lowest diversity (0.542) and novelty (1.344), so users get recommendations close to what
they already watch — the filter-bubble risk — and I give up CF's 0.16 MAE advantage, which
is acceptable because the product ranks items rather than displaying predicted scores. The
concrete cost falls on users whose tastes sit between genre clusters. Mitigations: cap the
Top-3 at two movies sharing a dominant genre, and reserve one slot for exploration.

## Limitations

1. **The ranking metrics rest on five users.** Only 5 of 12 held-out ratings reach the
   relevance threshold of 4, and each of those users has exactly one relevant item. Every
   ranking number is an average over 5 observations, Precision@3 moves in steps of 0.067 per
   hit, and the entire content-vs-popularity NDCG gap is one hit landing at rank 1 instead of
   rank 3. The recommendation above is directionally reasonable but not statistically
   supported.

2. **The dataset is too small and too dense to be representative.** 12 users, 20 movies and
   50% density is unlike any production catalogue, where density is typically well under 1%.
   Item–item similarities computed from 12 rating vectors are extremely unstable, and the
   cold-start and long-tail problems that dominate real systems cannot appear at all — every
   movie has training ratings, so the baseline's global-mean fallback never fires.

3. **`fillna(0)` conflates "not rated" with "rated zero".** Item similarity is therefore
   driven substantially by *co-rating* — which movies the same people happened to rate —
   rather than by agreement on quality. Two movies rated by the same crowd look similar even
   if that crowd disagreed about them, which plausibly contributes to CF's ranking failure.

4. **The content model's rating scale is uncalibrated.** `1 + 4·sim` is a teaching
   convenience with no grounding in the rating distribution; it compresses predictions into
   1.59–3.99 against held-out ratings averaging 3.2. The content model's MAE and RMSE are
   therefore artefacts of this mapping and are not comparable to the other two models'.

5. **A single relevance threshold and a single split.** Relevance is a hard cut at rating
   ≥ 4, which treats a 4 and a 5 as equivalent and a 3 as worthless; and leave-one-last-out
   produces exactly one split, with no cross-validation or repeated sampling, so none of
   these numbers carries an error bar. Offline relevance is also a proxy — a held-out rating
   shows what a user chose to watch, not what they would have chosen from the list the model
   produced.

## Reproducibility

The notebook was executed top to bottom from a fresh kernel; all 27 code cells ran without
error and outputs are saved in the committed `.ipynb`.

Requirements: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

```bash
jupyter nbconvert --to notebook --execute --inplace DSA4060XA_Week6_Guided_Evaluation_Lab.ipynb
```

All results are deterministic — no model in the lab uses random initialisation or sampling,
so re-running reproduces the figures in this README exactly.
