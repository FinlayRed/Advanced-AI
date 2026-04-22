# Bristol Regional Food Network 

# Recommendation Subsystem (Task 1)

This repository implements the AI-based recommendation and re-order subsystem
for the Bristol Regional Food Network digital marketplace. It is one of the
four AI components of the wider group project for the *Advanced Artificial
Intelligence* module (UFCFUR-15-3, 2025-26).

**Owner:** Sam (Person A) · **Module:** Advanced AI · **Group project**

---

## What this subsystem does

Given a customer's purchase history, the service produces two kinds of
recommendation that the marketplace UI exposes to end-users:

| Endpoint | Purpose |
|---|---|
| `quick_reorder(customer_id, k)` | The customer's top-k recurring staples, ranked by how often *and* how recently they have bought them. Powers a one-tap "re-order favourites" button. |
| `recommend_next_order(customer_id, k)` | Top-k personalised recommendations for the customer's next basket, optionally re-ranked for producer fairness. |

Both methods return a ranked list of `product_id`s (best first) that the UI
or API layer can dereference against the product catalogue.

## Architecture at a glance

```
                  ┌────────────────────────────────────────┐
                  │     RecommendationService              │
                  │     (the public interface)             │
                  └────────┬──────────────────┬────────────┘
                           │                  │
              ┌────────────▼────┐    ┌────────▼─────────┐
              │ FrequencyRecency│    │ NMFRecommender   │
              │  Recommender    │    │ (sklearn NMF on  │
              │  (baseline)     │    │  log-scaled      │
              └─────────────────┘    │  user x item)    │
                                     └──────────────────┘
                           │                  │
                           └────────┬─────────┘
                                    │
                  ┌─────────────────▼────────────────────┐
                  │ rerank_for_producer_diversity()      │
                  │ + FeedbackLogger (override capture)  │
                  └──────────────────────────────────────┘
```

Two recommenders are kept side-by-side so the service can be A/B tested at
runtime and so the technical report can compare a strong baseline against a
proper machine-learning approach. The production wrapper adds producer
fairness post-processing and override logging for monitoring.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end demo (generates data, trains both models,
#    evaluates, demonstrates fairness + cold-start + feedback logging,
#    saves the deployable .joblib artefact)
python -m src.demo

# 3. Run the test suite
pytest tests/ -v

# 4. Generate the charts used in the demo and the report
python -m src.charts   # writes PNGs into reports/figures/
```

## Repository layout

```
sam_recommender/
├── src/
│   ├── data_generator.py    Synthetic purchase log generator
│   ├── baseline.py          Frequency × recency baseline
│   ├── mf_model.py          NMF collaborative filtering
│   ├── evaluation.py        Chronological split + ranking metrics
│   ├── fairness.py          Producer-diversity re-ranker
│   ├── feedback.py          Override / outcome JSONL logger
│   ├── recommender.py       Public RecommendationService wrapper
│   ├── demo.py              End-to-end demo script
│   └── charts.py            Matplotlib charts for demo + report
├── tests/
│   └── test_recommender.py  Smoke tests (run in < 5 s)
├── data/                    Generated CSVs + saved .joblib artefact
├── logs/                    Feedback JSONL log
├── reports/figures/         PNG charts for the report and demo
├── requirements.txt
└── README.md
```

## Headline evaluation results

Chronological 80/20 split on 800 synthetic customers, 43 products,
~73,000 order lines (full numbers regenerable via `python -m src.demo`):

| Metric         | Baseline | NMF       |
|----------------|----------|-----------|
| Precision@5    | 0.547    | **0.618** |
| Precision@10   | 0.495    | **0.576** |
| Recall@10      | 0.436    | **0.508** |
| NDCG@10        | 0.554    | **0.634** |
| Hit Rate@10    | 0.984    | **0.994** |

NMF wins on every metric, with the largest gain on NDCG (the position-aware
metric) — i.e. the NMF model is better at putting the right items at the
*top* of the list, which is what matters for UI relevance.

## Integration notes

For the wider system integration owned by Finlay (Person C):

* The deployable artefact is a single `.joblib` file produced by
  `RecommendationService.save(path)`. Load it with
  `RecommendationService.load(path)` — no other state is needed.
* The two public methods (`quick_reorder`, `recommend_next_order`) are the
  stable contract. Their signatures will not change between training runs,
  even when the underlying model is swapped.
* The feedback logger writes newline-delimited JSON to a configurable path.
  In production this should be redirected to the platform's event database;
  the format will not change.
* The Task 3 "AI engineer uploads a new model" workflow is implemented
  by retraining offline (`service.fit(new_orders)`), saving the
  `.joblib`, and dropping it into the deployment directory.

## Design choices justified in the report

* **Synthetic data over real data.** The case study explicitly permits
  synthetic data and there is no real Bristol marketplace order log to
  draw on. Synthetic data also lets us inject ground-truth seasonality and
  producer-fairness scenarios that we can then verify the model exploits.
* **NMF over deeper architectures.** Catalogue size is small (~50
  products), so sequence models, two-tower neural networks, and graph
  neural networks would overfit. NMF gives the best precision-per-parameter
  on this scale and is interpretable.
* **Chronological evaluation, not random split.** Random splits leak
  future information and consistently overestimate recommender accuracy.
  Time-based splits are the standard in the literature for this reason.
* **Producer-fairness as a post-processing step, not a training
  constraint.** Easier to tune, easier to ablate (the report shows the
  impact with and without), and decouples fairness policy from model
  retraining.

## Limitations and known issues

* The synthetic dataset is generated, not measured — the absolute metric
  numbers are not directly comparable to public benchmarks. The
  *relative* comparison between baseline and NMF is meaningful.
* NMF must be re-fitted from scratch when new customers arrive. For the
  scale of the Bristol marketplace this is fine (sub-second on a laptop)
  but a production deployment should either warm-start or move to
  incremental ALS.
* Feedback log → retrain loop is not automated; only the data plumbing
  exists. Out of scope for the assignment but documented in the report.

## Generative AI usage

In line with the module's expectation that students trial and report on
generative AI use, every prompt and the resulting evaluation is recorded
in `reports/genai_log.md`. See the technical report for the reflection on
what worked and what didn't.

# Quality inspection subsystem (Task 2)

This part of the project implements the AI-based quality inspection workflow
for the Bristol Regional Food Network digital marketplace. It is one of the
four AI components of the wider group project for the *Advanced Artificial
Intelligence* module (UFCFUR-15-3, 2025-26).

**Owner:** Yas (Person B) · **Module:** Advanced AI · **Group project**

This repository includes tools for producers to check the quality of fruit and
vegetables. The `src/quality/` package scores produce quality, assigns grades
(A, B, or C), updates each producer's inventory, and recommends a next action
(e.g. sell normally, discount, clearance, or manual review).

## What this subsystem does

At **training** time, CSV rows supply image paths and target scores; at
**inference**, the model predicts class and scores from the image alone, then
rule-based grading assigns A/B/C. The main pieces are:

| Component | Purpose |
|---|---|
| `assign_grade(color, size, ripeness)` | Grades each item (A, B, or C) using fixed thresholds. |
| `ProducerInventory.process_inspection(...)` | Updates each producer's stock by grade & saves inspection records. |
| `src.quality.train` | Trains the model to classify produce condition & predict quality scores. |
| `src.quality.infer` | Checks one image & returns the predicted class, confidence, scores, & final grade. |

The grading policy in `src/quality/grading.py` checks **C before B** (the
stricter rule wins):

* Grade `C` if **any** metric is below the C cut-offs (`color < 65`,
  `size < 70`, `ripeness < 60`).
* Else grade `B` if **any** metric is below the B cut-offs (`color < 75`,
  `size < 80`, `ripeness < 70`).
* Else grade `A`.

## Architecture

```
            ┌──────────────────────────────────────────────┐
            │             QualityNet (ResNet18)            │
            │       class head + 3-score regression head   │
            └───────────────────┬──────────────────────────┘
                                │
                ┌───────────────▼────────────────┐
                │ quality_breakdown() + thresholds│
                │ assign_grade() => A / B / C     │
                └───────────────┬────────────────┘
                                │
                ┌───────────────▼─────────────────────────┐
                │ ProducerInventory.process_inspection()   │
                │ per-producer stock + action suggestion   │
                └──────────────────────────────────────────┘
```

Training uses labelled examples: the vision model predicts **class** and **three
quality scores**; those scores are clipped to 0–100 at inference
(`src/quality/infer.py`). The A/B/C grade always comes from the fixed rules in
`assign_grade(...)` so outcomes stay easy to audit.

## Quick start (quality pipeline)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the quality model
python -m src.quality.train \
  --train_csv data/quality_train.csv \
  --val_csv data/quality_val.csv \
  --image_root data/images \
  --epochs 10 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_dir models/quality

# 3. Run one-image inference
python -m src.quality.infer \
  --checkpoint models/quality/best_quality_model.pt \
  --image data/images/example.jpg

# 4. Run tests (includes quality tests)
python -m pytest tests/ -v
```

## Training data format

Each training and validation CSV row should include:

* `image_path` — path to the image, relative to `--image_root`
* `class_idx` — integer class label
* `color_score` — 0 to 100
* `size_score` — 0 to 100
* `ripeness_score` — 0 to 100

After training, the best model weights are saved as
`models/quality/best_quality_model.pt`.

## Integration notes

For wider platform integration:

* Each producer is keyed by `producer_id`, so many producers can use the same
  deployment without mixing stock.
* `process_inspection(...)` returns a plain record you can log to a file,
  database, or event stream.
* Suggested actions (`sell_normal`, `discount_10_20`,
  `clearance_or_remove`, `manual_review`) can drive UI labels, shop rules, or
  manual review queues.
* Grade cut-offs live in `QualityThresholds`; you can change them without
  retraining the image model.

## Limitations and known issues (quality)

* The CSV must already include the three quality scores; building those scores
  from raw photos (or labels) is a separate step.
* `class_idx` is only a number; a separate list of which number means which 
  produce type should be kept.
* Tests cover grading and inventory only; model metrics (e.g. accuracy, 
  score error) should be added.
