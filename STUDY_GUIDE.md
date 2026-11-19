# Study guide: what each project is for and how to come back to it

Written for revisiting the work later, not for a quick read. For each of the 30 projects you get: the reference article's approach, what my notebook did differently, the main lesson and the limits, then two prompts: **what to understand first** and **an exercise to write your own code before looking at mine**. Read the notebook's own summary for the exact numbers.

**How to use it.** (1) Read the project's section. (2) Try the exercise on your own, with the data from the notebook (`_data/` is filled on first run). (3) Open the notebook and compare: where do your numbers and mine differ, and why? (4) Run the app page to see the saved model.

**Honest notes about the comparison with the references.** The references are short tutorials; most print one score or one table. Where their data could be rerun I did; where it could not (for example project 01: the data file is no longer online) the notebook says so and compares with the numbers the article printed. In several projects my result differs from the tutorial's headline conclusion (for example 04, 14, 19, 25); those are the ones to study first.


---

## 01 - Car Insurance Modelling
**Topic:** Regression  |  [reference article](https://amanxai.com/2024/01/08/car-insurance-modelling-using-python/)  |  folder `01 Car Insurance Modelling - Regression/`

**Reference approach.** a Poisson GLM (statsmodels) on a small 1,000-row file with 3 predictors (it looks synthetic; I could not rerun it because its data file is no longer online), no exposure, no hold-out score.

**What my notebook does.** real data (678,013 French motor policies, `freMTPL2freq`), exposure handled properly, four models (constant, two Poisson GLMs, boosting) scored on a hold-out set with the Poisson deviance, and a check of whether the ranking is useful (lift by decile).

**Main lessons.** * The real portfolio shows the same direction as the reference for driver age (younger drivers claim more) but also a **non-linear age effect, a new-car spike, a strong bonus-malus effect and an urban effect** that the reference's small file does not show.
* **Model ranking (test deviance, 135,603 policies):** constant 0.6274, plain GLM 0.6078 (3.1% better), richer GLM 0.5976 (4.8%), boosting 0.5783 (7.8%). About a third of the plain GLM's gap to boosting came from the GLM's weak specification; the boosting advantage over the richer GLM is 0.0193 (95% interval 0.0170 to 0.0214, paired bootstrap on one split). The GLM is easier to explain to a regulator; choose by the use case.
* The reference could not be rerun on its data (the file is no longer online); I compared with the coefficients it printed. The reference has no exposure, hold-out score or over-dispersion check.
* **Limits:** one random split; capping rules (claims at 4, exposure at 1 year) are assumptions; no claim *severity* model (a price needs frequency x average claim cost); features are policy attributes only. **Age and region are regulated or sensitive rating factors in many jurisdictions**; a real pricing model needs a legal and fairness review. Next step: a Gamma severity model on `freMTPL2sev` and a combined pure-premium estimate.

**Understand first.** Why a count model with exposure, not least squares? Why was a hand-built GLM only a third of the way to boosting?

**Exercise (before reading my code).** Write your own Poisson GLM with an offset on `freMTPL2freq`, then add the shapes (splines/bins) one at a time and watch the deviance.


---

## 02 - Price Elasticity of Demand Analysis
**Topic:** Regression  |  [reference article](https://amanxai.com/2024/08/19/price-elasticity-of-demand-analysis-with-python/)  |  folder `02 Price Elasticity of Demand - Regression/`

**Reference approach.** compute `pct_change()` of price and of quantity row by row and divide. **My method:** first check how the file is built, then estimate one elasticity with a log-log regression that compares each store-item *with itself* over time, and test whether a model that uses price predicts next week's sales better than one that ignores it.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Unit of analysis | 100,000 raw rows (about 10 copies each) | 10,000 distinct store-item-weeks |
| Estimate | mean of row-wise %dQ/%dP ratios (about -0.35 reported; -0.46 / -0.33 when I re-ran it) | log-log regression with store-item and week effects: **+0.008 (95% CI -0.035 to +0.051)** |
| Uncertainty | none | clustered standard errors, subgroup intervals |
| Validation | none | forward-in-time hold-out week, with a no-price ablation |
| Conclusion | "demand is slightly inelastic" | no detectable price effect in this file; revenue what-if is not decision-grade |

**Main lessons.** 1. Check how a file is built before computing anything: the biggest result here (the duplicated rows and inconsistent money columns) came from `groupby`/`nunique`, not from a model.
2. A ratio of two noisy percentage changes is a poor estimator; a regression with a stated interval is much more honest.
3. "Not significant" is a finding: it says the data cannot support a pricing decision, not that elasticity is exactly zero.

**Limits.** Five weeks per item; competitor price is an average of inconsistent copies; no cost data; a synthetic-looking file. **Next step:** a dataset with real promotions or price experiments, where the price changes are large and not chosen by the sellers.

**Understand first.** The reference file had duplicated rows; how did `groupby`/`nunique` reveal it? Why is a ratio of two noisy % changes a poor elasticity?

**Exercise (before reading my code).** Estimate the elasticity yourself with a log-log regression and read the interval before looking at the notebook's.


---

## 03 - Bike Sharing Demand
**Topic:** Regression  |  not from the lists  |  folder `03 Bike Sharing Demand - Regression/`

**Reference approach.** see the notebook's first cell.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| Step | Result |
|---|---|
| Honest evaluation | time-ordered split: train to Sep 2012, test Oct-Dec 2012 |
| Floor | hour x working-day average: RMSLE 0.60, MAE 80 |
| Best model | gradient boosting (Poisson loss): RMSLE 0.43, MAE 45 bikes/hour |
| Main weakness | under-predicts the growing demand (+16 bikes/hour bias), worst at commute peaks |
| Uncertainty | 80% band covers about 73% of test hours after calibration (raw: 52%) |

**Main lessons.** (1) Plot the target by hour and day type before modelling: the whole problem is in those two curves. (2) A forward-in-time split exposes a growth bias a random split would hide. (3) Permutation importance can mislead when a feature barely varies in the evaluation set: confirm with an ablation. (4) Prediction intervals need an explicit coverage check on future data.

**Limits.** Two years of one city; no events/holiday detail beyond a flag; no lag features (yesterday's rentals) because a planner forecasting ahead would not have them. **Next step:** add an explicit trend term or refit weekly, and a separate casual/registered model.

**Understand first.** Why does a forward-in-time split give worse (more honest) scores than a random one on demand data?

**Exercise (before reading my code).** Plot demand by hour for working vs non-working days first, then predict it with only those two features and see how far you get.


---

## 04 - Loan Approval Prediction
**Topic:** Classification  |  [reference article](https://amanxai.com/2023/05/15/loan-approval-prediction-using-python/)  |  folder `04 Loan Approval Prediction - Classification/`

**Reference approach.** fill missing values, one random 80/20 split, `StandardScaler` + random forest, report one accuracy.

**What my notebook does.** data sets this small (491 labelled rows) make a single split's accuracy very noisy, so I (1) put preparation inside a pipeline so it cannot leak, (2) score with repeated cross-validation and show the spread, (3) compare against two trivial baselines, (4) check calibration and the cost of a wrong approval, and (5) look at how decisions differ between applicant groups.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Evaluation | one random 80/20 split | 5-fold CV x 10 repeats; plus 200 single-split seeds to show the instability |
| Baselines | none | approve-everyone (69.9%); credit-history rule (81.3%) |
| Result | random forest accuracy (one number) | **all models about 81%; the one-line rule matches them**; models add AUC (0.70 -> 0.77) |
| Decision | accuracy | cost-aware threshold (assumed 3:1), calibration, confusion matrix |
| Responsible use | not covered | group approval rates, model vs history |

**Main lessons.** (1) Always benchmark against the simplest rule an analyst would write: here it equals the ML models in accuracy. (2) On a few hundred rows a single split can mislead by +/-7 points. (3) Picking a threshold needs a stated cost assumption, and the result is often flat. (4) Check group differences before deploying anything that decides about people.

**Limits.** 491 labelled rows (reference 614); no repayment outcomes, only past approvals; the 3:1 cost is assumed; one lender's decisions. **Next step:** repayment data to model default risk instead of approval, and a proper fairness audit with legal input.

**Understand first.** Why does a one-line analyst rule match the ML models? When does an ML model earn its place?

**Exercise (before reading my code).** Write the simplest rule you can for loan approval, and check it against the notebook's models with repeated CV.


---

## 05 - Credit Scoring and Risk Tiers
**Topic:** Classification  |  [reference article](https://amanxai.com/2023/07/03/credit-scoring-and-segmentation-using-python/)  |  folder `05 Credit Scoring and Risk Tiers - Classification/`

**Reference approach.** a 1,000-row synthetic table with **no repayment outcome**; the score is a hand-weighted formula over a few columns, then k-means groups the scores. Nothing in it can tell whether the score predicts defaults.

**What my notebook does.** use the real German Credit data (1,000 loans, **with** a good/bad outcome), build a classic **scorecard** (weight-of-evidence + logistic regression, points scaled like a bureau score), validate it with the industry metrics (AUC, Gini, KS), compare with a more flexible model, and check that the tiers actually separate bad rates.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | 1,000 synthetic rows, no outcome | German Credit: 1,000 real loans, 30% bad |
| Score | hand-weighted formula | WoE + logistic regression, scaled to points (600 = 30:1 odds, +30 points = double the odds) |
| Segments | k-means on the score | risk tiers by score quintile |
| Validation | none possible (no outcome) | repeated CV: AUC 0.79, Gini 0.57, KS 0.47; bad rate 63.5% -> 5.0% across tiers |
| Interpretability | hidden in the formula | points per group, information value per variable |

**Main lessons.** (1) A score without outcome data can be formatted nicely but cannot be validated; with outcomes, tier monotonicity and KS/Gini are the checks. (2) A transparent scorecard matched gradient boosting here. (3) WoE tables must be learned inside each fold, or the validation is flattered. (4) Always read the points: counter-intuitive groups are a warning about data quirks or small samples.

**Limits.** 1,000 loans from 1994 Germany; some categories have very few cases; WoE bins are simple quantile bins; the tier cut-offs are illustrative. **Next step:** monotonic binning, reject-inference for applicants who were never approved, and removal of protected attributes with a re-validation.

**Understand first.** What does weight-of-evidence do, and why must it be learned inside each CV fold?

**Exercise (before reading my code).** Build a scorecard for a different binary outcome using WoE + logistic regression and scale it to points.


---

## 06 - Customer Churn: Comparing Many Models
**Topic:** Classification  |  [reference article](https://amanxai.com/2024/02/19/compare-multiple-machine-learning-models/)  |  folder `06 Customer Churn Model Comparison - Classification/`

**Reference approach.** "Compare Multiple Machine Learning Models" (amanxai.com, Feb 2024)

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference article | This notebook |
|---|---|---|
| Models | a few regressors on 414 rows | 14 models (incl. baseline) on 7,043 customers |
| Evaluation | one test split | 5-fold x 3 repeats, mean +/- sd, PR-AUC as well as AUC |
| Comparison | rank by one score | corrected paired t-test: 7 of 13 models tied with the best |
| Cost | mentioned | fit time measured; logistic regression 0.05 s |
| Outcome | "pick the best" | pick the simplest model in the top group; translate ranking into campaign gain |

**Main lessons.** (1) Differences smaller than the fold-to-fold spread are not rankings. (2) Include a trivial baseline and a logistic regression in every comparison; they are often in the top group. (3) A model comparison is only a starting point: tuning, calibration and the business value of the ranking decide what to deploy.

**Limits.** No hyper-parameter tuning; one telecom data set; the campaign economics are assumed; churn here is a single snapshot label without a time dimension. **Next step:** tune the top models in a nested CV and add survival analysis (time to churn).

**Understand first.** What does 'tied with the best' mean when you compare 14 models? Why use a corrected paired t-test?

**Exercise (before reading my code).** Run the same comparison on another classification set and see if the top group is again full of simple models.


---

## 07 - Geospatial Clustering (earthquake hotspots)
**Topic:** Clustering  |  [reference article](https://amanxai.com/2025/04/22/geospatial-clustering-with-python/)  |  folder `07 Geospatial Clustering - Earthquake Hotspots - Clustering/`

**Reference approach.** k-means on food-delivery coordinates (Kaggle data, not downloadable here) to create delivery zones, plus a `geodesic` distance calculation.

**What my notebook does.** use real, public geographic points (USGS, 17,000 M5+ earthquakes, 2015-2024) whose shape is *not* round, so the choice of algorithm matters. I compare **k-means**, **DBSCAN** and **HDBSCAN** on the sphere (not on raw lat/lon degrees), check them with a quantitative score, test stability, and turn the best one into a lookup ("which hotspot is this point in, and how far is it?").

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | delivery coordinates (Kaggle) | 16,941 USGS earthquakes M5+, 2015-2024 |
| Geometry | lat/lon with k-means; geodesic distance for one feature | unit-sphere vectors, so distances and the date line are handled correctly |
| Algorithms | k-means | k-means, DBSCAN, HDBSCAN with settings sensitivity |
| Checks | visual map | silhouette, region purity, subsample stability (ARI), noise share |
| Output | zones | hotspot table + a point-to-hotspot lookup with distance |

**Main lessons.** (1) k-means is right when you need a *partition* (delivery zones, where every address needs an owner) and wrong when the real structure is thin belts and noise. (2) DBSCAN's two parameters change the answer a lot; always show a sensitivity grid. (3) Scores like silhouette favour some algorithms by construction; use several checks. (4) "Not in a cluster" must be read as "not dense in this data", never as "safe".

**Limits.** Ten years only, magnitude 5+ only; place labels are crude ground truth; no time dimension (aftershock sequences are clustered in time as well); the 100 km / 25 setting is a judgement. **Next step:** space-time clustering (ST-DBSCAN) to separate aftershock swarms from long-lived zones.

**Understand first.** When is k-means (a partition) right, and when is DBSCAN (belts and noise)? What does 'noise' mean?

**Exercise (before reading my code).** Cluster the same earthquakes with a different eps/min_samples and describe what changes.


---

## 08 - User Profiling & Segmentation
**Topic:** Clustering  |  [reference article](https://amanxai.com/2024/02/26/user-profiling-and-segmentation-using-python/)  |  folder `08 User Profiling and Segmentation - Clustering/`

**Reference approach.** pick some columns, scale/one-hot them, run k-means with a chosen k, describe each cluster.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Method | k-means on chosen columns, describe clusters | first test whether structure exists, then cluster, then test the clusters |
| Dependence between columns | not checked | largest correlation 0.08; 0 of 70 tests survive correction |
| Clusterability | not checked | Hopkins 0.50; best silhouette 0.052 = shuffled-data range |
| Stability | not checked | bootstrap ARI 0.25 (unstable) |
| Segment value | "personalised campaigns" | segments' effect sizes are inside the null range; no rule predicts CTR/conversion |

**Main lessons.** (1) k-means always returns k clusters: a result is not evidence of structure. (2) Shuffle each column independently to create a *null data set*, and see whether your score beats it. (3) Check stability by resampling before naming segments. (4) Test any proposed segment on an outcome it was *not* built from, with multiple-testing correction.

**Limits.** The file appears to be randomly generated (inferred from the flat distributions and zero dependence), so the negative result is about this file; real user data usually has structure. The tests used are standard but not exhaustive (for example, non-linear dependence between numeric columns was not examined beyond what k-means itself would find). **Next step:** repeat the same pipeline on real behavioural data, where I would expect the null comparison to show a clear gap.

**Understand first.** What is a null data set (shuffled columns) and why does a silhouette need one?

**Exercise (before reading my code).** Take any customer table, run k-means, then shuffle each column independently and rerun: compare the scores.


---

## 09 - RFM Analysis
**Topic:** Clustering  |  [reference article](https://amanxai.com/2025/08/26/a-guide-to-rfm-analysis-using-python/)  |  folder `09 RFM Analysis - Clustering/`

**Reference approach.** compute R, F, M per customer, score each 1-4 by quartiles, concatenate the scores into segments.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Scoring | quartile cut-offs on R, F, M | rank-based quintiles for R and M, business bins for F (cut-offs inspected first) |
| Data check | not mentioned | 94.6% of customers ordered once; F quartiles all equal 1 -> reference scoring gives 94.6% score 1 and nobody 2-3 |
| Segments | RFM score codes | 6 named rule segments with customers, revenue share and action |
| Discovery | none | k-means vs shuffled null: not better than noise |
| Validation | none | back-test: 5.5% return; no reliable predictive value (28 events) |

**Main lessons.** (1) Print the percentile cut-offs before using quantile scores: discrete variables break them. (2) RFM segments are rules, useful as a communication tool, but their value must be shown by a forward-in-time test. (3) Revenue concentration should be measured, not assumed; here it is far below 80/20. (4) A nice-looking silhouette needs a null comparison.

**Limits.** 61 days is too short to measure recency or loyalty properly; 28 repeat events give little statistical power; the file looks generated; monetary value is revenue, not margin. **Next step:** a year or more of real transactions, a customer-lifetime-value model (BG/NBD + Gamma-Gamma), and the same back-test.

**Understand first.** Why do quantile RFM scores fail with discrete variables? How concentrated is revenue really?

**Exercise (before reading my code).** Build RFM scores by hand with `qcut`, print the cut-offs, then test the segments forward in time.


---

## 10 - Handwriting Recognition
**Topic:** Computer Vision  |  [reference article](https://amanxai.com/2020/10/10/handwriting-recognition-with-python/)  |  folder `10 Handwriting Recognition - Computer Vision/`

**Reference approach.** k-nearest-neighbours on 32x32 black-and-white images flattened into 1,024-long vectors (a small teaching data set that is not publicly downloadable).

**What my notebook does.** use the standard **MNIST** data (60,000 + 10,000 images), reproduce the reference idea (k-NN on raw pixels), add a logistic regression and a small **CNN** (PyTorch, CPU), study the **mistakes**, check **confidence calibration**, and then test **robustness**: rotated, shifted, noisy, thicker/thinner and inverted digits, with and without training on augmented data. A model that scores 99% on MNIST but fails on slightly different writing is not a handwriting reader.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | 32x32 binary text images (thousands) | MNIST: 55,000 train / 5,000 validation / 10,000 test |
| Model | k-NN on flattened pixels | k-NN (same idea), logistic regression, CNN, CNN + augmentation |
| Test accuracy | not reported as one comparable number | k-NN 96.99%, logistic 92.49%, **CNN 99.09%**, CNN+aug 98.96% |
| Confidence | none | CNN calibrated (ECE 0.0025); 93% coverage at 99.95% accuracy |
| Robustness | not tested | rotation, shift, scale, noise, stroke width, inversion: big drops outside the training conditions |

**Main lessons.** (1) The k-NN idea works (97%) but a CNN makes about 3 times fewer errors with one fast epoch loop. (2) A near-perfect benchmark score says little about messy input: shift by 4 pixels cost the plain CNN 30 points. (3) Augmentation fixes what it simulates and can hurt elsewhere. (4) A calibrated confidence enables a "refer to a human" rule.

**Limits.** MNIST digits are centred, size-normalised and written by US census workers and students; photos, other scripts or other pens will differ. Only 5 epochs, one seed per model (so tiny differences like 99.09 vs 98.96 are not meaningful). **Next step:** train with milder, mixed augmentations plus noise and stroke jitter, and evaluate on digits drawn by real users.

**Understand first.** Why does a 4-pixel shift cost a plain CNN 30 points? What does augmentation fix and what does it not?

**Exercise (before reading my code).** Train a small CNN on MNIST yourself and test it on shifted digits before adding augmentation.


---

## 11 - Sign Language Classification
**Topic:** Computer Vision  |  [reference article](https://amanxai.com/2020/12/05/sign-language-classification-with-python/)  |  folder `11 Sign Language Classification - Computer Vision/`

**Reference approach.** Kaggle "Sign Language MNIST" (28x28 grey-scale pixels, a separate test file), a small Keras network and image augmentation, one accuracy report. (That Kaggle file needs an account; it is not on the public Hugging Face hub in a loadable form, so I use a different public data set below.)

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | Sign-MNIST (28 x 28 grey, separate signers in the test file) | 10,873 real colour photos, 26 letters, varied backgrounds |
| Model | small Keras network + augmentation | ResNet-18 features + logistic regression / k-NN, and a small CNN from scratch |
| Score | one accuracy | random split 75.3% vs **group split 58.3%** (range 45-75% by grouping rule) |
| Leakage | not examined | near-copies found: 34% of images have a neighbour with similarity >= 0.85; the gap is about 17 points |
| Errors | classification report | per-letter F1, confusions (fist letters, G/H, P/Q) |
| Robustness | not examined | mirror harmless; rotation, blur, lighting and cropping drop accuracy to 20-45% |

**Main lessons.** (1) A classifier evaluated with a random split on photo data can report a score 17 points too high when near-identical shots exist. (2) Always look at the nearest neighbours of your images before you trust a number. (3) Pretrained features and a simple classifier beat a small CNN from scratch on a few thousand varied photos. (4) Check the embedding space before using it: raw positive features make every pair look similar.

**Limits.** The grouping rule is my own and the true "same photo" boundary is unknown (hence the range); one fold for the model comparison; the CNN was still improving; no hand detection or cropping; J and Z are static here. **Next step:** hand cropping with a detector (MediaPipe) before classification and a person-level split if signer IDs were available.

**Understand first.** Why did a random split give 75% and a group split 58%? What are near-duplicates?

**Exercise (before reading my code).** Split a photo set by person/session and compare with a random split on your own model.


---

## 12 - Flower Recognition
**Topic:** Computer Vision  |  [reference article](https://amanxai.com/2020/11/24/flower-recognition-with-python/)  |  folder `12 Colour Recognition - Computer Vision/`

**Reference approach.** a CSV of ~1,000 colour names with RGB values; for a clicked pixel, loop over every row and return the name with the smallest **sum of absolute RGB differences** (an OpenCV double-click program).

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Naming | nearest name by sum of absolute RGB differences, Python loop | four distance measures, vectorised / k-d tree |
| Data | CSV of about 1,000 colour names (origin undocumented) | xkcd colour survey (949 names, shipped with matplotlib) |
| Validation | by eye on one picture | leave-one-out family agreement (86-89%), independent CSS test (71-76%), classifier (86-88% CV) |
| Images | clicked pixels | dominant colours by k-means in CIELAB, with name and family |
| Robustness | not tested | noise and compression harmless; lighting changes cut family accuracy to 67-83% |
| Speed | about 138 ms per pixel | 0.09 s for 40,000 pixels |

**Main lessons.** (1) A perceptual colour space is the right default but, measured honestly on 570 and 78 colours, the gain over RGB is within the uncertainty (2-4 points). (2) The hard part of colour naming is lighting, not distance. (3) A "ground truth" built from name text (last word) is itself noisy; imbalance (teal: 12 names) decides some of the errors. (4) Vectorisation matters more than the choice of metric for speed.

**Limits.** English names from one survey; no skin-tone or textile-specific calibration; no white-balance step; flat-patch stress tests only. **Next step:** add a grey-world white-balance step and re-test the brightness conditions.

**Understand first.** Why is lighting, not distance, the hard part of colour naming? How noisy is a label built from the last word of a name?

**Exercise (before reading my code).** Name colours with k-NN in RGB and in a perceptual space and compare with intervals.


---

## 13 - Text Classification Pipeline with Hugging Face Transformers
**Topic:** NLP  |  [reference article](https://amanxai.com/2025/07/22/text-classification-pipeline-with-hugging-face-transformers/)  |  folder `13 Text Classification Pipeline - NLP/`

**Reference approach.** the 20 Newsgroups data (headers, footers and quotes removed), a **random 80/20 split** of all 18,846 posts, DistilBERT fine-tuned with the Hugging Face `Trainer`.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Split | random 80/20 over all 18,846 posts | official date-based split (7,532 later posts as the test set) |
| Evaluation set-up | one | four set-ups: 70.3% (honest) to 93.3% (random split with headers kept) for the same model |
| Models | DistilBERT fine-tuned (full data, 512 tokens) | TF-IDF LR 70.3%, TF-IDF SVM 70.6%, MiniLM + LR 65.4%, **DistilBERT on a CPU budget 66.7%** (6,000 posts, 128 tokens) |
| Fair comparison | none | TF-IDF trained on the same 6,000 posts: 67.6%; majority class 5.3% |
| Errors | not examined | overlapping "misc"/religion/politics groups; quirk words learned as topic evidence |

**Main lessons.** (1) The way a text set is split and cleaned changes the headline score (70% to 93% for one model) by more than the choice of model does; always state the set-up. (2) A fine-tuned transformer is not automatically better than TF-IDF + a linear model: with 6,000 examples, 128 tokens and 2 CPU hours it was not. (3) Always include the simple baseline trained on the *same* data. (4) Look at the largest weights of a linear model: they reveal shortcuts.

**Limits.** My DistilBERT run is a reduced budget (about 53% of the training posts, 128 tokens, 2 epochs, one seed, no tuning), so it says nothing about the full-data, 512-token transformer used by the reference. No confidence intervals were computed for the model differences. 20 Newsgroups is old, informal English from 1993; the date-based split is stricter than a random one, but the same authors and threads can still appear in both parts. **Next step:** run the full-data/512-token fine-tune on a GPU, add a paired test between the best TF-IDF and the transformer, and check with a split by author or thread.

**Understand first.** Why does the same TF-IDF model score 70% to 93% depending on set-up? Why did DistilBERT not win here?

**Exercise (before reading my code).** Fine-tune a transformer yourself on the full data with a GPU (Colab) and compare with the 70.3% TF-IDF number.


---

## 14 - Text Summarization
**Topic:** NLP  |  [reference article](https://amanxai.com/2024/10/07/text-summarization-model-using-llms/)  |  folder `14 Text Summarization - NLP/`

**Reference approach.** load `t5-small`, summarise **one** hand-picked text, look at the output. No score, no comparison.

**What my notebook does.** evaluate on **120 real CNN/DailyMail test articles** with their human-written highlights, using ROUGE (word-overlap with the reference summary), against four systems: **Lead-3** (just the first three sentences), a **supervised extractive model I train myself** (gradient boosting that picks sentences), **t5-small** (the reference's model) and **DistilBART-CNN** (a model fine-tuned for this exact data). I add two checks ROUGE cannot give: how much of each summary is **copied vs newly written**, and whether any **number in a summary is missing from the article** (a simple test for invented facts). Bootstrap confidence intervals show which differences are real.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Evaluation | one hand-picked text, read by eye | 120 CNN/DailyMail test articles, ROUGE-1/2/L with bootstrap intervals |
| Baselines | none | Lead-3 and a supervised extractive model I trained (110,732 sentences) |
| Models | t5-small | t5-small, DistilBART-CNN-6-6 |
| Result | a summary | DistilBART 43.0 > Lead-3 39.1 = my extractor 39.3 > t5-small 36.5 (ROUGE-1) |
| Extra checks | none | copying vs writing (10-11% new word pairs for neural models), unsupported-number alarm (0-1.7%) |

**Main lessons.** (1) Without a baseline a summary score means nothing: the reference's t5-small does *worse* than copying the first three sentences. (2) A model trained for the task (DistilBART on CNN/DM) is clearly better than a general model. (3) A learned sentence picker did not beat a position rule: check that complexity buys something. (4) ROUGE measures overlap, not truth; add checks such as the number alarm.

**Limits.** 120 articles (intervals about +/-2 points); only the first 512 tokens are read by the neural models; ROUGE-Lsum splitting uses my own sentence splitter; no human judgement of fluency or accuracy; the number check is crude and the 3 flagged cases were not verified. **Next step:** a larger sample, human ratings on a subset, and a faithfulness metric based on entailment.

**Understand first.** Why does t5-small lose to 'first three sentences'? What does ROUGE measure and not measure?

**Exercise (before reading my code).** Write the Lead-3 baseline in three lines and score it with ROUGE before using any model.


---

## 15 - Document Analysis
**Topic:** NLP  |  [reference article](https://amanxai.com/2024/10/21/document-analysis-using-llms-with-python/)  |  folder `15 Document Analysis - NLP/`

**Reference approach.** extract the text of a long PDF (Google's Terms of Service), summarise it with `t5-small` (which reads only the first 512 tokens), and look at the output. No questions, no scores.

**What my notebook does.** (1) split the document into sections and passages; (2) measure how much of the document a "first 512 tokens" summary can possibly cover; (3) build **retrieval + extractive question answering** (a search step picks passages, a SQuAD-trained DistilBERT reader extracts the answer) and **score it** on 14 questions whose gold answers I verified word-for-word in the text, plus 4 questions the document does not answer; (4) analyse the document itself: readability by section, and who the obligations fall on ("you must..." vs "we may...").

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Document | Google Terms of Service (PDF) | GitHub Terms of Service (public markdown, about 7,500 words) |
| Summary | t5-small on the first 512 tokens | same, plus per-section summaries: coverage of 14 key facts **0 vs 8** |
| Questions | not covered | retrieval + extractive QA scored on 14 verified questions + 4 unanswerable |
| Retrieval | n/a | MiniLM finds the right passage in the top 3 for 14 of 14 questions |
| Answers | n/a | best selection strategy: right passage 93%, F1 0.60, about 12 of 14 substantively correct (my reading) |
| Failure mode found | not tested | reader returns an answer to all unanswerable questions; confidence cannot reject them |

**Main lessons.** (1) A "summary" built from the first 512 tokens is a summary of the preamble: always compute how much of the document the model sees. (2) Retrieval quality was good; the weak point was how to choose between candidate answers, and a simple combination of retrieval and reader scores helped. (3) Evaluate with questions whose answers you have verified in the source: the code asserts that each gold answer is present word-for-word. (4) Extractive readers do not abstain; design for that.

**Limits.** One document; 14 + 4 questions written by me (so the gold set is small and shares my view of the text); exact-match and F1 are rough proxies; the "answers are right" judgement is mine alone; no legal advice is implied. **Next step:** a reader that supports "no answer", a larger question set from several people, and a comparison with an LLM that must quote its source passage.

**Understand first.** Why does summarising the first 512 tokens summarise the preamble? How was answer selection the weak point?

**Exercise (before reading my code).** Ask your own five questions about a document and check the retrieved passage by hand.


---

## 16 - Music Recommendation System
**Topic:** Recommendation Systems  |  [reference article](https://amanxai.com/2023/07/31/music-recommendation-system-using-python/)  |  folder `16 Music Recommendation System - Recommendation Systems/`

**Reference approach.** get a Spotify developer token, download the audio features of some tracks through the API, and recommend by similarity of those numbers. (Spotify removed access to audio-feature data for new apps in November 2024 and the method needs personal credentials, so I use a public snapshot of **114,000 Spotify tracks with the same audio features** from the Hugging Face hub.)

**What my notebook does.** clean the data (the same song appears many times), compare **ways of measuring "similar"** (raw numbers, standardised numbers, and a **learned embedding** from a small neural network trained to tell genres apart), and evaluate with measurable proxies on held-out songs: do the neighbours share a genre, how varied are the lists, and are a few songs recommended over and over ("hubs")?

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | Spotify API (needs credentials; audio-feature access closed to new apps in Nov 2024) | public snapshot: 114,000 rows -> 81,195 unique songs, same audio features |
| Cleaning | none | duplicates merged (title + artist), genres kept as a set |
| Similarity | cosine on the API numbers | raw vs standardised vs PCA-whitened vs learned embedding |
| Evaluation | by listening | genre precision@10 on held-out songs: **raw 5.3% < whitened 15.0% < standardised 15.4% < learned 16.6%** (random 1.2%), coverage, hubness, popularity drift, listening check |

**Main lessons.** (1) Putting features on a common scale mattered more than any model: raw numbers gave 5% vs 15% after standardising. (2) Duplicated songs would have made every score look better; they must be merged before evaluating. (3) Without user data we can only measure proxies; genre agreement is useful but is not "I would enjoy this". (4) Audio descriptors alone are weak for mainstream music but fine for distinctive acoustic styles (classical).

**Limits.** One random split of songs; genre is a crude proxy for taste; one embedding architecture, one seed; no listener data, lyrics or artist information; the data set has an artificial 1,000-songs-per-genre balance. **Next step:** evaluate with real playlists (does the recommended song co-occur in a playlist with the seed?) and blend with collaborative filtering.

**Understand first.** Why did scaling matter more than the model for song similarity? Why merge duplicated songs first?

**Exercise (before reading my code).** Compute cosine similarity on raw vs standardised features for a song you know and compare the neighbours.


---

## 17 - Hybrid Recommendation System
**Topic:** Recommendation Systems  |  [reference article](https://amanxai.com/2023/06/05/hybrid-recommendation-system-using-python/)  |  folder `17 Hybrid Recommendation System - Recommendation Systems/`

**Reference approach.** a fashion-products table; content-based filtering with TF-IDF text similarity, collaborative filtering with `surprise` SVD to predict ratings, and a weighted blend; no hold-out evaluation. (The `surprise` package does not install on this machine's Python version, and the fashion file's ratings cannot be validated, so I use the standard **MovieLens 100K** data: 100,000 real ratings by 943 users on 1,682 films.)

**What my notebook does.** a proper **time-based hold-out** (every user's last 20% of ratings is hidden), **ranking metrics** (Recall@10, NDCG@10, hit rate), five recommenders (popularity, content, item-kNN, SVD, hybrid), the **blend weights tuned on a separate validation slice**, plus coverage, novelty, diversity, **cold-start items** and **new users**, and a check that a lower rating-prediction error (RMSE) is not the same as a better top-10 list.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | fashion products table (ratings not verifiable) | MovieLens 100K: 100,000 real ratings, 943 users, 1,682 movies |
| Models | content (TF-IDF text) + `surprise` SVD, blended | popularity, content, item-kNN, SVD, hybrid with weights tuned on a validation slice |
| Evaluation | none | time-based hold-out, NDCG@10 / recall@10 / hit rate, paired bootstrap, coverage, novelty, diversity |
| Result | a blend | hybrid 0.189 > SVD 0.175 = item-kNN 0.174 > popularity 0.104 (NDCG@10; hybrid gain clear but small) |
| Stress tests | none | cold movies (hybrid best), new users (popularity best), rating RMSE vs ranking quality (RMSE-optimal model ranks poorly) |

**Main lessons.** (1) Evaluate with a time-based split and ranking metrics, not rating error. (2) A blend of two strong models helped a little, with a clear but small interval; weights must be tuned away from the test set. (3) The best method depends on the situation: popularity wins for new users, collaborative methods for cold items. (4) Coverage, novelty and diversity move in different directions from accuracy.

**Limits.** One 20% time cut; explicit ratings turned into binary "liked" signals; genres only as content; 907 evaluated users (intervals shown are for user sampling only); no online test. **Next step:** richer content (plot text embeddings), implicit-feedback ALS, and a rule that switches to popularity for users with fewer than about 10 ratings.

**Understand first.** Why use ranking metrics and a time-based split, not RMSE? When does popularity win?

**Exercise (before reading my code).** Implement popularity and item-kNN recommenders and compute NDCG@10 on a time split.


---

## 18 - Market Basket Analysis
**Topic:** Recommendation Systems  |  [reference article](https://amanxai.com/2023/10/16/market-basket-analysis-using-python/)  |  folder `18 Market Basket Analysis - Recommendation Systems/`

**Reference approach.** a small basket file, Apriori to find frequent itemsets, association rules ranked by support / confidence / lift, and a reading of the top rules as business insights.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | 250-basket toy file (7 items) | the same file, plus real groceries: 9,835 baskets, 171 items |
| Method | Apriori and rules by support / confidence / lift | + FP-growth check, null test by shuffling, hold-out validation, basket-completion test |
| Is it real? | not tested | **groceries: yes** (1,325 rules with lift >= 1.5 vs 0.4 in shuffled data; top rules keep 88% of their lift on new baskets). **Reference file: no** (lifts 1.11-1.16, intervals include 1, two halves share 0.1 of the top 5 rules) |
| Is it useful? | "business insights" | rules complete a basket no better than popularity (26-27% vs 25.8% hit rate@5) |

**Main lessons.** (1) Rank by lift, not confidence: confidence just rediscovers popular items. (2) Test rules against shuffled data and on held-out baskets before believing them. (3) A statistically real pattern need not be a useful prediction: say what the rule is *for*. (4) Small baskets with few items can give impressive-looking but empty rules.

**Limits.** The groceries file has no dates or customers, so the validation split is random; support threshold 0.5% hides rarer rules; the completion test hides one random item (one of several possible tasks); confidence intervals are over baskets. **Next step:** price and margin data to rank rules by profit, and a sequence-aware (next basket) model with timestamps.

**Understand first.** Why rank rules by lift, not confidence? What does a null test (shuffled baskets) show?

**Exercise (before reading my code).** Run Apriori on the groceries data and check one rule against shuffled baskets.


---

## 19 - Netflix Subscriptions Forecasting
**Topic:** Time Series  |  [reference article](https://amanxai.com/2023/08/07/netflix-subscriptions-forecasting-using-python/)  |  folder `19 Netflix Subscriptions Forecasting - Time Series/`

**Reference approach.** read the ACF/PACF plots of the differenced series, pick ARIMA(1,1,1), fit on all 42 quarters and forecast the next 10 quarters. No test of the forecast.

**What my notebook does.** compare ARIMA with simpler baselines (last value, last value plus average growth, linear trend) and with exponential smoothing, using **rolling-origin back-testing** (fit on the past, forecast the next 1-4 quarters, move forward, repeat); look at errors **before and after the 2022 slowdown**; check whether the **prediction intervals** contain the truth as often as they claim; then forecast 8 quarters with the best method and say what the forecast can and cannot tell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Model choice | ACF/PACF by eye -> ARIMA(1,1,1) | ARIMA(1,1,1), ARIMA with AIC order, ETS, drift variants, linear trend, naive |
| Evaluation | none | rolling-origin back-test, 81 forecasts per model, horizon 1-4, MASE, before/after 2022 |
| Result | a 10-quarter forecast | top five methods within 0.6 M (MAE about 6-6.7 M); naive 14.5 M; error of 9-10 M at 4 quarters |
| Uncertainty | none shown | 80% intervals contain the truth only 43-47% of the time |
| Forecast | ARIMA | 8 quarters, ARIMA +5.0 M/quarter (279 M by 2025Q3) vs ETS 269.5 M, labelled as a scenario |

**Main lessons.** (1) A forecast is only as good as its back-test: the reference's model is no better than "last value plus recent growth". (2) Plot the ACF with its confidence band in mind: a borderline lag-1 spike cannot fix an order. (3) Check interval coverage: nominal 80% intervals covered 43%. (4) Series with regime changes (a pandemic surge, a slowdown) punish methods that extrapolate a fitted trend.

**Limits.** 42 points; a single series; no covariates (price changes, content releases, password-sharing policy); the back-test period contains one regime change; interval checks are based on a few dozen forecasts. **Next step:** add external regressors and a model with a changing trend (a local-linear-trend state-space model), and evaluate on a later data release.

**Understand first.** Why is 'last value plus growth' as good as ARIMA here? Why did 80% intervals cover only about 45%?

**Exercise (before reading my code).** Back-test two forecasting methods with a rolling origin on any short series.


---

## 20 - Multivariate Time Series Forecasting
**Topic:** Time Series  |  [reference article](https://amanxai.com/2024/03/11/multivariate-time-series-forecasting-using-python/)  |  folder `20 Multivariate Time Series Forecasting - Time Series/`

**Reference approach.** four stock prices over about 3 months, differenced to make them stationary, a **Vector AutoRegression (VAR)** fitted and used to forecast; no comparison with anything simpler.

**What my notebook does.** two experiments. **(A) The reference's own setting** (stocks) tested against a "no change" forecast, on its file and on 7 years of daily prices. **(B) A data set where multivariate information should matter:** hourly power-transformer data (ETTh1: 6 electrical load measurements and the oil temperature, 17,420 hours). I forecast the oil temperature 1, 6, 12 and 24 hours ahead with a seasonal-naive baseline, a VAR, and ridge regression / gradient boosting each fitted twice: **using only the oil temperature's own past (univariate)** and **using all 7 series (multivariate)**. The difference between the two fits is the value of the other series.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | 62 days of 4 stock prices | the same file, **plus** 7 years of stock returns **and** 17,420 hours of 7 power-transformer series (ETTh1) |
| Model | VAR on differenced prices | VAR, ridge and gradient boosting, univariate vs multivariate, level vs change target |
| Baseline | none | no-change, last value, same hour yesterday |
| Stocks | forecast shown | VAR order by AIC = 0; no better than "no change" (RMSE 2.97 vs 3.02 for AAPL); next-day returns R2 -2% to -6% vs zero, sign right 48-50% of days |
| Transformer | n/a | other series help only with the right setup: LightGBM predicting change, +6h and +12h (6-9% lower MAE); linear models and level targets get worse |
| Best result | n/a | LightGBM, 7 series, change target: mean MAE 1.10 vs 1.20 for "last value" (about 8% better) |

**Main lessons.** (1) "Multivariate" is not automatically better; test the univariate version of the *same* learner. (2) Where the data have no predictable structure (daily stock returns) a VAR learns nothing (order 0). (3) For series that drift, predict the change, not the level, or tree models fail. (4) Always compare with last-value and seasonal-naive forecasts.

**Limits.** One test period; one seed; no hyper-parameter search for the linear models; ETTh1's training year is not like its test months; no intervals for the small differences. **Next step:** several rolling test periods, a tuned and regularised linear model, and a neural sequence model for completeness.

**Understand first.** Why does 'multivariate' not help automatically? Why predict the change instead of the level?

**Exercise (before reading my code).** Forecast a drifting series with a tree model on levels, then on differences, and compare.


---

## 21 - Stock Market Anomaly Detection
**Topic:** Time Series  |  [reference article](https://amanxai.com/2024/04/08/stock-market-anomaly-detection-using-python/)  |  folder `21 Stock Market Anomaly Detection - Time Series/`

**Reference approach.** download one year of prices for five stocks, compute the **z-score of the adjusted close price and of the volume over the whole period** and call every point with |z| > 2 an anomaly.

**What my notebook does.** (1) show what that rule actually flags; (2) build detectors on the right quantity (**daily returns**, scaled by recent volatility) and compare them with a rolling z-score, a robust (median/MAD) version, an EWMA-volatility z-score and an Isolation Forest; (3) **plant synthetic jumps and volume bursts of known size** in the real series and measure which detector finds them at a fixed false-alarm budget (the only honest "ground truth" available); (4) look at what the best detector flags in 7 years of real data: market-wide days versus single-stock days.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Quantity analysed | z-score of the *price level* and volume over the whole year | daily *returns*, scaled by recent volatility; volume vs its recent history |
| What the rule flags | 0-5% of days, long stretches that follow trends | isolated surprises: 33 stock-days of 8,495 |
| Validation | none | planted jumps (3-8x volatility) and volume bursts; recall@60: EWMA 55% for 3x jumps vs global z 46%, Isolation Forest 36% |
| Threshold | |z| > 2 from a bell curve | empirical: |z| > 3 happens 4-6x more often than a bell curve says; alarm budget 0.39% |
| Real-data check | none | 76% of flags fall on earnings days (3% of days); March 2020 is not flagged by the adaptive detector, but is by the global rule |

**Main lessons.** (1) Choose the quantity before the rule: z-scores of price levels measure trends, not anomalies. (2) With no labels, plant anomalies of known size to compare detectors, and state the recall at a fixed number of alarms. (3) A simple volatility-adjusted score beat the Isolation Forest here. (4) "Anomaly" depends on the reference regime: against the long-run average, March 2020 is the anomaly; against the previous weeks it is not.

**Limits.** Five large US stocks, 7 years; planted anomalies are one kind (a persistent jump; a volume multiplier); earnings dates unverified; a threshold of 4.5 is a choice; no intraday data. **Next step:** a GARCH model for the volatility, a test on a stock with fewer news events, and a labelled set of known events.

**Understand first.** Why do z-scores of price levels measure trends, not anomalies? What does planting anomalies of known size let you do?

**Exercise (before reading my code).** Plant 20 jumps of 3 sigma into a return series and compare two detectors at a fixed number of alarms.


---

## 22 - Netflix Content Strategy Analysis
**Topic:** Data Visualization  |  [reference article](https://amanxai.com/2024/09/30/netflix-content-strategy-analysis-with-python/)  |  folder `22 Netflix Content Strategy Analysis - Data Visualization/`

**Reference approach.** load a table of 24,812 titles with viewing hours, group by content type, language and release month/season, draw bar and pie charts, and read strategy lessons off them.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data audit | none | 67% of rows have no release date (not at random), every repeated title is a pair with different hours (92%), hours stored as Indian-style text |
| Averages | mean hours by group | medians with bootstrap intervals (mean is 6x the median; top 10% of rows = 68% of hours) |
| Release timing | hours by month/season | restricted to 2023 releases, exposure examined: raw and per-day views both misleading; month explains about 4% of variance |
| Seasons | later seasons draw more | rise is real in the table but is survivorship |
| Conclusions | "strategy lessons" | what the table supports (concentration, shows vs movies, Korean titles) vs what it does not (timing strategy, effect of global release) |

**Main lessons.** (1) Audit missing values: *which* rows are missing changes the meaning of a chart. (2) With a heavy-tailed target, report medians and concentration. (3) Time-since-release is a confounder in any "release month" chart. (4) A chart is evidence for a claim only if the comparison is fair; say what the data cannot answer.

**Limits.** No reporting period, no runtime, no genre, no budget, no marketing; two stacked rows per repeated title; 33% date coverage. **Next step:** get the reporting windows and runtimes (hours per runtime hour), then compare launches by days-in-window.

**Understand first.** How does the choice to drop or keep missing rows change a chart's meaning? Why is release month confounded with age?

**Exercise (before reading my code).** Redo one chart with medians and a fair comparison, and say what it cannot answer.


---

## 23 - Analyzing the Impact of Carbon Emissions
**Topic:** Data Visualization  |  [reference article](https://amanxai.com/2024/11/25/carbon-emissions-impact-analysis-with-python/)  |  folder `23 Carbon Emissions Impact Analysis - Data Visualization/`

**Reference approach.** merge a monthly CO2 table with annual country temperatures, draw a correlation heat map (reported r = 0.96), fit trend lines, and simulate "what-if" scenarios with a regression.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | merged two files | found two series stacked in the CO2 file (ppm and year-over-year %); separated them and checked the relationship (r = 1.000) |
| CO2 | trend line | trend, seasonal swing (7 ppm), acceleration (0.86 -> 2.4 ppm/yr; quadratic fit 1.7 ppm error vs 12.2 for a line on 2013-23) |
| Temperature | trend lines | World +0.28 degrees C/decade (CI 0.24-0.32), a change in slope near 1975, 99% of 185 countries warming |
| Link CO2-temperature | correlation 0.96 (levels) | 0.94 for levels, 0.47 after detrending, 0.31 for yearly changes; slope 1.7 degrees C per 100 ppm |
| What-if | regression scenarios | held-out test: CO2 alone RMSE 0.18 vs time trend 0.33; both together 0.63 (unstable) |

**Main lessons.** (1) Check what is *in* a column: two series shared one. (2) A correlation between two trending series says little; compare detrended and differenced versions. (3) The acceleration of CO2 is visible and a model without it fails within a decade. (4) A what-if tool built on two collinear trending series needs a stated caveat.

**Limits.** Annual world means only; no baseline given for temperature; per-country slopes have no intervals and differing missing years; a regression on two trending series cannot establish cause (the physical mechanism is not tested here); the break year is at the edge of the search grid. **Next step:** add emissions data (not just concentration), ocean and volcanic indices, and a physically-based energy-balance model.

**Understand first.** What does the file mix in one column? Why do two trending series correlate by construction?

**Exercise (before reading my code).** Correlate CO2 and temperature in levels, then in first differences, and compare.


---

## 24 - Rainfall Trends in India Analysis
**Topic:** Data Visualization  |  [reference article](https://amanxai.com/2024/11/18/rainfall-trends-in-india-analysis-with-python/)  |  folder `24 Rainfall Trends in India - Data Visualization/`

**Reference approach.** plot annual rainfall and a 10-year moving average, read "no apparent long-term trend" off the picture, flag years beyond +/-1.5 standard deviations as drought / extreme-rain years, and apply an Isolation Forest.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Trend | "no apparent trend" from a picture | tested: -3.0 mm/decade (95% interval -12.1 to +6.1), Mann-Kendall p = 0.39; 30-year windows range -49 to +71 mm/decade |
| Seasons | bar chart of means | monsoon = 75% of rain; no season has a robust trend (one nominal decline in Jan-Feb, p = 0.03/0.08, among 5 tests) |
| Persistence | not examined | lag-2 autocorrelation 0.31; no significant mean shift between thirds (p = 0.23) |
| Extremes | 1.5 sd rule: 5 dry, 7 wet | same, plus 1 sd and percentile rules; counts match a normal distribution (no excess of extremes) |
| Isolation Forest | listed as anomaly detection | equivalent to taking the most extreme values of one column |
| Drivers of dry years | not examined | June-October deficit, 84% in the monsoon months (JJAS) |

**Main lessons.** (1) "No trend visible" and "no trend" are different claims: the interval shows what a century of data can and cannot exclude. (2) Slopes over short windows are unstable for variable series with mild persistence. (3) Extreme-year counts change with the definition; compare them with what the distribution would predict. (4) A multivariate anomaly algorithm on one variable is a threshold.

**Limits.** One all-India series (regional trends could differ); area-weighted annual totals hide the distribution within the year (heavy-rain days, dry spells); no temperature, ENSO or station information; only 115 values per series. **Next step:** regional and daily data, extreme-rain indices and a link with sea-surface-temperature indices.

**Understand first.** What is the difference between 'no trend visible' and 'no trend'? Why do slopes change with the window?

**Exercise (before reading my code).** Fit a trend with an interval over several window lengths and plot slope vs window.


---

## 25 - Build an AI Agent to Master a Game
**Topic:** AI Agents  |  [reference article](https://amanxai.com/2025/09/09/build-an-ai-agent-to-master-a-game-using-python/)  |  folder `25 AI Agent to Master a Game - AI Agents/`

**Reference approach.** a Deep Q-Network (DQN) in PyTorch trained on Gymnasium's CartPole-v1, shown by a learning curve and a rendered episode.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Method | one DQN, one learning curve, a rendered episode | random policy, a one-line rule, tabular Q-learning, DQN; **3 seeds each** |
| Benchmark | none | random 23 steps; **hand-written rule with no learning: 500 in 100/100 episodes** |
| Result | the DQN "masters" the game | DQN 500 on 3/3 seeds, Q-learning 500 / 218 / 486; DQN learns faster (first success at 20-44k steps vs 55-85k) but both are unstable during training |
| Robustness | not tested | 6 modified worlds: all learned policies and the rule at 491-500, so the test did not discriminate |
| Saved | - | DQN weights, Q-table, GIFs of a random and a trained agent |

**Main lessons.** (1) Compare against the simplest thing that could work: a one-line rule is as good as either learner here. (2) Report seeds, not a single run: Q-learning had one bad seed. (3) Performance during training oscillates, so the "final" score depends on when you stop; keep a checkpoint chosen on separate episodes. (4) A robustness test must be able to fail: this one could not separate the policies.

**Limits.** One game with four state variables; 3 seeds; 10-episode periodic tests are noisy; hyper-parameters were not tuned (a different bin layout or learning rate could change Q-learning a lot); wall-clock times were measured on a busy shared CPU and are not reported. **Next step:** keep the best checkpoint on separate validation episodes, tune the Q-table bins, and try a harder environment (LunarLander) where a hand rule is not available.

**Understand first.** Why is a one-line rule as good as DQN on CartPole? Why report seeds, not a single run?

**Exercise (before reading my code).** Write the rule 'push toward the side the pole leans' yourself and score it for 100 episodes.


---

## 26 - Multi-Agent Research Assistant with a Verifier
**Topic:** AI Agents  |  [reference article](https://amanxai.com/2025/09/16/building-a-multi-agent-system-using-gemini-api/)  |  folder `26 Multi-Agent Research Assistant - AI Agents/`

**Reference approach.** three Gemini-powered agents in sequence: **Planner** (splits a topic into sub-topics), **Researcher** (writes about each), **Synthesizer** (merges into a report). The agents' output is judged by reading it.

**What my notebook does.** build the same pipeline but with **grounding**: the researcher retrieves passages from real Wikipedia articles and the writer must cite them; I add a fourth agent, the **Verifier**, that checks each sentence against its cited evidence with a language-inference model. Then I **measure**: (1) can the verifier catch planted errors (changed numbers, negations, swapped names, off-topic sentences)? (2) how often does an *abstractive* writer produce sentences the evidence does not support, as judged by a second, independent model? (3) does the verifier-guarded pipeline reduce that rate, and at what cost?
**LLM back-end.** The agents run without any paid API: the planner uses fixed question templates, the researcher uses sentence-embedding retrieval, the writers are an extractive picker and **DistilBART** (a small summarisation model). A `ClaudeAgent` class in section 9 shows where an LLM would replace the planner and writer, but **it was not run** (no API key was available in this session).

**Side by side.**

| | Reference (Gemini agents) | This notebook |
|---|---|---|
| Agents | Planner, Researcher, Synthesizer (all LLM) | Planner (templates), Researcher (retrieval from 807 Wikipedia passages), Writer (extractive or DistilBART), **Verifier** |
| Evidence | model's own knowledge | cited Wikipedia passages |
| Quality check | read the output | verifier tested on planted errors; independent judge; unsupported-sentence rate with bootstrap interval |
| Findings | a report | verifier catches 86-100% of planted errors **but false-alarms on 64% of genuine sentences**; judge: 0% false alarms, 83-100% detection; unsupported rate 0.7-2.5% for all three writers with overlapping intervals; the guard replaced 56.6% of sentences for no measurable gain |

**Main lessons.** (1) Test a checker on **correct** inputs as well as on damaged ones: the verifier looked excellent on errors and was unusable on good text. (2) The checker needs its own checker: I validated the judge on the same planted errors before trusting it. (3) A small summarisation model stays close to its sources, so its unsupported rate is already low; the guard's cost (half of the sentences rewritten) outweighs its benefit here. (4) Faithful sentences can still lose context (pronouns, cut quotations).

**Limits.** No LLM was run (`ClaudeAgent` in section 9 is illustrative); 12 articles and fixed question templates; planted errors are crude and few for numbers (2) and names (7); the judge has its own error floor of a few percent; retrieval returns similar sentences for different questions. **Next steps:** run a real LLM writer with the same verifier and judge, tune the verifier threshold on genuine sentences (or use the DeBERTa model as verifier), and re-rank retrieval with a cross-encoder.

**Understand first.** Why can a verifier look excellent on planted errors and be useless on good text? Who checks the checker?

**Exercise (before reading my code).** Run the verifier on 20 sentences you know are true and count the false alarms.


---

## 27 - End to End Chatbot
**Topic:** AI Agents  |  [reference article](https://amanxai.com/2023/03/27/end-to-end-chatbot-using-python/)  |  folder `27 End to End Chatbot - AI Agents/`

**Reference approach.** a small hand-written intents file (a few intents with example phrases), a TF-IDF + logistic-regression classifier, a canned response per intent, and a Streamlit chat page.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | a tiny hand-written intents file | CLINC150: 150 intents x 100 questions + out-of-scope (4,500 + 1,000 test) |
| Model | TF-IDF + logistic regression | TF-IDF 91.4%, embeddings + 5-NN 89.9%, **embeddings + logistic regression 95.3%** |
| Unknown questions | not handled | three strategies: never refuse (0% refused), threshold 0.79 (91.7% refused, in-scope accuracy 86.0%), trained oos class (60.6% refused, 96.0% precision) |
| Robustness | not tested | typos: 95.3% -> 84.7% (4% of letters) -> 52.5% (15%) |
| Chatbot | canned responses | handlers for 12 intents, honest fallbacks, restricted calculator, 25/25 scripted messages |

**Main lessons.** (1) A chatbot is mostly an *out-of-scope* problem: accuracy on known intents (95%) says little about what happens on the 18% of questions it should not act on. (2) A confidence threshold and a trained reject class are two points on a trade-off; choose by the cost of a wrong action. (3) Pretrained sentence encoders beat TF-IDF on paraphrases but not on garbled text. (4) Test dialogues I wrote myself are a smoke test, not an evaluation.

**Limits.** English only; single-turn (no dialogue memory or slot filling); handlers for 12 of 150 intents; the typo noise is synthetic; the CPU timings were taken on a busy machine. **Next step:** multi-turn state (follow-up questions), spelling correction, and a calibrated reject option tuned to a stated cost.

**Understand first.** Why is a chatbot mostly an out-of-scope problem? Threshold vs a trained reject class?

**Exercise (before reading my code).** Plot in-scope accuracy against out-of-scope refusal as you move a confidence threshold.


---

## 28 - Predictive Keyboard Model (PyTorch LSTM)
**Topic:** Deep Learning  |  [reference article](https://amanxai.com/2025/05/13/building-a-predictive-keyboard-model-with-pytorch/)  |  folder `28 Predictive Keyboard LSTM - Deep Learning/`

**Reference approach.** a word-level LSTM in PyTorch trained on a small text, shown with a few example completions.

**What my notebook does.** see the notebook's first cell.

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Data | a small text | WikiText-2 (2.0 M training tokens), validation and test sets |
| Model | one word-level LSTM | LSTM (2.25 M parameters, 2 x 200 units, tied weights) vs **trigram counts** vs **word frequency** |
| Metric | example completions | perplexity 83 vs 132 vs 352; top-3 accuracy 42.0 / 37.8 / 23.3%; **keystrokes saved 46.6 / 45.6 / 35.1%** |
| Analysis | none | accuracy by word frequency, example suggestions, vocabulary coverage |

**Main lessons.** (1) A counting baseline is a strong competitor on the product metric: the LSTM wins on perplexity and accuracy but only by 1 point on keystrokes saved. (2) The score depends on the vocabulary: with 8,000 words 12% of tokens cannot be suggested at all. (3) Language models learn the style of their data (Wikipedia), not "English". (4) My LSTM was still improving when training stopped.

**Limits.** Six epochs on a shared CPU (validation perplexity still falling); one run, no seeds; the keystroke simulation uses 150 windows and a three-suggestion bar, with no personalisation; perplexity is not comparable with the literature. **Next steps:** train to convergence with learning-rate decay, a Transformer language model, a larger vocabulary or sub-word units for rare words, and per-user adaptation.

**Understand first.** Why does the LSTM win on perplexity but barely on keystrokes saved? Why does vocabulary size cap the score?

**Exercise (before reading my code).** Compute keystrokes saved for a trigram counter on a text you know and compare with the notebook.


---

## 29 - Generative AI Model from Scratch (mini GPT)
**Topic:** Deep Learning  |  [reference article](https://amanxai.com/2024/08/05/generative-ai-model-from-scratch-with-python/)  |  folder `29 Mini GPT from Scratch - Deep Learning/`

**Reference approach.** a small generative model built and run, with a text sample shown.

**What my notebook does.** write the transformer **from the individual pieces** (token and position embeddings, masked multi-head self-attention, feed-forward blocks, residual connections, layer norm), train it on the 1.1-million-character Tiny Shakespeare text, and then **measure** it: (1) validation loss against simple baselines (uniform, letter frequencies, and counting models that look back 1, 2 or 4 characters); (2) how much of the context the model actually uses; (3) generated text under different sampling temperatures, scored with objective measures (share of generated words that occur in the training text, repetition, diversity, and **verbatim copying** of the training text).

**Side by side.**

| | Reference | This notebook |
|---|---|---|
| Model | a small generative model built and run | a decoder-only transformer written from its parts (0.82 M parameters, 4 layers, 4 heads, context 128) |
| Evaluation | a text sample shown | validation loss against counting baselines: **1.546** vs 1.757 (4-character n-gram), 4.174 (uniform) |
| Context | not examined | loss flattens after about 8 characters of context (1.577 at 8, 1.523 at 128) |
| Generated text | one sample | four temperatures, objective measures: valid words 97.9% (T=0.5) to 76.7% (T=1.3), no verbatim 40-character copying |
| Inside the model | not shown | attention distance grows with depth (about 2 characters in layer 0, up to about 12 in layer 3) |

**Main lessons.** (1) A language model should be judged first by held-out loss against simple baselines; here the transformer's gain over a 4-character counter is real but moderate (about 12%). (2) Most of the useful context is the last few characters; a longer window helped little for a model this small. (3) Sample quality measures at this scale (valid words, repetition, copying) are blunt: several did not separate the settings, and at T=1.0 the counting model matched the transformer on valid words. (4) Temperature is a trade between fluency and variety, not a quality knob. (5) Reading generated text by eye is easy to over-trust: it *sounds* like Shakespeare while meaning almost nothing.

**Limits.** Characters, not words (so the vocabulary is tiny and spelling is part of the task); one training run, one seed, no tuning of size or learning rate; the validation set is the last 10% of one play collection, so style is not independent of the training part; the measures use 20 samples per setting with no intervals; the model is far too small to say anything about large models. **Next step:** repeat with several seeds, add dropout or early stopping (the train-validation gap is growing), and compare a word-piece vocabulary.

**Understand first.** What does masking do in self-attention? Why is most of the gain in the last 8 characters?

**Exercise (before reading my code).** Re-implement one attention head in 15 lines and check the mask by printing the weights.


---

## 30 - Synthetic Records with GANs
**Topic:** Deep Learning  |  [reference article](https://amanxai.com/2025/08/12/building-synthetic-medical-records-using-gans/)  |  folder `30 Synthetic Medical Records GAN - Deep Learning/`

**Reference approach.** train a GAN on a medical table and compare a few generated rows or distributions with the real ones.

**What my notebook does.** use a real public medical table (**Pima Indians Diabetes**, 768 women, 8 measurements + a diabetes outcome) and compare **four ways to make synthetic data**: (1) *copying* real rows, (2) *shuffling each column independently*, (3) a **Gaussian copula** (a classical statistical generator), (4) a **WGAN-GP** (a GAN that trains stably). Each is scored on three axes that pull against each other: **fidelity** (does it look real?), **utility** (does a model trained on it work on real patients? "train on synthetic, test on real") and **privacy** (does it leak individuals? distance to the closest real record and a membership-inference attack).

**Side by side.**

| | Reference (as I understood it) | This notebook |
|---|---|---|
| Question | can a GAN make medical-looking rows? | is the synthetic table realistic **and** useful **and** private, and does the GAN beat simple generators? |
| Generators | one GAN | resample (copy), independent columns, **Gaussian copula**, **WGAN-GP** (3 seeds) |
| Checks | look at samples / distributions | fidelity (KS, correlations, a classifier that tries to tell real from fake), utility (train on synthetic, test on real) and privacy (nearest real patient, membership attack) |
| Result | plausible rows | copula and GAN reach about the same utility (AUC 0.85 LR vs real-data 0.87); **the copula is the more faithful**, the GAN exaggerates zero insulin (80% vs 53%) and spreads into implausible ranges; copying wins on fidelity and fails privacy (membership AUC 0.82) |

**Main lessons.** (1) Fidelity, utility and privacy are three different questions and the trivial generator (copy the rows) wins two of them and fails the third, so a single "looks real" check says little. (2) A GAN is not automatically better than a classical statistical model on a small table of 768 rows: it needed 16 minutes of training for three seeds and was less faithful than a copula that needs no training loop. (3) Missing values coded as zero must be treated before generation, or the generator learns the coding. (4) Baselines that should fail (independent columns) are useful: they showed that the utility score depends on the dependence between columns.

**Limits.** 768 patients; one table; 3 GAN seeds; one simple privacy attack and no formal guarantee; utility measured on a single outcome with two classifiers; the hold-out set is only 231 patients (AUC differences of 0.02 are inside the noise). **Next step:** impute the zeros first, try conditional tabular GANs and a differentially-private variant, and add a stronger membership attack.

**Understand first.** Why are fidelity, utility and privacy three different questions? Why did copying the rows beat the GAN on two of them?

**Exercise (before reading my code).** Compare a Gaussian copula with a GAN on a table you know using the three tests.
