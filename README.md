# Skill enhancement: 30 data-science projects

Thirty practice projects, three per topic across ten topics (regression, classification, clustering, computer vision, NLP, recommendation systems, time series, data visualization, AI agents, deep learning). The problems come from the project lists published on *Python in Plain English* and *amanxai.com*. For each one I solved the problem with my own code, **compared it with the reference article's method and results**, and wrote down where the two differ and why.

Every folder holds an executed notebook (outputs included), a README generated from the notebook, and a `model/` folder with the trained models. One multipage Streamlit app (`app/`) lets you try the saved models.

> **Status: reviewed, ready to publish.** The GitHub repository and Hugging Face Space links will be added here once they exist.

## Where to start

* [STUDY_GUIDE.md](STUDY_GUIDE.md): per project, the reference's approach vs mine, the lessons, the limits and an exercise to try before reading my code.
* Each folder's README.md and notebook for the details; the Streamlit app to try the saved models.

## How to read these projects

The references are mostly short tutorials: build a model, print one score, stop. I kept their *question* and added the checks a tutorial leaves out: a simple baseline, an honest train/test design (for example forward-in-time splits), uncertainty intervals, a null test where one is possible, and a limits section. Several projects ended with a finding that differs from the headline one in the tutorial: where a baseline matched the model, where the reference's data turned out to be synthetic or duplicated, and where a leaked split inflated the score. The one-line findings below are taken from the notebooks' own summaries; the exact numbers and the caveats are in each notebook.

## The 30 projects

| # | Topic | Project | What the notebook found |
|---|---|---|---|
| 01 | Regression | [Car Insurance Modelling](01%20Car%20Insurance%20Modelling%20-%20Regression/) | Boosting beat both a plain and a better-specified GLM (paired-bootstrap interval excludes 0), but a third of the plain GLM's gap was its weak specification; the reference's data file is no longer online, so I compared with its printed table. |
| 02 | Regression | [Price Elasticity of Demand](02%20Price%20Elasticity%20of%20Demand%20-%20Regression/) | The reference file has duplicated rows and inconsistent money columns; a regression with an interval gives an elasticity the data cannot distinguish from zero. |
| 03 | Regression | [Bike Sharing Demand](03%20Bike%20Sharing%20Demand%20-%20Regression/) | The whole problem lives in the hour-by-day-type curves; a forward-in-time split exposes a growth bias a random split hides. *Not on the lists* (replaces a project that repeated earlier course work). |
| 04 | Classification | [Loan Approval Prediction](04%20Loan%20Approval%20Prediction%20-%20Classification/) | On a few hundred rows the simplest analyst rule matches the ML models; a single split can mislead by about 7 points. |
| 05 | Classification | [Credit Scoring and Risk Tiers](05%20Credit%20Scoring%20and%20Risk%20Tiers%20-%20Classification/) | A transparent WoE scorecard matched gradient boosting (AUC 0.79, KS 0.47); the reference had no outcome to validate against. |
| 06 | Classification | [Customer Churn: Comparing Many Models](06%20Customer%20Churn%20Model%20Comparison%20-%20Classification/) | 7 of 13 models are statistically tied with the best; the simplest model in the top group is the sensible pick. |
| 07 | Clustering | [Geospatial Clustering (earthquakes)](07%20Geospatial%20Clustering%20-%20Earthquake%20Hotspots%20-%20Clustering/) | k-means gives zones, DBSCAN finds belts and noise; the answer to "how many clusters" depends on the question and the parameters. |
| 08 | Clustering | [User Profiling & Segmentation](08%20User%20Profiling%20and%20Segmentation%20-%20Clustering/) | k-means always returns k clusters; a shuffled-columns null and a stability test decide whether the segments mean anything. |
| 09 | Clustering | [RFM Analysis](09%20RFM%20Analysis%20-%20Clustering/) | RFM segments are rules, not discoveries; revenue concentration is far below "80/20" in this data. |
| 10 | Computer vision | [Handwriting Recognition](10%20Handwriting%20Recognition%20-%20Computer%20Vision/) | A CNN makes about three times fewer errors than k-NN, but a 4-pixel shift costs the plain CNN 30 points of accuracy until augmentation is added. |
| 11 | Computer vision | [Sign Language Classification](11%20Sign%20Language%20Classification%20-%20Computer%20Vision/) | A random split reports about 75% on near-duplicate photos; a group split gives about 58%. |
| 12 | Computer vision | [Flower / Colour Recognition](12%20Colour%20Recognition%20-%20Computer%20Vision/) | The gain from a perceptual colour space over RGB is within the uncertainty; lighting, not distance, is the hard part. |
| 13 | NLP | [Text Classification (Hugging Face)](13%20Text%20Classification%20Pipeline%20-%20NLP/) | The same TF-IDF model scores 70% to 93% depending on the split and on leftover headers; a CPU-budget fine-tuned DistilBERT (66.7%) did not beat TF-IDF (70.3%), which says little about a full-data transformer. |
| 14 | NLP | [Text Summarization](14%20Text%20Summarization%20-%20NLP/) | The reference's t5-small does worse than copying the first three sentences; DistilBART trained for the task is clearly better. |
| 15 | NLP | [Document Analysis (GitHub terms of service)](15%20Document%20Analysis%20-%20NLP/) | Retrieval was good, picking between candidate answers was the weak point; summarising "the first 512 tokens" summarises the preamble. |
| 16 | Recommendation | [Music Recommendation](16%20Music%20Recommendation%20System%20-%20Recommendation%20Systems/) | Standardising features mattered more than the model (5% to 15% genre agreement); duplicated songs would have flattered every score. |
| 17 | Recommendation | [Hybrid Recommendation (MovieLens)](17%20Hybrid%20Recommendation%20System%20-%20Recommendation%20Systems/) | Hybrid NDCG@10 0.189 vs SVD 0.175 vs popularity 0.104; the best method depends on the situation (popularity for new users). |
| 18 | Recommendation | [Market Basket Analysis](18%20Market%20Basket%20Analysis%20-%20Recommendation%20Systems/) | On the grocery data the rules are real (far stronger than in shuffled baskets) but complete a basket no better than popularity; the reference's own 250-basket file shows no real rules. |
| 19 | Time series | [Netflix Subscriptions Forecasting](19%20Netflix%20Subscriptions%20Forecasting%20-%20Time%20Series/) | The reference's ARIMA is no better than "last value plus recent growth"; nominal 80% intervals covered only 43-47%. |
| 20 | Time series | [Multivariate Time Series Forecasting](20%20Multivariate%20Time%20Series%20Forecasting%20-%20Time%20Series/) | "Multivariate" is not automatically better; predicting the change (not the level) was what made the tree model work. |
| 21 | Time series | [Stock Market Anomaly Detection](21%20Stock%20Market%20Anomaly%20Detection%20-%20Time%20Series/) | A volatility-adjusted score beat Isolation Forest on planted jumps; 76% of flags fell on earnings days. |
| 22 | Data visualization | [Netflix Content Strategy](22%20Netflix%20Content%20Strategy%20Analysis%20-%20Data%20Visualization/) | Missing-value audit and time-since-release confounding change what the standard charts say. |
| 23 | Data visualization | [Carbon Emissions and Warming](23%20Carbon%20Emissions%20Impact%20Analysis%20-%20Data%20Visualization/) | The file mixes a ppm series and a year-on-year % series in one column; trending series correlate by construction. |
| 24 | Data visualization | [Rainfall Trends in India](24%20Rainfall%20Trends%20in%20India%20-%20Data%20Visualization/) | "No visible trend" is not "no trend": the interval shows what a century of data can and cannot exclude. |
| 25 | AI agents | [Game Agent (CartPole)](25%20AI%20Agent%20to%20Master%20a%20Game%20-%20AI%20Agents/) | A one-line rule scores the maximum 500; DQN reached it on 3 of 3 seeds, Q-learning had one bad seed. |
| 26 | AI agents | [Multi-Agent Research Assistant with a Verifier](26%20Multi-Agent%20Research%20Assistant%20-%20AI%20Agents/) | The small verifier looked good on planted errors but false-alarmed on 64% of genuine sentences; no LLM was run. |
| 27 | AI agents | [End-to-End Chatbot](27%20End%20to%20End%20Chatbot%20-%20AI%20Agents/) | 95.3% on known intents but 0% refusal of out-of-scope questions unless designed in; typos cost 11 points at 4% noise. |
| 28 | Deep learning | [Predictive Keyboard (LSTM)](28%20Predictive%20Keyboard%20LSTM%20-%20Deep%20Learning/) | The LSTM wins on perplexity but saves only about 1 point more keystrokes than a trigram counter (46.6% vs 45.6%). |
| 29 | Deep learning | [Mini GPT from Scratch](29%20Mini%20GPT%20from%20Scratch%20-%20Deep%20Learning/) | The 0.8 M-parameter transformer beats a 4-character counter by about 12% in loss (1.546 vs 1.757); sample-quality measures were blunt. |
| 30 | Deep learning | [Synthetic Records with GANs](30%20Synthetic%20Medical%20Records%20GAN%20-%20Deep%20Learning/) | On 768 rows a copula needing no training loop was more faithful than the GAN; copying the rows wins fidelity and utility and fails privacy. |

## Run it

```
pip install -r requirements.txt
jupyter lab                          # open any notebook, e.g. 03/"03 Bike Sharing Demand.ipynb"
streamlit run app/app.py             # the multipage app, loads the saved models
```

The notebooks download their data on first run into `_data/` (not part of the repository; several files are large). The notebooks were run with Python 3.14.6 and the CPU build of torch; every notebook runs on a laptop CPU, but the longest (13, 25, 26, 28, 29) take from one to several hours (13 took 2.5 hours, 29 two hours).

## Data, models and licences

* Each notebook states where its data come from. The data belong to their original providers and keep their own licences; the repository does not redistribute the raw downloads.
* Some `model/` folders contain small derived files that are not models (a cleaned table, a passage index, embeddings): for example `22/model/netflix_clean.csv.gz`, `26/model/wikipedia_passages.csv.gz` (Wikipedia text, CC BY-SA 4.0) and `16/model/embeddings.npy`. Check the licence of the source before reusing them.
* The fine-tuned DistilBERT from project 13 is about 250 MB; it is not stored in Git and is recreated by the notebook (the Space and app use the two smaller models for 13).
* Saved scikit-learn models are pickles made with scikit-learn 1.9.0. **Do not load pickles from sources you do not trust.**

## Caveats

These are learning projects, not products. Several data sets are small, old or synthetic and each notebook says so. Nothing here is advice for lending, medicine, trading or any decision about real people. Numbers quoted are from one run each unless a notebook states repeated seeds or folds.
