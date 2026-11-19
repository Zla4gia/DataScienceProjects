# 18 - Market Basket Analysis
**Topic:** Recommendation Systems  |  **Where the project came from:** author's deep-learning/ML ideas list  |  **Reference:** [reference article](https://amanxai.com/2023/10/16/market-basket-analysis-using-python/)

**Topic:** Recommendation Systems  |  **Reference:** "Market Basket Analysis using Python" (amanxai.com, Oct 2023)

**Question.** Which products are bought together, and can that be used to suggest an extra item for a basket?

**Reference method:** a small basket file, Apriori to find frequent itemsets, association rules ranked by support / confidence / lift, and a reading of the top rules as business insights.
**What I add:** (1) the reference's own file (250 baskets) and a **real grocery data set (9,835 baskets, 171 items)**, so I can show what changes with data size; (2) a **null test**: are the rules any stronger than rules found in baskets with the same item popularity but random contents? (3) **hold-out validation**: do rules found in one part of the data keep their lift in another part? (4) a **basket-completion test**: hide one item from a held-out basket and see whether the rules put it in the top 5, against a popularity baseline.

---

## 9. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | 250-basket toy file (7 items) | the same file, plus real groceries: 9,835 baskets, 171 items |
| Method | Apriori and rules by support / confidence / lift | + FP-growth check, null test by shuffling, hold-out validation, basket-completion test |
| Is it real? | not tested | **groceries: yes** (1,325 rules with lift >= 1.5 vs 0.4 in shuffled data; top rules keep 88% of their lift on new baskets). **Reference file: no** (lifts 1.11-1.16, intervals include 1, two halves share 0.1 of the top 5 rules) |
| Is it useful? | "business insights" | rules complete a basket no better than popularity (26-27% vs 25.8% hit rate@5) |

**What I learned.** (1) Rank by lift, not confidence: confidence just rediscovers popular items. (2) Test rules against shuffled data and on held-out baskets before believing them. (3) A statistically real pattern need not be a useful prediction: say what the rule is *for*. (4) Small baskets with few items can give impressive-looking but empty rules.

**Limits.** The groceries file has no dates or customers, so the validation split is random; support threshold 0.5% hides rarer rules; the completion test hides one random item (one of several possible tasks); confidence intervals are over baskets. **Next step:** price and margin data to rank rules by profit, and a sequence-aware (next basket) model with timestamps.

---

## Files
* `18 Market Basket Analysis.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `basket_cooccurrence.joblib`, `groceries_rules.csv`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "18 Market Basket Analysis.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
