# 30 - Synthetic Records with GANs
**Topic:** Deep Learning  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2025/08/12/building-synthetic-medical-records-using-gans/)

**Topic:** Deep Learning  |  **Reference:** "Building Synthetic Medical Records using GANs" (amanxai.com, Aug 2025)

**Question.** Can a generative adversarial network (GAN) produce fake patient records that look and behave like real ones without exposing the real patients?

**Reference method:** train a GAN on a medical table and compare a few generated rows or distributions with the real ones.
**What I do:** use a real public medical table (**Pima Indians Diabetes**, 768 women, 8 measurements + a diabetes outcome) and compare **four ways to make synthetic data**: (1) *copying* real rows, (2) *shuffling each column independently*, (3) a **Gaussian copula** (a classical statistical generator), (4) a **WGAN-GP** (a GAN that trains stably). Each is scored on three axes that pull against each other: **fidelity** (does it look real?), **utility** (does a model trained on it work on real patients? "train on synthetic, test on real") and **privacy** (does it leak individuals? distance to the closest real record and a membership-inference attack).

---

## 9. Summary: my approach vs the reference
| | Reference (as I understood it) | This notebook |
|---|---|---|
| Question | can a GAN make medical-looking rows? | is the synthetic table realistic **and** useful **and** private, and does the GAN beat simple generators? |
| Generators | one GAN | resample (copy), independent columns, **Gaussian copula**, **WGAN-GP** (3 seeds) |
| Checks | look at samples / distributions | fidelity (KS, correlations, a classifier that tries to tell real from fake), utility (train on synthetic, test on real) and privacy (nearest real patient, membership attack) |
| Result | plausible rows | copula and GAN reach about the same utility (AUC 0.85 LR vs real-data 0.87); **the copula is the more faithful**, the GAN exaggerates zero insulin (80% vs 53%) and spreads into implausible ranges; copying wins on fidelity and fails privacy (membership AUC 0.82) |

**What I learned.** (1) Fidelity, utility and privacy are three different questions and the trivial generator (copy the rows) wins two of them and fails the third, so a single "looks real" check says little. (2) A GAN is not automatically better than a classical statistical model on a small table of 768 rows: it needed 16 minutes of training for three seeds and was less faithful than a copula that needs no training loop. (3) Missing values coded as zero must be treated before generation, or the generator learns the coding. (4) Baselines that should fail (independent columns) are useful: they showed that the utility score depends on the dependence between columns.

**Limits.** 768 patients; one table; 3 GAN seeds; one simple privacy attack and no formal guarantee; utility measured on a single outcome with two classifiers; the hold-out set is only 231 patients (AUC differences of 0.02 are inside the noise). **Next step:** impute the zeros first, try conditional tabular GANs and a differentially-private variant, and add a stronger membership attack.

---

## Files
* `30 Synthetic Medical Records GAN.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `gan_generator.pt`, `gan_support.joblib`, `gan_synthetic_sample.csv`, `meta.json`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "30 Synthetic Medical Records GAN.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
