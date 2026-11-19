# 16 - Music Recommendation System
**Topic:** Recommendation Systems  |  **Where the project came from:** page list  |  **Reference:** [reference article](https://amanxai.com/2023/07/31/music-recommendation-system-using-python/)

**Topic:** Recommendation Systems  |  **Reference:** "Music Recommendation System using Spotify API" (amanxai.com, Jul 2023)

**Question.** Given a song, which other songs sound similar, and how would we know the list is any good when we have no listener data?

**Reference method:** get a Spotify developer token, download the audio features of some tracks through the API, and recommend by similarity of those numbers. (Spotify removed access to audio-feature data for new apps in November 2024 and the method needs personal credentials, so I use a public snapshot of **114,000 Spotify tracks with the same audio features** from the Hugging Face hub.)
**What I do:** clean the data (the same song appears many times), compare **ways of measuring "similar"** (raw numbers, standardised numbers, and a **learned embedding** from a small neural network trained to tell genres apart), and evaluate with measurable proxies on held-out songs: do the neighbours share a genre, how varied are the lists, and are a few songs recommended over and over ("hubs")?

---

## 9. Summary: my approach vs the reference
| | Reference | This notebook |
|---|---|---|
| Data | Spotify API (needs credentials; audio-feature access closed to new apps in Nov 2024) | public snapshot: 114,000 rows -> 81,195 unique songs, same audio features |
| Cleaning | none | duplicates merged (title + artist), genres kept as a set |
| Similarity | cosine on the API numbers | raw vs standardised vs PCA-whitened vs learned embedding |
| Evaluation | by listening | genre precision@10 on held-out songs: **raw 5.3% < whitened 15.0% < standardised 15.4% < learned 16.6%** (random 1.2%), coverage, hubness, popularity drift, listening check |

**What I learned.** (1) Putting features on a common scale mattered more than any model: raw numbers gave 5% vs 15% after standardising. (2) Duplicated songs would have made every score look better; they must be merged before evaluating. (3) Without user data we can only measure proxies; genre agreement is useful but is not "I would enjoy this". (4) Audio descriptors alone are weak for mainstream music but fine for distinctive acoustic styles (classical).

**Limits.** One random split of songs; genre is a crude proxy for taste; one embedding architecture, one seed; no listener data, lyrics or artist information; the data set has an artificial 1,000-songs-per-genre balance. **Next step:** evaluate with real playlists (does the recommended song co-occur in a playlist with the seed?) and blend with collaborative filtering.

---

## Files
* `16 Music Recommendation System.ipynb` - executed notebook (outputs included)
* Saved models (`model/`): `embeddings.npy`, `genre_embedding_net.pt`, `meta.json`, `scaler_genres.joblib`, `songs.csv.gz`

## Run it
```
pip install -r ../requirements.txt
jupyter lab "16 Music Recommendation System.ipynb"
```
Data is downloaded and cached into `../_data/` on first run (that folder is not part of the repository).
