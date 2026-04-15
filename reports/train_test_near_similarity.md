# Train/Test Near Similarity Report

This report checks for soft leakage by finding test reviews that are highly similar to train reviews using TF-IDF + cosine nearest neighbors.

- Train rows: **25000**
- Test rows: **25000**
- Test rows after exact-overlap removal: **24877**
- Exact overlap unique texts: **123**
- Exact overlap test rows removed: **123**
- Top-k neighbors per test row: **3**
- Suspicious similarity threshold: **0.90**

## Counts by threshold

- Pairs with similarity ≥ **0.85**: **5** (conflicting labels: **0**)
- Pairs with similarity ≥ **0.90**: **2** (conflicting labels: **0**)
- Pairs with similarity ≥ **0.95**: **1** (conflicting labels: **0**)

CSV output: `reports/train_test_near_similarity.csv`
JSON summary: `reports/train_test_near_similarity_summary.json`

## Top 20 suspicious pairs

 test_id  train_id  neighbor_rank  similarity  test_label  train_label  same_label                                                                                                                                                                                                                                        test_text                                                                                                                                                                                                                                       train_text
   22793     16351              1    0.991582           1            1        True The Three Stooges has always been some of the many actors that I have loved. I love just about every one of the shorts that they have made. I love all six of the Stooges (Curly, Shemp, Moe, Larry, Joe, and Curly Joe)! All of the shorts a... The Three Stooges has always been some of the many actors that I have loved. I love just about every one of the shorts that they have made. I love all six of the Stooges (Curly, Shemp, Moe, Larry, Joe, and Curly Joe)! All of the shorts a...
    7787      7819              1    0.900366           0            0        True Want a great recipe for failure? Take a crappy, leftist political plot, add in some weak & completely undeveloped characters and then throw in the worst sequences a movie has ever known. Let stew for a week (the amount of time probably s... Want a great recipe for failure? Take a s****y plot, add in some weak, completely undeveloped characters and than throw in the worst special effects a horror movie has known. Let stew for a week (the amount of time probably spent making ...