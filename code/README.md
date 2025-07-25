#### Description of codes

- ```compute_rankings.py```: computes the rankings based on the indicated feature extraction and classification algorithms. 
Rankings are computed 200 times for each review and algorithm combination.
The algorithms for computing the rankings are based on the codes of the semi-automated screening tool [ASReview](https://github.com/asreview/asreview) (version 1).

- ```data_preprocessing.ipynb```: loads the datasets used for ranking and merges the corresponding TRIPOD scores and other abstract characterstics to the dataframe.


- ```results_generation.ipynb```: uses the datasets and computed rankings to create results on the associations between abstract characteristics and ranking positions.
