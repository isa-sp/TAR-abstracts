# Import the packages

import numpy as np
import pandas as pd
import pickle
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# parser = argparse.ArgumentParser(description='Running simulations in  parallel')
# parser.add_argument('-sim_id', type=int, default=1, help='Number')
# args = parser.parse_args()

path_data = '/Users/ispiero2/TAR-abstracts/data/processed/for_ranking/'  # /home/julius_te/ispiero/systematicreviews/data/data_HPC
path_results = '/Users/ispiero2/TAR-abstracts/output/output/Results_tmp/'  # /home/julius_te/ispiero/systematicreviews/results
path_results_HPC = '/Users/ispiero2/TAR-abstracts/output/output/simulations_original_datasets/'

# Import the prognosis datasets (numbering of the datasets is ordered by author, so numbers do not correspond to numbers in data prep)

#os.chdir(path_data)

# Prognosis dataset 1: model reporting
df_prog1 = pd.read_excel(path_data+'Prog_reporting_labeled.xlsx')

# Prognosis dataset 3: tripod
df_prog3 = pd.read_excel(path_data+'Prog_tripod_labeled.xlsx')

# Create dictionaries of the datasets

review_dic = {'Prog1': df_prog1,
              'Prog3': df_prog3,
              }

for review in review_dic:
    review_dic[review]['title'] = review_dic[review]['title'].replace(np.nan, '')
    review_dic[review]['abstract'] = review_dic[review]['abstract'].replace(np.nan, '')
    review_dic[review]['title'] = review_dic[review]['title'].astype(str)
    review_dic[review]['abstract'] = review_dic[review]['abstract'].astype(str)

# Create a path to store the simulation results

os.chdir(path_results)

from pathlib import Path
from asreview import ASReviewData, ASReviewProject
from asreview.review import ReviewSimulate

path_name = "simulation_results_new_evaluation"
project_path = Path(path_name)  # leave out on HPC
project_path.mkdir(exist_ok=True)  # leave out on HPC

# Define the function for ASReview simulation(s)

# Import the functions from ASReview
# Classifiers:
from asreview.models.classifiers import LogisticClassifier, LSTMBaseClassifier, LSTMPoolClassifier, \
    NaiveBayesClassifier, NN2LayerClassifier, RandomForestClassifier, SVMClassifier

# Query models:
from asreview.models.query import ClusterQuery, MaxQuery, MaxRandomQuery, MaxUncertaintyQuery, RandomQuery, \
    UncertaintyQuery

# Balance models:
from asreview.models.balance import DoubleBalance, SimpleBalance, UndersampleBalance

# Feature extraction models:
from asreview.models.feature_extraction import Doc2Vec, EmbeddingIdf, EmbeddingLSTM, SBERT, Tfidf

# For evaluation:
from asreview import open_state


def ASReview_simulation(review_id, review_data, path_name,
                        # sim_id, # add on HPC
                        train_model=NaiveBayesClassifier(),
                        query_model=MaxQuery(), balance_model=DoubleBalance(), feature_model=Tfidf(),
                        n_simulations=10,  # leave out on HPC
                        n_model_update=10, n_prior_included=10, n_prior_excluded=10):
    """
        Performs semi-automated title-abstract screening simulations
        based on a labeled review dataset using the open source codes of ASReview.

        review_id (str):                  name under which the user wants to save the output for the respective review
        review_data (pandas.DataFrame):   dataframe containing at least columns with the 'title', 'abstract', and 'label_included'
        path_name (str):                  name of the path at which the simulation results are stored temporarily in each iteration
        train_model:                      classification model used to classify (rank) the records for relevance, being either:
                                               LogisticClassifier(), LSTMBaseClassifier(), LSTMPoolClassifier(), NaiveBayesClassifier(),
                                               NN2LayerClassifier(), RandomForestClassifier(), or SVMClassifier()
                                               -> Default by ASReview is NaiveBayesClassifier()
        query_model:                      query method by which the screener is presented with records, being either:
                                               ClusterQuery(), MaxQuery(), MaxRandomQuery(), MaxUncertaintyQuery(), RandomQuery(), or UncertaintyQuery()
                                               -> Default by ASReview is MaxQuery()
        balance_model:                    balance method by which class imbalance is handled, being either:
                                               DoubleBalance(), SimpleBalance(), or UndersampleBalance()
                                               -> Default by ASReview is DoubleBalance()
        feature_model:                    feature model used to derive features from the title and abstract, being either:
                                               Doc2Vec(), EmbeddingIdf(), EmbeddingLSTM(), SBERT(), Tfidf()
                                               -> Default by ASReview is Tfidf()
        n_simulations (int):              number of complete simulations to perform, each time with a different initial training set
        n_prior_included (int):           number of included (relevant) records that are used in the initial training set
                                               -> Default by ASReview is 10
        n_prior_excluded (int):           number of excluded (irrelevant) records that are used in the initial training set
                                               -> Default by ASReview is 10
        n_model_update (int):             number of records of which the screening is simulated before the model is updated
                                               -> Default by ASReview is 10
    """

    # Create a list of numbers ranging from 1 to n_simulations
    # sim = list(range(1, n_simulations+1)) # sim = sim_id # replace on HPC

    # Create an id for the review based on name and models
    review_id = review_id + "_" + str(train_model.name) + "_" + str(feature_model.name) + "_" + str(query_model.name)

    # Derive the review length
    review_length = len(review_data)

    # Create an empty dictionary to store the rankings of the records of each simulation
    dict_rank = {}

    # Run the simulations
    for i in list(range(1, n_simulations + 1)):  # leave out on HPC

        # Set the path for the temporary results
        # path_name = path_name # leave out on HPC
        project_path = Path(
            path_name)  # project_path = Path(path_name+"{y}.{x}_simulation".format(x=sim, y=review_id)) # replace on HPC
        project_path.mkdir(exist_ok=True)

        # Run the simulation based on the settings and store the output at the project path
        print("Running simulation number", i)  # print(sim) # replace on HPC
        # (Derived from ASReview from here...
        reviewer = ReviewSimulate(
            init_seed=int(i),  # init_seed=int(sim) # replace on HPC
            as_data=ASReviewData(review_data),
            model=train_model,
            query_model=query_model,
            balance_model=balance_model,
            feature_model=feature_model,
            n_instances=n_model_update,
            project=ASReviewProject.create(
                project_path=project_path / "{y}.{x}_simulation".format(x=i, y=review_id),
                project_id="{y}.{x}".format(x=i, y=review_id),
                project_mode="simulate",
                project_name="{y}.{x}".format(x=i, y=review_id)
            ),
            n_prior_included=n_prior_included,
            n_prior_excluded=n_prior_excluded
        )
        reviewer.project.update_review(status="review")
        try:
            reviewer.review()
            reviewer.project.mark_review_finished()
        except Exception as err:
            reviewer.project.update_review(status="error")
            raise err
        # ...until here)

        # Open the stored simulation results
        with open_state(reviewer.project) as s:
            labeled = s.get_labeled()
            priors = s.get_priors()
            labels = labeled[~labeled['record_id'].isin(priors['record_id'])]

            # Derive the rankings of the records and store in the dictionary
            dict_rank[i] = []  # dict_rank[sim] = [] # replace on HPC
            dict_rank[i].append(labeled)  # dict_rank[sim].append(labeled) # replace on HPC

        # Close and remove the simulation results from the path
        shutil.rmtree(project_path)

        # Save the simulation # add on HPC
        os.chdir(path_results) # add on HPC

        with open('sim_{review}_{simu}.p'.format(review=review_id, simu=i), 'wb') as f: # add on HPC
            pickle.dump(dict_rank, f) # add on HPC

    # Return the relevant results of the simulations
    return review_id, review_length, n_simulations, dict_rank  # return review_id, review_length, dict_rank #n_simulations, dict_rank # replace on HPC


# A test run to test the function
# test_simulation = ASReview_simulation(review_id='Prog1', review_data=review_dic['Prog1'], path_name=path_name,
#                                       train_model=NaiveBayesClassifier(), query_model=MaxQuery(),
#                                       balance_model=DoubleBalance(), feature_model=Tfidf(),
#                                       n_simulations=10,
#                                       n_model_update=10, n_prior_included=10, n_prior_excluded=10)

# To assess the performance of ASReview, the reviews, the classification models,
# feature extraction models, and/or query models are varied.

# Define the classification model(s) to be tested
train_models = [LogisticClassifier()]

# Define the feature extraction model(s) to be tested
feature_models = [EmbeddingLSTM(), Doc2Vec()]  # EmbeddingIdf() does not work

# Define the query model(s) to be tested (for now no variation in query models)
query_models = [MaxQuery()]

# Test each different method (and interactions) with the following loop for Tfidf():
sim_list = []
multiple_sims_saved = []
for review in review_dic:
    for train_model in train_models:
        for feature_model in feature_models:
            for query_model in query_models:
                sim_list.append([review, train_model, feature_model, query_model])
                sim = ASReview_simulation(review_id = review, review_data = review_dic[review], # leave out on HPC
                                          path_name = path_name,
                                          train_model = train_model, query_model = query_model,
                                          balance_model = DoubleBalance(), feature_model = feature_model,
                                          n_simulations = 200, n_model_update = 10,
                                          n_prior_included = 10, n_prior_excluded = 10)
                multiple_sims_saved.append(sim) # leave out on HPC
