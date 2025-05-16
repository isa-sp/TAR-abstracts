import numpy as np
import pandas as pd
import pickle
import os
import shutil
#import seaborn as sns
from pathlib import Path
from asreview import ASReviewData, ASReviewProject
from asreview.review import ReviewSimulate
from asreview.models.classifiers import LogisticClassifier, NaiveBayesClassifier #, LSTMBaseClassifier, LSTMPoolClassifier, NN2LayerClassifier, RandomForestClassifier, SVMClassifier
from asreview.models.query import MaxQuery #, ClusterQuery, MaxRandomQuery, MaxUncertaintyQuery, RandomQuery, UncertaintyQuery
from asreview.models.balance import DoubleBalance #, SimpleBalance, UndersampleBalance
from asreview.models.feature_extraction import Doc2Vec, Tfidf #, SBERT, EmbeddingIdf, EmbeddingLSTM,
from asreview import open_state

path_data = '../data/processed/for_ranking/'
path_results = '../output/tmp/'
path_results_HPC = '../output/rankings'

# Import the systematic review datasets (numbering of the datasets is according to another related research)
df_prog1 = pd.read_excel(path_data+'Prog_reporting_labeled.xlsx')
df_prog3 = pd.read_excel(path_data+'Prog_tripod_labeled.xlsx')

review_dic = {#'Prog1': df_prog1,
              'Prog3': df_prog3#,
              }
for review in review_dic:
    review_dic[review]['title'] = review_dic[review]['title'].replace(np.nan, '')
    review_dic[review]['abstract'] = review_dic[review]['abstract'].replace(np.nan, '')
    review_dic[review]['title'] = review_dic[review]['title'].astype(str)
    review_dic[review]['abstract'] = review_dic[review]['abstract'].astype(str)

# Create a path to store the simulation results
os.chdir(path_results)
path_name = "simulation_results_new_evaluation"
project_path = Path(path_name)
project_path.mkdir(exist_ok=True)

# Define the function for ranking simulation(s)
def ASReview_simulation(review_id, review_data, path_name,
                        train_model=NaiveBayesClassifier(),
                        query_model=MaxQuery(), balance_model=DoubleBalance(), feature_model=Tfidf(),
                        n_simulations=10, n_model_update=10, n_prior_included=10, n_prior_excluded=10):
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
    # Create an id for the review based on name and models
    review_id = review_id + "_" + str(train_model.name) + "_" + str(feature_model.name) + "_" + str(query_model.name)

    # Derive the review length
    review_length = len(review_data)

    # Create an empty dictionary to store the rankings of the records of each simulation
    dict_rank = {}

    # Run the simulations
    for i in list(range(118, n_simulations + 1)):

        # Set the path for the temporary results
        project_path = Path(path_name)
        project_path.mkdir(exist_ok=True)

        # Run the simulation based on the settings and store the output at the project path
        print("Running simulation number", i)
        # (Derived from ASReview from here...
        reviewer = ReviewSimulate(
            init_seed=int(i),
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

            # Derive the rankings of the records and store in the dictionary
            dict_rank[i] = []
            dict_rank[i].append(labeled)

        # Close and remove the simulation results from the path
        shutil.rmtree(project_path)

        # Save the simulation
        os.chdir("../" + path_results_HPC)

        with open('sim_{review}_{simu}.p'.format(review=review_id, simu=i), 'wb') as f:
            pickle.dump(dict_rank, f)

    # Return the relevant results of the simulations
    return review_id, review_length, n_simulations, dict_rank

# Define the model(s) to be tested
train_models = [LogisticClassifier()]
feature_models = [Doc2Vec()]
query_models = [MaxQuery()]

# Test each different method (and interactions) with the following loop for Tfidf():
sim_list = []
multiple_sims_saved = []
for review in review_dic:
    for train_model in train_models:
        for feature_model in feature_models:
            for query_model in query_models:
                sim_list.append([review, train_model, feature_model, query_model])
                sim = ASReview_simulation(review_id = review, review_data = review_dic[review],
                                          path_name = path_name,
                                          train_model = train_model, query_model = query_model,
                                          balance_model = DoubleBalance(), feature_model = feature_model,
                                          n_simulations = 200, n_model_update = 10,
                                          n_prior_included = 10, n_prior_excluded = 10)
                multiple_sims_saved.append(sim)
