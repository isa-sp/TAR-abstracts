import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel

os.environ["TRANSFORMERS_CACHE"] = "./models/transformers/cache/"

class ComputeRankings:
    def __init__(self, dataset, feature_extractor, classifier, initial_inclusions=10, initial_exclusions=10,
                 batch_size=10, random_seed=None):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.initial_inclusions = initial_inclusions
        self.initial_exclusions = initial_exclusions
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

        if 'abstract' not in self.dataset.columns or 'label_included' not in self.dataset.columns:
            raise ValueError("Dataset must have 'abstract' and 'label_included' columns.")

        # Initialize data storage
        self.original_order = list(self.dataset.index)
        self.final_order = []
        self.training_data = None
        self.training_labels = None
        self.remaining_data = None

    def initialize_training_data(self):
        """Randomly selects initial inclusions and exclusions for training."""
        available_inclusions = self.dataset[self.dataset['label_included'] == 1]
        available_exclusions = self.dataset[self.dataset['label_included'] == 0]

        if len(available_inclusions) < self.initial_inclusions or len(available_exclusions) < self.initial_exclusions:
            raise ValueError("Not enough inclusions or exclusions in the dataset.")

        inclusions = available_inclusions.sample(n=self.initial_inclusions, random_state=self.random_seed)
        exclusions = available_exclusions.sample(n=self.initial_exclusions, random_state=self.random_seed)

        self.training_data = pd.concat([inclusions, exclusions])
        self.training_labels = self.training_data['label_included']

        # Store remaining records
        self.remaining_data = self.dataset.drop(self.training_data.index)
        self.final_order = self.training_data.index.tolist()

    def preprocess(self):
        """Converts text abstracts into features based on the feature extractor."""
        self.training_data['abstract'] = self.training_data['abstract'].fillna('NA').astype(str)
        self.remaining_data['abstract'] = self.remaining_data['abstract'].fillna('NA').astype(str)

        # Initialize X_train and X_remaining in case no condition matches
        X_train, X_remaining = None, None

        # Handle the feature extractor types
        if isinstance(self.feature_extractor, TfidfVectorizer):
            print("Using TfidfVectorizer")  # Debugging line
            X_train = self.feature_extractor.fit_transform(self.training_data['abstract'])
            X_remaining = self.feature_extractor.transform(self.remaining_data['abstract'])

        elif isinstance(self.feature_extractor, SentenceTransformer):
            print("Using SentenceTransformer")  # Debugging line
            X_train = np.array(self.feature_extractor.encode(self.training_data['abstract'].tolist()))
            X_remaining = np.array(self.feature_extractor.encode(self.remaining_data['abstract'].tolist()))

        elif isinstance(self.feature_extractor, BertModel):  # Check specifically for BertModel
            print("Using BERT Model")  # Debugging line
            X_train = self._get_huggingface_embeddings(self.training_data['abstract'])
            X_remaining = self._get_huggingface_embeddings(self.remaining_data['abstract'])

        elif isinstance(self.feature_extractor, AutoModel):  # For other Hugging Face models
            print("Using Hugging Face Model (AutoModel)")  # Debugging line
            X_train = self._get_huggingface_embeddings(self.training_data['abstract'])
            X_remaining = self._get_huggingface_embeddings(self.remaining_data['abstract'])

        else:
            # If no known feature extractor is passed, raise an error
            raise ValueError(f"Feature extractor type '{type(self.feature_extractor)}' is not recognized.")

        # If no valid extractor was found, raise an error
        if X_train is None or X_remaining is None:
            raise ValueError("Feature extractor type is not recognized or is incorrectly configured.")

        return X_train, X_remaining

    def train_model(self, X_train):
        """Trains the classifier on the training data."""
        self.classifier.fit(X_train, self.training_labels)

    def predict_and_rank(self, X_remaining):
        """Predicts probabilities and sorts remaining data by relevance."""
        probabilities = self.classifier.predict_proba(X_remaining)[:, 1]
        self.remaining_data['probability'] = probabilities
        self.remaining_data = self.remaining_data.sort_values(by='probability', ascending=False)

    def expand_training_data(self):
        """Moves the top-ranked batch of records into training data."""
        batch_size = min(self.batch_size, len(self.remaining_data))
        if batch_size == 0:
            return

        top_data = self.remaining_data.head(batch_size)
        self.training_data = pd.concat([self.training_data, top_data])
        self.training_labels = self.training_data['label_included']
        self.remaining_data = self.remaining_data.iloc[batch_size:]

        new_indices = top_data.index.tolist()
        self.final_order.extend(new_indices)

    def iterative_training(self):
        """Runs the active learning loop to iteratively improve rankings."""
        total_remaining = len(self.remaining_data)
        total_iterations = -(-total_remaining // self.batch_size)

        for _ in tqdm(range(total_iterations), desc="Screening progress", unit="iteration"):
            X_train, X_remaining = self.preprocess()
            self.train_model(X_train)
            self.predict_and_rank(X_remaining)
            self.expand_training_data()

        # Ensure all original indices are in final order (fix missing values issue)
        missing_indices = set(self.original_order) - set(self.final_order)
        self.final_order.extend(missing_indices)

        return self.original_order, self.final_order

    def _get_huggingface_embeddings(self, texts):
        """Generates Hugging Face model embeddings for a given set of texts."""
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Example model: BERT
        model = self.feature_extractor  # Hugging Face model passed here

        # Ensure texts is a list of strings
        texts = texts.tolist() if isinstance(texts, pd.Series) else texts

        # Tokenize the texts (ensure the input is in the correct format)
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return embeddings.numpy()

# Example Usage:
if __name__ == "__main__":
    data_path = "./data/processed/"
    output_path = "./output/rankings/"
    os.makedirs(output_path, exist_ok=True)


    def load_dataset(file_path):
        """Loads a dataset from CSV or Excel."""
        if file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            df['abstract'] = df['abstract'].fillna('NA').astype(str)
            return df
        return None


    # Loop over all datasets
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        dataset = load_dataset(file_path)

        if dataset is None:
            continue  # Skip if not a valid dataset

        # Define parameters
        initial_inclusions = 10
        initial_exclusions = 10
        batch_size = 50
        random_seeds = range(1, 4)

        # Define list of feature extractors to loop over
        feature_extractors = [
            #('huggingface', AutoModel.from_pretrained('bert-base-uncased')),
            #('sentence_transformer', SentenceTransformer('paraphrase-MiniLM-L6-v2')),
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000))
        ]

        # Loop over different feature extractors
        for feature_name, feature_extractor in feature_extractors:
            for random_seed in random_seeds:
                compute_rankings = ComputeRankings(
                    dataset,
                    feature_extractor,
                    LogisticRegression(),
                    initial_inclusions=initial_inclusions,
                    initial_exclusions=initial_exclusions,
                    batch_size=batch_size,
                    random_seed=random_seed
                )

                # Run ranking algorithm
                compute_rankings.initialize_training_data()
                original_order, final_order = compute_rankings.iterative_training()

                # Convert ranking outputs to DataFrames
                original_df = pd.DataFrame({'Original_Index': original_order})
                original_df['Original_Label'] = dataset.loc[original_df['Original_Index'], 'label_included'].values

                final_df = pd.DataFrame({'Final_Index': final_order})
                final_df['Final_Label'] = dataset.loc[final_df['Final_Index'], 'label_included'].values

                # Merge original and final rankings
                ranking_output = pd.concat([original_df, final_df], axis=1)

                # Save rankings to CSV
                output_file = f"{output_path}{os.path.splitext(filename)[0]}_rankings_{feature_name}_{random_seed}.csv"
                ranking_output.to_csv(output_file, index=False)
