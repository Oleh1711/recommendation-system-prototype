import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import pickle
import os


class ContentBasedModel:
    def __init__(self, train_path, catalog_path, cache_dir):
        self.train_path = train_path
        self.catalog_path = catalog_path
        self.cache_dir = cache_dir

        self.catalog = None
        self.train_data = None
        self.vectorizer = None
        self.item_vectors = None
        self.user_profiles = None
        self.item_id_to_idx = {}

    def load_data(self):
        print("Завантаження даних для контентної моделі")
        self.train_data = pd.read_parquet(self.train_path)
        self.catalog = pd.read_parquet(self.catalog_path)

    def build_item_vectors(self):
        print("Векторизація каталогу (TF-IDF)")
        self.catalog['text_features'] = (
                self.catalog['category_l1'].astype(str) + " " +
                self.catalog['category_l2'].astype(str) + " " +
                self.catalog['brand'].astype(str)
        ).str.lower().str.replace('unknown', '')

        self.vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.item_vectors = self.vectorizer.fit_transform(self.catalog['text_features'])

        self.item_id_to_idx = {prod_id: idx for idx, prod_id in enumerate(self.catalog.index)}

    def build_user_profiles(self):
        print("Побудова контентних профілів користувачів")

        event_weights = {'view': 1.0, 'cart': 3.0, 'purchase': 5.0}
        self.train_data['weight'] = self.train_data['event_type'].map(event_weights).fillna(1.0).astype(float)

        user_item_weights = self.train_data.groupby(['user_id', 'product_id'])['weight'].sum().reset_index()
        valid_items = user_item_weights['product_id'].isin(self.item_id_to_idx)
        user_item_weights = user_item_weights[valid_items]

        unique_users = user_item_weights['user_id'].unique()
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}

        row_indices = user_item_weights['user_id'].map(user_id_to_idx).values
        col_indices = user_item_weights['product_id'].map(self.item_id_to_idx).values
        data_weights = user_item_weights['weight'].values

        interaction_matrix = sp.csr_matrix(
            (data_weights, (row_indices, col_indices)),
            shape=(len(unique_users), self.item_vectors.shape[0])
        )

        raw_user_profiles = interaction_matrix.dot(self.item_vectors)

        normalized_profiles = normalize(raw_user_profiles, norm='l2', axis=1)

        self.user_profiles = {
            user_id: normalized_profiles[idx]
            for user_id, idx in user_id_to_idx.items()
        }

    def save_to_feature_store(self):
        print("Консервація профілів та векторизатора у Feature Store")
        os.makedirs(self.cache_dir, exist_ok=True)

        item_data = {
            'vectors': self.item_vectors,
            'id_to_idx': self.item_id_to_idx
        }

        with open(os.path.join(self.cache_dir, 'content_item_vectors.pkl'), 'wb') as f:
            pickle.dump(item_data, f)

        with open(os.path.join(self.cache_dir, 'content_user_profiles.pkl'), 'wb') as f:
            pickle.dump(self.user_profiles, f)

        with open(os.path.join(self.cache_dir, 'content_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def run(self):
        self.load_data()
        self.build_item_vectors()
        self.build_user_profiles()
        self.save_to_feature_store()
        print("Контентна модель успішно розрахована.")