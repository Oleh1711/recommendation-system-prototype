import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import pickle
import os


class CollaborativeModel:
    def __init__(self, train_path, catalog_path, cache_dir, n_components=50):
        self.train_path = train_path
        self.catalog_path = catalog_path
        self.cache_dir = cache_dir
        self.n_components = n_components

        self.train_data = None
        self.catalog = None
        self.item_id_to_idx = {}
        self.user_profiles = None
        self.item_factors = None

    def load_data(self):
        print("Завантаження даних для колаборативної моделі")
        self.train_data = pd.read_parquet(self.train_path)
        self.catalog = pd.read_parquet(self.catalog_path)

        self.item_id_to_idx = {prod_id: idx for idx, prod_id in enumerate(self.catalog.index)}

    def build_matrix_and_factorize(self):
        print("Побудова розрідженої матриці та SVD факторизація")

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
            shape=(len(unique_users), len(self.item_id_to_idx))
        )

        svd = TruncatedSVD(n_components=self.n_components, random_state=42)

        user_factors_raw = svd.fit_transform(interaction_matrix)
        item_factors_raw = svd.components_.T

        user_factors_normalized = normalize(user_factors_raw, norm='l2', axis=1)
        self.item_factors = normalize(item_factors_raw, norm='l2', axis=1)

        self.user_profiles = {
            user_id: user_factors_normalized[idx]
            for user_id, idx in user_id_to_idx.items()
        }

    def save_to_feature_store(self):
        print("Консервація колаборативних факторів у Feature Store")
        os.makedirs(self.cache_dir, exist_ok=True)

        item_data = {
            'factors': self.item_factors,
            'id_to_idx': self.item_id_to_idx
        }

        with open(os.path.join(self.cache_dir, 'collab_item_factors.pkl'), 'wb') as f:
            pickle.dump(item_data, f)

        with open(os.path.join(self.cache_dir, 'collab_user_factors.pkl'), 'wb') as f:
            pickle.dump(self.user_profiles, f)

    def run(self):
        self.load_data()
        self.build_matrix_and_factorize()
        self.save_to_feature_store()
        print("Колаборативна модель успішно розрахована.")