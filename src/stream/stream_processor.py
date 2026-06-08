import pandas as pd
import numpy as np
import pickle
import os


class StreamRecommender:
    def __init__(self, cache_dir, catalog_path):
        self.cache_dir = cache_dir
        self.catalog_path = catalog_path

        self.catalog = None
        self.content_items = None
        self.content_users = None
        self.collab_items = None
        self.collab_users = None

        self.popularity_penalty = None
        self.master_idx = {}
        self.idx_to_item_id = {}

        self.content_idx_map = None
        self.content_raw_idx = None
        self.collab_idx_map = None
        self.collab_raw_idx = None
        self.item_categories_array = None

    def load_feature_store(self):
        print("Підняття Feature Store в оперативну пам'ять")

        self.catalog = pd.read_parquet(self.catalog_path)

        catalog_item_ids = self.catalog.index.values
        self.master_idx = {item_id: idx for idx, item_id in enumerate(catalog_item_ids)}
        self.idx_to_item_id = {idx: item_id for item_id, idx in self.master_idx.items()}

        self.popularity_penalty = self.catalog['popularity_weight'].values

        self.item_categories_array = np.array([
            self.catalog.loc[self.idx_to_item_id[idx], 'category_l1']
            for idx in range(len(self.master_idx))
        ])

        with open(os.path.join(self.cache_dir, 'content_item_vectors.pkl'), 'rb') as f:
            self.content_items = pickle.load(f)
        with open(os.path.join(self.cache_dir, 'content_user_profiles.pkl'), 'rb') as f:
            self.content_users = pickle.load(f)

        with open(os.path.join(self.cache_dir, 'collab_item_factors.pkl'), 'rb') as f:
            self.collab_items = pickle.load(f)
        with open(os.path.join(self.cache_dir, 'collab_user_factors.pkl'), 'rb') as f:
            self.collab_users = pickle.load(f)

        c_items, c_raw = [], []
        for item_id, idx in self.content_items['id_to_idx'].items():
            if item_id in self.master_idx:
                c_items.append(self.master_idx[item_id])
                c_raw.append(idx)
        self.content_idx_map = np.array(c_items)
        self.content_raw_idx = np.array(c_raw)

        col_items, col_raw = [], []
        for item_id, idx in self.collab_items['id_to_idx'].items():
            if item_id in self.master_idx:
                col_items.append(self.master_idx[item_id])
                col_raw.append(idx)
        self.collab_idx_map = np.array(col_items)
        self.collab_raw_idx = np.array(col_raw)

    def get_recommendations(self, user_id, top_k=10, alpha=0.5, lambda_param=0.5, n_explore=0):
        if user_id not in self.content_users or user_id not in self.collab_users:
            return None

        n_items = len(self.master_idx)
        content_scores = np.zeros(n_items)
        collab_scores = np.zeros(n_items)

        u_content = self.content_users[user_id]
        raw_content = u_content.dot(self.content_items['vectors'].T).toarray().flatten()
        content_scores[self.content_idx_map] = raw_content[self.content_raw_idx]

        u_collab = self.collab_users[user_id]
        raw_collab = np.dot(self.collab_items['factors'], u_collab.T).flatten()
        collab_scores[self.collab_idx_map] = raw_collab[self.collab_raw_idx]

        def max_normalize(arr):
            arr = np.maximum(arr, 0)
            max_v = arr.max()
            if max_v > 1e-9:
                return arr / max_v
            return arr

        c_norm = max_normalize(content_scores)
        col_norm = max_normalize(collab_scores)

        hybrid_scores = (alpha * c_norm) + ((1 - alpha) * col_norm)

        final_scores = hybrid_scores + (0.2 * self.popularity_penalty)

        candidate_indices = np.argsort(final_scores)[::-1][:50]
        selected_indices = []
        selected_categories = set()

        while len(selected_indices) < top_k and len(candidate_indices) > 0:
            if not selected_indices:
                best_idx = candidate_indices[0]
            else:
                best_mmr_score = -np.inf
                best_idx = None

                for idx in candidate_indices:
                    relevance = final_scores[idx]
                    item_cat = self.item_categories_array[idx]
                    diversity = 0.0 if item_cat in selected_categories else 1.0
                    mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = idx

            selected_indices.append(best_idx)
            selected_categories.add(self.item_categories_array[best_idx])
            candidate_indices = candidate_indices[candidate_indices != best_idx]

        if n_explore > 0 and len(selected_indices) == top_k:
            shown_cats = np.unique(self.item_categories_array[selected_indices])
            explore_mask = ~np.isin(self.item_categories_array, shown_cats)
            explore_pool = np.where(explore_mask)[0]

            if len(explore_pool) > 0:
                pool_weights = self.popularity_penalty[explore_pool]
                weights_norm = pool_weights / pool_weights.sum()

                n_to_pick = min(n_explore, len(explore_pool))
                top_explore_rel = np.random.choice(
                    len(explore_pool),
                    size=n_to_pick,
                    replace=False,
                    p=weights_norm
                )

                top_explore = explore_pool[top_explore_rel]
                selected_indices[-n_to_pick:] = top_explore.tolist()

        recommendations = []
        for idx in selected_indices:
            item_id = self.idx_to_item_id[idx]
            item_info = self.catalog.loc[item_id]
            recommendations.append({
                'product_id': item_id,
                'category': item_info['category_code'],
                'brand': item_info['brand'],
                'score': round(final_scores[idx], 4)
            })

        return pd.DataFrame(recommendations)