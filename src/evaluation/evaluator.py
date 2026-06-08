
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns



class ModelEvaluator:
    def __init__(self, test_path, recommender, top_k=10):
        self.test_path = test_path
        self.recommender = recommender
        self.top_k = top_k
        self.test_data = None
        self.ground_truth = {}
        self.results = []

    def load_test_data(self):
        print("Завантаження тестової вибірки")
        self.test_data = pd.read_parquet(self.test_path)

        for user_id, group in self.test_data.groupby('user_id'):
            self.ground_truth[user_id] = set(group['product_id'].tolist())

    def calculate_metrics(self, recommended_ids, actual_ids):
        relevant = 0
        dcg = 0
        for i, item_id in enumerate(recommended_ids[:self.top_k]):
            if item_id in actual_ids:
                relevant += 1
                dcg += 1.0 / np.log2(i + 2)

        precision = relevant / self.top_k
        recall = relevant / len(actual_ids)

        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_ids), self.top_k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return precision, recall, ndcg

    def evaluate_model(self, name, alpha, use_penalty, baseline=False):
        print(f"Тестування моделі {name}")

        original_penalty = self.recommender.popularity_penalty.copy()

        if not use_penalty:
            self.recommender.popularity_penalty = np.ones_like(original_penalty)
        else:
            self.recommender.popularity_penalty = original_penalty

        metrics_sum = {'precision': 0, 'recall': 0, 'ndcg': 0}
        total_latency = 0
        recommended_catalog = set()
        valid_users = 0

        if baseline:
            top_popular_indices = self.recommender.catalog['unique_users'].values.argsort()[::-1][:self.top_k]
            baseline_recs = [self.recommender.idx_to_item_id[idx] for idx in top_popular_indices]

        for user_id, actual_ids in self.ground_truth.items():
            if user_id not in self.recommender.content_users or user_id not in self.recommender.collab_users:
                continue

            start_time = time.perf_counter()

            if baseline:
                rec_ids = baseline_recs
            else:
                current_explore = 3 if name == "Hybrid (Stream + Bias)" else 0

                recs_df = self.recommender.get_recommendations(
                    user_id,
                    top_k=self.top_k,
                    alpha=alpha,
                    n_explore=current_explore
                )

                if recs_df is None or recs_df.empty:
                    continue
                rec_ids = recs_df['product_id'].tolist()

            latency = (time.perf_counter() - start_time) * 1000

            p, r, n = self.calculate_metrics(rec_ids, actual_ids)

            metrics_sum['precision'] += p
            metrics_sum['recall'] += r
            metrics_sum['ndcg'] += n
            total_latency += latency
            recommended_catalog.update(rec_ids)
            valid_users += 1

        self.recommender.popularity_penalty = original_penalty

        if valid_users == 0:
            return

        total_catalog_size = len(self.recommender.master_idx)
        coverage = (len(recommended_catalog) / total_catalog_size) * 100

        self.results.append({
            'Model': name,
            'Precision@10': round(metrics_sum['precision'] / valid_users, 4),
            'Recall@10': round(metrics_sum['recall'] / valid_users, 4),
            'NDCG@10': round(metrics_sum['ndcg'] / valid_users, 4),
            'Coverage (%)': round(coverage, 2),
            'Latency (ms)': round(total_latency / valid_users, 2)
        })

    def run_experiment(self):
        self.load_test_data()

        self.evaluate_model(name="Baseline (Top-N)", alpha=0.5, use_penalty=False, baseline=True)
        self.evaluate_model(name="Content-Based", alpha=1.0, use_penalty=False)
        self.evaluate_model(name="Collaborative", alpha=0.0, use_penalty=False)
        self.evaluate_model(name="Hybrid (Stream + Bias)", alpha=0.5, use_penalty=True)

        results_df = pd.DataFrame(self.results)
        print("\n ФІНАЛЬНІ РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТУ ")
        print(results_df.to_string(index=False))

        self.plot_results(results_df)

    def plot_results(self, df):
        print("\nГенерація розширеного пакету графіків для дипломної роботи")
        sns.set_theme(style="whitegrid")
        colors = ['#7f8c8d', '#3498db', '#f39c12', '#2ecc71']

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df['Model']))
        width = 0.25

        ax.bar(x - width, df['Precision@10'], width, label='Precision@10', color='#3498db')
        ax.bar(x, df['Recall@10'], width, label='Recall@10', color='#2ecc71')
        ax.bar(x + width, df['NDCG@10'], width, label='NDCG@10', color='#f39c12')

        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=15)
        ax.set_title('Порівняння базових метрик якості (Precision, Recall, NDCG)')
        ax.legend()
        plt.tight_layout()
        plt.savefig('plot_1_metrics_comparison.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, row in df.iterrows():
            ax.scatter(row['Coverage (%)'], row['NDCG@10'], color=colors[i], s=300, edgecolor='black', zorder=5)
            ax.text(row['Coverage (%)'] + 1.5, row['NDCG@10'], row['Model'], fontsize=11, fontweight='bold')

        ax.set_xlabel('Різноманітність (Coverage %)', fontsize=12)
        ax.set_ylabel('Якість ранжування (NDCG@10)', fontsize=12)
        ax.set_title('Trade-off аналіз Точність проти Різноманітності', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('plot_2_tradeoff.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(9, 5))
        heat_df = df.set_index('Model')[['Precision@10', 'Recall@10', 'NDCG@10', 'Coverage (%)']]
        heat_norm = heat_df / heat_df.max()
        sns.heatmap(heat_norm, annot=heat_df, cmap="YlGnBu", fmt=".4g", cbar=False, ax=ax, annot_kws={"size": 12})
        ax.set_title('Теплова карта ефективності моделей', fontsize=14)
        ax.set_ylabel('')
        plt.tight_layout()
        plt.savefig('plot_3_heatmap.png', dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Model', y='Latency (ms)', data=df, palette=colors, ax=ax)
        ax.set_title('Швидкість реакції системи (Latency)', fontsize=14)
        ax.set_ylabel('Мілісекунди (ms)', fontsize=12)
        ax.set_xlabel('')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig('plot_4_latency.png', dpi=300)
        plt.close()

