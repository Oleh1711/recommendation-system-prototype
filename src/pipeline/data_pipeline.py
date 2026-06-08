import pandas as pd
import numpy as np
import os


class DataPipeline:
    def __init__(self, input_path, output_dir, sample_size=None):
        self.input_path = input_path
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.raw_data = None
        self.clean_data = None
        self.item_catalog = None
        self.train_data = None
        self.test_data = None
        self.snapshot_before = None
        self.snapshot_after = None

    def load_data(self):
        print("Читаємо сирий CSV...")
        dtypes = {
            'event_type': 'category',
            'product_id': 'int32',
            'category_id': 'int64',
            'brand': 'category',
            'price': 'float32',
            'user_id': 'int32',
            'user_session': 'category'
        }

        self.raw_data = pd.read_csv(
            self.input_path,
            dtype=dtypes,
            parse_dates=['event_time']
        )

        if self.sample_size:
            print(f"Беремо повну історію {self.sample_size} випадкових юзерів для тесту")
            all_users = self.raw_data['user_id'].unique()
            actual_sample = min(self.sample_size, len(all_users))
            sampled_users = np.random.choice(all_users, size=actual_sample, replace=False)
            self.raw_data = self.raw_data[self.raw_data['user_id'].isin(sampled_users)]

        self.raw_data = self.raw_data[self.raw_data['price'] > 0]
        self.raw_data.dropna(subset=['product_id', 'user_id'], inplace=True)
        self.snapshot_before = self.raw_data.copy()

    def clean_noise_and_bots(self):
        df = self.raw_data.copy()

        min_date = df['event_time'].min()
        cutoff_date = min_date + pd.DateOffset(days=7)
        df = df[df['event_time'] >= cutoff_date]

        df = df.sort_values(by=['user_id', 'event_time'])

        df['time_diff'] = df.groupby('user_id')['event_time'].diff().dt.total_seconds()

        valid_interactions = df['time_diff'].isna() | (df['time_diff'] >= 1.0)
        df = df[valid_interactions]

        view_counts = df[df['event_type'] == 'view'].groupby('user_id').size()
        purchase_counts = df[df['event_type'] == 'purchase'].groupby('user_id').size()

        user_stats = pd.DataFrame({
            'total_clicks': view_counts,
            'total_purchases': purchase_counts
        }).fillna(0)

        normal_users = user_stats[
            (user_stats['total_clicks'] <= 3500) &
            (user_stats['total_purchases'] <= 1000)
            ].index

        df = df[df['user_id'].isin(normal_users)]

        user_counts = df['user_id'].value_counts()
        item_counts = df['product_id'].value_counts()

        active_users = user_counts[user_counts >= 10].index
        popular_items = item_counts[item_counts >= 10].index

        df = df[df['user_id'].isin(active_users) & df['product_id'].isin(popular_items)]

        self.clean_data = df.drop(columns=['time_diff']).reset_index(drop=True)
        self.snapshot_after = self.clean_data.copy()

    def plot_cleaning_proof(self, output_dir='.'):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if self.snapshot_before is None or self.snapshot_after is None:
            print("Знімки відсутні, спочатку запусти load_data та clean_noise_and_bots")
            return

        df_b = self.snapshot_before
        df_a = self.snapshot_after

        # Малюємо стан ДО
        fig_b, axes_b = plt.subplots(1, 2, figsize=(16, 6))
        fig_b.suptitle(f'Стан датасету ДО очищення\nВсього рядків {len(df_b):,}', fontsize=14, fontweight='bold')

        users_before = df_b['user_id'].value_counts()
        axes_b[0].hist(np.log1p(users_before), bins=60, color='#e74c3c', edgecolor='black', alpha=0.85)
        axes_b[0].axvline(np.log1p(3500), color='darkred', linestyle='--', linewidth=2, label='Межа бота (3500 дій)')
        axes_b[0].set_title(f'Активність юзерів ({len(users_before):,} унікальних)', fontweight='bold')
        axes_b[0].set_xlabel('log(кількість дій)')
        axes_b[0].set_ylabel('Кількість користувачів')
        axes_b[0].legend()

        items_before = df_b['product_id'].value_counts().values
        cutoff_b = max(1, int(len(items_before) * 0.10))
        axes_b[1].plot(range(cutoff_b), items_before[:cutoff_b], color='#e74c3c', linewidth=2)
        axes_b[1].fill_between(range(cutoff_b), items_before[:cutoff_b], color='#e74c3c', alpha=0.25)
        axes_b[1].set_title(f'Довгий хвіст товарів ({len(items_before):,} унікальних)', fontweight='bold')
        axes_b[1].set_xlabel('Ранг товару (топ 10%)')
        axes_b[1].set_ylabel('Кількість взаємодій')

        plt.tight_layout()
        save_path_before = os.path.join(output_dir, 'cleaning_proof_before.png')
        fig_b.savefig(save_path_before, dpi=300, bbox_inches='tight')
        plt.close(fig_b)

        # Малюємо стан ПІСЛЯ
        fig_a, axes_a = plt.subplots(1, 2, figsize=(16, 6))
        fig_a.suptitle(f'Стан датасету ПІСЛЯ очищення\nВсього рядків {len(df_a):,}', fontsize=14, fontweight='bold')

        users_after = df_a['user_id'].value_counts()
        axes_a[0].hist(np.log1p(users_after), bins=60, color='#2ecc71', edgecolor='black', alpha=0.85)
        axes_a[0].set_title(f'Активність юзерів ({len(users_after):,} унікальних)', fontweight='bold')
        axes_a[0].set_xlabel('log(кількість дій)')
        axes_a[0].set_ylabel('Кількість користувачів')

        items_after = df_a['product_id'].value_counts().values
        cutoff_a = max(1, int(len(items_after) * 0.10))
        axes_a[1].plot(range(cutoff_a), items_after[:cutoff_a], color='#2ecc71', linewidth=2)
        axes_a[1].fill_between(range(cutoff_a), items_after[:cutoff_a], color='#2ecc71', alpha=0.25)
        axes_a[1].set_title(f'Довгий хвіст товарів ({len(items_after):,} унікальних)', fontweight='bold')
        axes_a[1].set_xlabel('Ранг товару (топ 10%)')
        axes_a[1].set_ylabel('Кількість взаємодій')

        plt.tight_layout()
        save_path_after = os.path.join(output_dir, 'cleaning_proof_after.png')
        fig_a.savefig(save_path_after, dpi=300, bbox_inches='tight')
        plt.close(fig_a)

    def extract_catalog(self):
        print("Формуємо каталог товарів та рахуємо вагу популярності")

        def get_first_valid(series):
            valid = series.dropna()
            return valid.iloc[0] if not valid.empty else 'unknown'

        catalog = self.clean_data.groupby('product_id').agg(
            category_id=('category_id', 'first'),
            category_code=('category_code', get_first_valid),
            brand=('brand', get_first_valid),
            unique_users=('user_id', 'nunique')
        ).reset_index()

        catalog[['category_l1', 'category_l2']] = catalog['category_code'].astype(str).str.split('.', n=1, expand=True)
        catalog['category_l2'] = catalog['category_l2'].fillna('unknown')

        catalog['popularity_weight'] = 1.0 / (1.0 + np.log1p(catalog['unique_users']))

        self.item_catalog = catalog.set_index('product_id')

    def create_holdout_split(self):
        print("Формуємо тренувальну та тестову вибірки")
        df = self.clean_data.copy()

        purchases = df[df['event_type'] == 'purchase']
        self.test_data = purchases.groupby('user_id').tail(1)
        self.train_data = df.drop(self.test_data.index)

    def save_processed_data(self):
        print("Зберігаємо результати у Parquet")
        os.makedirs(self.output_dir, exist_ok=True)

        self.train_data.to_parquet(os.path.join(self.output_dir, 'train_events.parquet'), index=False)
        self.test_data.to_parquet(os.path.join(self.output_dir, 'test_events.parquet'), index=False)
        self.item_catalog.to_parquet(os.path.join(self.output_dir, 'item_catalog.parquet'), index=True)

    def run(self):
        self.load_data()
        self.clean_noise_and_bots()
        self.plot_cleaning_proof(output_dir=self.output_dir)
        self.extract_catalog()
        self.create_holdout_split()
        self.save_processed_data()
        print(f"Залишилось чистих записів для тренування: {len(self.train_data)}")