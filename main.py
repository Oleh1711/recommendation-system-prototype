from src.pipeline.data_pipeline import DataPipeline
from src.models.content_based import ContentBasedModel
from src.models.collaborative import CollaborativeModel
from src.stream.stream_processor import StreamRecommender
from src.evaluation.evaluator import ModelEvaluator

if __name__ == "__main__":
    input_file = 'data/2019-Nov.csv'
    processed_folder = 'data/processed'
    cache_folder = 'data/cache'

    pipeline = DataPipeline(input_path=input_file, output_dir=processed_folder, sample_size=1000000)
    pipeline.run()

    print("\n Старт пакетної обробки: Контентна модель ")
    content_model = ContentBasedModel(
        train_path=f'{processed_folder}/train_events.parquet',
        catalog_path=f'{processed_folder}/item_catalog.parquet',
        cache_dir=cache_folder
    )
    content_model.run()

    print("\nСтарт пакетної обробки: Колаборативна модель ")
    collab_model = CollaborativeModel(
        train_path=f'{processed_folder}/train_events.parquet',
        catalog_path=f'{processed_folder}/item_catalog.parquet',
        cache_dir=cache_folder
    )
    collab_model.run()

    print("\n Підняття денного контуру ")
    recommender = StreamRecommender(cache_dir=cache_folder, catalog_path=f'{processed_folder}/item_catalog.parquet')
    recommender.load_feature_store()

    print("\nСтарт фінального порівняльного експерименту ")
    evaluator = ModelEvaluator(
        test_path=f'{processed_folder}/test_events.parquet',
        recommender=recommender,
        top_k=10
    )
    evaluator.run_experiment()
    print("\n Симуляція реального кліку користувача ")

    import random
    import time
    test_user = random.choice(list(recommender.content_users.keys()))

    print(f"Юзер {test_user} переглянув товар")
    print(f"Система генерує персональні рекомендації:\n")

    start = time.perf_counter()
    recs = recommender.get_recommendations(test_user, top_k=10, n_explore=3)
    latency = (time.perf_counter() - start) * 1000

    print(recs.to_string(index=False))
    print(f"\nЧас відповіді системи: {latency:.2f} мс")


    user_id = test_user
    old_profile = recommender.content_users[user_id]

    random_new_item_id = recommender.catalog.sample(1).index[0]
    item_info = recommender.catalog.loc[random_new_item_id]
    print(f"Юзер раптово клікає на новий товар: {random_new_item_id}")
    print(f"Категорія: {item_info['category_code']}, Бренд: {item_info['brand']}")

    item_idx = recommender.content_items['id_to_idx'][random_new_item_id]
    new_item_vector = recommender.content_items['vectors'][item_idx]

    updated_profile = old_profile + (new_item_vector * 3.0)

    recommender.content_users[user_id] = updated_profile

    print("\nСистема перебудовує вектор профілю та перераховує видачу")

    start = time.perf_counter()
    new_recs = recommender.get_recommendations(user_id, top_k=10, n_explore=2)
    latency = (time.perf_counter() - start) * 1000

    print(new_recs.to_string(index=False))
    print(f"\nЧас перерахунку всієї матриці після кліку: {latency:.2f} мс")

    recommender.content_users[user_id] = old_profile
