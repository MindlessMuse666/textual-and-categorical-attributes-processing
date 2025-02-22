import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from feature_engineering import FeatureEngineering
from model_training import ModelTraining
from data_loader import DataLoader


def main():
    '''
    Основная функция для запуска процесса загрузки данных, feature engineering, обучения модели и оценки результатов.
    '''
    
    # 1. Загрузка данных
    data_loader = DataLoader(url='https://github.com/datasciencedojo/datasets/blob/master/titanic.csv?raw=true')
    df = data_loader.load_data()

    if df is None:
        print('Не удалось загрузить данные. Выход.')
        return


    # 2. Feature Engineering
    feature_engineer = FeatureEngineering()
    df = feature_engineer.extract_name_features(df)
    df = feature_engineer.encode_embarked(df)
    df = feature_engineer.encode_pclass(df)
    df = feature_engineer.encode_sex(df)
    df = feature_engineer.create_description(df)
    df = feature_engineer.vectorize_description(df)
    df = feature_engineer.create_cabin_feature(df)


    # 3. Заполнение пропущенных значений в столбце 'Age'
    model_trainer = ModelTraining()
    df = model_trainer.impute_missing_age(df)


    # 4. Обучение модели и оценка результатов
    # Определяем признаки и целевую переменную для модели (берём, например: предсказание 'Fare')
    features = ['Pclass_0', 'Pclass_1', 'Pclass_2', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'HasCabin']
    target = 'Fare'

    # Удаляем строки с пропущенными значениями в выбранных признаках
    df_cleaned = df[features + [target]].dropna()

    # До feature engineering
    rmse_before = model_trainer.train_and_evaluate(df_cleaned, ['Age', 'SibSp', 'Parch'], target) # ограниченные признаки
    print(f'RMSE до feature engineering: {rmse_before}')

    # После feature engineering
    rmse_after = model_trainer.train_and_evaluate(df_cleaned, features, target)
    print(f'RMSE после feature engineering: {rmse_after}')


    # 5. Визуализация (берём, например: Распределение возраста)
    
    # 5.1. Визуализация через Seaborn
    plt.figure(figsize=(10, 6), num='Распределение возраста')
    sns.histplot(df['Age'], kde=True)
    plt.title('Распределение возраста', pad=20)
    plt.xlabel('Возраст')
    plt.ylabel('Частота')
    plt.savefig('report/figures/age_distribution.png') # Сохраняем график в PNG
    plt.show()

    # 5.2. Визуализация через Plotly
    fig = px.scatter(df, x='Age', y='Fare', color='Pclass', hover_data=['Name'])
    fig.update_layout(title='Зависимость стоимости билета от возраста и класса') # Название графика
    fig.write_image('report/figures/age_fare_scatter.png') # Сохраняем график в PNG

    # Сохраняем в HTML
    html_file = "report/figures/age_fare_scatter.html"
    fig.write_html(html_file)

    fig.show()


if __name__ == '__main__':
    main()