import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


class ModelTraining:
    '''
    Класс для обучения и оценки моделей машинного обучения.
    '''
    def __init__(self, random_state=42, test_size=0.2):
        '''
        Конструктор класса ModelTraining.

        Args:
            random_state (int): Random state для воспроизводимости результатов.
            test_size (float): Размер тестовой выборки.
        '''
        self.random_state = random_state
        self.test_size = test_size
        self.model = LinearRegression()
        self.imputer = SimpleImputer(strategy='mean') # Impute missing values with the mean

    def prepare_data_for_age_prediction(self, df):
        '''
        Подготавливает данные для предсказания возраста.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            tuple: X (features), y (target).
        '''
        # Select features for age prediction
        features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin']
        X = df[features].copy() # Use .copy() to avoid SettingWithCopyWarning

        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=['Pclass', 'Sex', 'Embarked'])

        y = df['Age'].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Combine X and y for NaN removal
        combined = pd.concat([X, y], axis=1)
        combined = combined.dropna()

        # Separate X and y after NaN removal
        X = combined.iloc[:, :-1]
        y = combined.iloc[:, -1]


        return X, y

    def impute_missing_age(self, df):
        '''
        Заполняет пропущенные значения в столбце 'Age' с использованием модели машинного обучения.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с заполненными значениями в столбце 'Age'.
        '''
        X, y = self.prepare_data_for_age_prediction(df)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Train the model
        self.model.fit(X_train, y_train)

        # Impute missing values in 'Age' column using the trained model
        age_null_index = df['Age'].isnull()
        # Prepare the data for imputation using the same transformations as for training
        X_missing_age = pd.get_dummies(df.loc[age_null_index, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'HasCabin']],
                                        columns=['Pclass', 'Sex', 'Embarked'])
        # Ensure the columns in X_missing_age match the columns used during training
        X_missing_age = X_missing_age.reindex(columns=X.columns, fill_value=0)


        predicted_ages = self.model.predict(X_missing_age)
        df.loc[age_null_index, 'Age'] = predicted_ages
        return df



    def train_and_evaluate(self, df, features, target):
        '''
        Обучает модель и оценивает ее производительность.

        Args:
            df (pandas.DataFrame): DataFrame с данными.
            features (list): Список признаков для обучения.
            target (str): Целевая переменная.

        Returns:
            float: RMSE (Root Mean Squared Error).
        '''

        # Remove rows with NaN in features and target *before* splitting
        df_cleaned = df[features + [target]].dropna().copy()

        X = df_cleaned[features]
        y = df_cleaned[target]


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse
