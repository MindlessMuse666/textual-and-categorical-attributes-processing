import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class FeatureEngineering:
    '''
    Класс для выполнения Feature Engineering на данных Titanic.
    '''
    def __init__(self):
        '''
        Конструктор класса FeatureEngineering. Инициализирует TF-IDF векторизатор,
        One-Hot энкодер и Label энкодер.
        '''
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english') #added stop words
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.label_encoder = LabelEncoder()

    def extract_name_features(self, df):
        '''
        Разделяет столбец 'Name' на 'Фамилия' и 'Имя'.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с добавленными столбцами 'Фамилия' и 'Имя'.
        '''
        df[['Фамилия', 'Имя']] = df['Name'].str.split(',', n=1, expand=True)
        df['Имя'] = df['Имя'].str.split('.', n=1, expand=True)[1].str.strip()
        return df

    def encode_embarked(self, df):
        '''
        Преобразует столбец 'Embarked' в числовой формат.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с преобразованным столбцом 'Embarked'.
        '''
        df['Embarked'] = df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3}).fillna(0) # Fills NaN with 0
        return df

    def encode_pclass(self, df):
        '''
        Применяет One-Hot Encoding к столбцу 'Pclass'.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с добавленными столбцами One-Hot Encoding для 'Pclass'.
        '''
        pclass_encoded = self.onehot_encoder.fit_transform(df['Pclass'].values.reshape(-1, 1))
        pclass_df = pd.DataFrame(pclass_encoded, columns=[f'Pclass_{i}' for i in range(pclass_encoded.shape[1])])
        pclass_df.index = df.index # Keep the index in case of row deletion
        df = pd.concat([df, pclass_df], axis=1)

        return df

    def encode_sex(self, df):
        '''
        Применяет Label Encoding к столбцу 'Sex'.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с преобразованным столбцом 'Sex'.
        '''
        df['Sex'] = self.label_encoder.fit_transform(df['Sex'])
        return df

    def create_description(self, df):
        '''
        Создает столбец 'Description' на основе 'Name', 'Sex', 'Age' и 'Pclass'.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с добавленным столбцом 'Description'.
        '''
        df['Description'] = df['Name'].fillna('') + ' ' + df['Sex'].astype(str).fillna('') + ' ' + df['Age'].astype(str).fillna('') + ' ' + df['Pclass'].astype(str).fillna('')
        return df

    def vectorize_description(self, df):
        '''
        Применяет TF-IDF к столбцу 'Description'.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с добавленными столбцами TF-IDF для 'Description'.
        '''
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['Description'].fillna(''))
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
        tfidf_df.index = df.index # Keep the index in case of row deletion
        df = pd.concat([df, tfidf_df], axis=1)
        return df

    def create_cabin_feature(self, df):
        '''
        Создает бинарный признак 'HasCabin', указывающий на наличие информации в столбце 'Cabin'.

        Args:
            df (pandas.DataFrame): DataFrame с данными Titanic.

        Returns:
            pandas.DataFrame: DataFrame с добавленным столбцом 'HasCabin'.
        '''
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        return df
