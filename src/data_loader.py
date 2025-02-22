import pandas as pd
import requests
from io import StringIO


class DataLoader:
    '''
    Класс для загрузки данных из URL.
    '''
    def __init__(self, url):
        '''
        Конструктор класса DataLoader.

        Args:
            url (str): URL для загрузки данных.
        '''
        self.url = url

    def load_data(self):
        '''
        Загружает данные из CSV-файла по указанному URL.

        Returns:
            pandas.DataFrame: DataFrame с загруженными данными, или None в случае ошибки.
        '''
        try:
            response = requests.get(self.url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return df
        except requests.exceptions.RequestException as e:
            print(f'Ошибка при загрузке данных: {e}')
            return None
        except pd.errors.ParserError as e:
            print(f'Ошибка при парсинге CSV данных: {e}')
            return None
        except Exception as e:
            print(f'Произошла непредвиденная ошибка: {e}')
            return None