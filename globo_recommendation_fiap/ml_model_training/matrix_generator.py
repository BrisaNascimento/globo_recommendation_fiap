import pandas as pd
from scipy.sparse import coo_matrix


class MatrixGenerator:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def get_matrix(self, column: str, row: str, value: str) -> list:
        user_map = {u: i for i, u in enumerate(self.data[row])}
        item_map = {p: i for i, p in enumerate(self.data[column])}
        self.data[f'{row}_idx'] = self.data[row].map(user_map)
        self.data[f'{column}_idx'] = self.data[column].map(item_map)
        matrix = coo_matrix((
            self.data[value],
            (self.data[f'{row}_idx'], self.data[f'{column}_idx']),
        ))
        return [user_map, item_map, matrix]


if __name__ == '__main__':
    data = {
        'user_id': [1, 1, 2, 2, 3],
        'page_id': [101, 102, 101, 103, 104],
        'interaction_score': [5, 3, 4, 2, 1],
    }
    row = 'user_id'
    column = 'page_id'
    value = 'interaction_score'
    df = pd.DataFrame(data)
    matrix_gen = MatrixGenerator(df)
    result = matrix_gen.get_matrix(column=column, row=row, value=value)[2]

    num_users, num_pages = result.shape
    print(f'Matrix Shape: {num_users} users x {num_pages} pages')
    print(result)
