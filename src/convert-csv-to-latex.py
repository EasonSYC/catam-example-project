import pandas as pd
import numpy as np

file_list: list[str] = [
    # 'question-3-i',
    # 'question-3-iii-1',
    # 'question-3-iii-2',
    # 'question-3-iv',
    # 'question-4',
    # 'question-5-eq-4-1-diverge',
    'question-5-eq-4-2-converge',
    # 'question-5-eq-4-3-converge-slow',
    # 'question-5-eq-5',
]

for file_name in file_list:
    df: pd.DataFrame = pd.read_csv(f'data/{file_name}.csv', index_col = 0)
    df.to_latex(f'tex/{file_name}.tex')