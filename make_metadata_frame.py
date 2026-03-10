import pandas as pd
from ast import literal_eval

def safe_literal_eval(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError):
        if pd.isna(val):
            return float('nan')
        elif isinstance(val, (int, float)):
            return val
        else:
            return float('nan')


csv_file_path = './meta-data/Jax_mice_with_splits_df.csv'
output_pickle_file_path = 'meta-data/Jax_mice_with_splits_df.pickle'
df2 = pd.read_csv(csv_file_path, converters={'filepaths': literal_eval,
                                             'file_lengths':literal_eval,
                                             'file_train_end':literal_eval,
                                             'file_test_start':literal_eval,'exclusions':safe_literal_eval})

df2.to_pickle(output_pickle_file_path)
print(f"Successfully converted {csv_file_path} to {output_pickle_file_path}")
