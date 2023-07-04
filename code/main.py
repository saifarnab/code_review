import sys

from prepossessing import prepossessing
from add_code_model import add_code_model
from remove_code_model import remove_code_model
from operation_model import operation_model
from utils import calculate_confidence_score

if __name__ == '__main__':
    processed_df, df_without_not_enough, df_with_not_enough = prepossessing.process(str(sys.argv[1]))
    add_code_df = add_code_model.process(processed_df)
    remove_code_df = remove_code_model.process(processed_df)
    operation_df = operation_model.process(processed_df)
    calculate_confidence_score(df_without_not_enough, df_with_not_enough, add_code_df, remove_code_df, operation_df)
