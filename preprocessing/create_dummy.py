import pandas as pd

def create_dummy(dataframe):
    cat_vars = ['REASON', 'JOB']
    for var in cat_vars:
        cat_list = 'var' + '_' + 'var'
        cat_list = pd.get_dummies(dataframe[var], prefix=var)
        dataframe_new = dataframe.join(cat_list)
        dataframe = dataframe_new
    data_vars = dataframe.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]

    dataframe_final = dataframe[to_keep]

    return dataframe_final
