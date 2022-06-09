import os
import numpy as np
import pandas as pd
import woe.feature_process as fp

def woe(infile_path, target_column_name):
    # CREATE CONFIG CSV
    config_path = os.getcwd()+'\\../data/woe/config_woe.csv'
    # data_path = os.getcwd()+'\\../data/woe/woe_data.csv'
    data_path = infile_path
    feature_detail_path = os.getcwd()+'\\features_detail.csv'
    rst_pkl_path = os.getcwd()+'\\../pickles/woe/woe_rule.pkl'
    # train woe rule
    feature_detail, rst = fp.process_train_woe(infile_path=data_path
                                           ,outfile_path=feature_detail_path
                                           ,rst_path=rst_pkl_path
                                           ,config_path=config_path)

    # proc woe transformation
    woe_path = os.getcwd()+'\\dataset_woed.csv'
    fp.process_woe_trans(data_path, rst_pkl_path, woe_path, config_path)
    return woe_path