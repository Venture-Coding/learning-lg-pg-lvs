import os
import sys
import json
from time import time 

import pandas as pd

path = "/Users/s748779/IAG_test/st/parts"
sys.path.insert(0, path) 

import prompt_texts as pt
import prompt_texts_cc as pc
import aa_utils as ut



def read_few_shot_data():
    n_shots_file = "/Users/s748779/IAG_test/st/files/train_data_l1l2_aug.csv"
    # Load the CSV file into a DataFrame
    df = pd.read_csv(n_shots_file)
    df2 = df.copy(deep=True)
    return df, df2





    
def main():

    ptc = pc.prompt_guide
    df, df2 = read_few_shot_data()
    
    for i in df2.index:
        context = str(df2.loc[i].to_json())
        cost_rationale = ut.get_cost_rationale_for_query(context, ptc)
        df2.loc[i, "cost_rationale"] = [cost_rationale]
        df2.to_csv("train_data_l1l2_aug_w_cr.csv", index=False)
    
    # df3 = df2.iloc[: ,  [0,1,2,3,4,6,7]]
    # df3.to_csv("train_data_l1l2_aug2.csv", index=False)

    return True

if __name__ == "__main__" :
    pt = pt.pt
    ptc = pc.prompt_guide
    main()

