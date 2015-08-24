import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import trackpy as tp

toLink = pd.read_csv('../locate/test.csv') 

t = tp.link_df(toLink, 10, memory=3)

t.to_csv('output.csv')