import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import av
from pandas import DataFrame, Series  # for convenience

import pims
import trackpy as tp

toLink = pd.read_csv('test.csv') 

t = tp.link_df(toLink, 5, memory=3)

t.to_csv('output.csv')