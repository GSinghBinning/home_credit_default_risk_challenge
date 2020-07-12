
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

RAW_DIRECTORY = 'data\\raw\\'

import src.data.load as sdl

# sdl.load_dataset(RAW_DIRECTORY)

df = sdl.read_test_train(RAW_DIRECTORY)
