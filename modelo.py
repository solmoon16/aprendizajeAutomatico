import pandas as pd
import sys
from pycaret.regression import *
from pycaret.regression import RegressionExperiment

path = sys.argv[1]

dataset = pd.read_csv(path)
df = pd.DataFrame(dataset)

df = df.dropna(subset=['rating', 'description'])
df = df.drop(columns=['certificate', 'votes'])
s = setup(df, target = 'rating', session_id = 123)

exp = RegressionExperiment()

exp.setup(df, target = 'rating', session_id = 123)

best = create_model('knn')
final_model = finalize_model(best)
save_model(final_model, 'knn_model')