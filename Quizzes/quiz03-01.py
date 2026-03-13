# %%
import numpy as np
from io import StringIO
from sklearn.feature_selection import mutual_info_classif

# Step 1: Prepare data

data = """
1.31	0.75	0
2.88	0.18	0
3.77	0.66	0
4.88	0.37	0
5.71	0.80	0
6.43	0.19	0
7.55	0.86	0
8.77	0.61	0
9.35	0.44	0
10.47	0.17	0
11.80	0.87	1
12.23	0.18	1
13.46	0.32	1
14.44	0.56	1
15.80	0.73	1
16.61	0.47	1
17.95	0.03	1
18.10	0.40	1
19.59	0.51	1
"""

array = np.genfromtxt(StringIO(data))

# %%
X = array[:, :-1]  # All columns except the last
y = array[:, -1]   # The last column

# %%

# Step 2: Calculate mutual information
mi = mutual_info_classif(X, y, random_state=2025)

# Step 3: Display results
print("MI(x1, y) =", mi[0])
print("MI(x2, y) =", mi[1])

# %%
