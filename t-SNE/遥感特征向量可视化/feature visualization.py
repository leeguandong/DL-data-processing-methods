import pandas as pd
import numpy as np

# feature = []
for i in range(100):
    if i < 10:
        df = pd.read_table('agricultural/agricultural0' + str(i) + '.jpg.txt')
        print('-----')
        print(df.shape)
    else:
        df = pd.read_table('agricultural/agricultural' + str(i) + '.jpg.txt')
        print('*******')
        print(df.shape)

    X = np.stack(df, axis=0)
# print(df)

print(X)
print(X.shape)
X = X.reshape((100, 2048))
# feature.append(df)
print(X.shape)
# print(feature.shape)
# X = np.array(feature)


# print(X)
