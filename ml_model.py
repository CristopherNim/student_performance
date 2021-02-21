import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
import pickle
from flask import Flask, request

np.random.seed(42)

df = pd.read_csv('StudentsPerformance.csv')
df.rename(columns={'race/ethnicity': 'race', 'parental level of education': 'parent_level_of_education',
                   'test preparation course': 'test_prep_course', 'math score': 'math_score',
                   'reading score': 'reading_score', 'writing score': 'writing_score'}, inplace=True)
# creating a categorical boolean mask

categorical_feature_mask = df.dtypes == object

# filtering out the categorical columns

categorical_cols = df.columns[categorical_feature_mask].tolist()

# instantiate the OneHotEncoder Object

one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)

# applying data
one_hot.fit(df[categorical_cols])
cat_one_hot = one_hot.transform(df[categorical_cols])

# creating Dataframe of the hot encoded columns

hot_df = pd.DataFrame(cat_one_hot, columns=one_hot.get_feature_names(input_features=categorical_cols))
df_OneHotEncoder = pd.concat([df, hot_df], axis=1).drop(columns=categorical_cols, axis=1)

X = df_OneHotEncoder.drop('math_score', axis=1)
y = df_OneHotEncoder['math_score']

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = Ridge(alpha=.99).fit(X, y)
model_scores = cross_val_score(estimator=model, X=X, y=y, cv=cv, n_jobs=-1)

print('accuracy for ridge model: %.1f' % (model_scores.mean() * 100))


def row_pred(row):
    row = np.column_stack(row)
    cols = ['gender', 'race', 'parent_level_of_education', 'lunch', 'test_prep_course', 'reading_score',
            'writing_score']

    newdf = pd.DataFrame(row, columns=cols)
    cat_ohe_new = one_hot.transform(newdf[categorical_cols])

    ohe_new_df = pd.DataFrame(cat_ohe_new, columns=one_hot.get_feature_names(input_features=categorical_cols))

    df_ohe_new = pd.concat([newdf, ohe_new_df], axis=1).drop(columns=categorical_cols, axis=1)

    pred_score = model.predict(df_ohe_new)
    a = pred_score.tolist()
    print(f'predicted math score: {a[0]:.0f}')
    # print(f'{a[0]:.0f}')
    return f'{a[0]:.1f}'


pickle.dump(model, open('model.pkl', 'wb'))
row = ['male', 'group_a', 'some high school', 'standard', 'none', 80,  80]
result = row_pred(row)
print(result)

