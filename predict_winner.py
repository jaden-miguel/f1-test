import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

fastf1.Cache.enable_cache('cache')


def load_data(year=2024):
    return pd.read_csv('data.csv')


def build_model():
    categorical = ['Abbreviation', 'TeamName']
    numeric = ['GridPosition']
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numeric)
    ])
    model = Pipeline([
        ('preprocess', pre),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    return model


def train_and_predict():
    df = load_data()
    df['Winner'] = (df['Position'] == 1).astype(int)
    last_round = df['Round'].max()

    train_df = df[df['Round'] < last_round]
    test_df = df[df['Round'] == last_round]

    features = [
        "Abbreviation",
        "TeamName",
        "GridPosition",
        "DriverNumber",
        "DriverPointsBefore",
        "TeamPointsBefore",
    ]

    X_train = train_df[features]
    y_train = train_df['Winner']

    model = build_model()
    model.fit(X_train, y_train)

    X_test = test_df[features]
    probs = model.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df['WinProbability'] = probs
    pred = test_df.sort_values('WinProbability', ascending=False).iloc[0]
    print("Predicted winner for round", last_round, "is", pred['Abbreviation'],
          "with probability", f"{pred['WinProbability']:.3f}")
    actual_winner = test_df[test_df['Winner'] == 1].iloc[0]
    print("Actual winner was", actual_winner['Abbreviation'])

    # compute accuracy on full dataset via train/test split
    X = df[features]
    y = df['Winner']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_model()
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print("Overall accuracy", f"{acc:.3f}")


if __name__ == '__main__':
    train_and_predict()
