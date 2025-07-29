import fastf1
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

    return model


def train_and_predict():
    df = load_data()

    probs = model.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df['WinProbability'] = probs
    pred = test_df.sort_values('WinProbability', ascending=False).iloc[0]
    print("Predicted winner for round", last_round, "is", pred['Abbreviation'],
          "with probability", f"{pred['WinProbability']:.3f}")
    actual_winner = test_df[test_df['Winner'] == 1].iloc[0]
    print("Actual winner was", actual_winner['Abbreviation'])

    # compute accuracy on full dataset via train/test split
    y = df['Winner']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_model()
    model.fit(X_tr, y_tr)

    print("Overall accuracy", f"{acc:.3f}")


if __name__ == '__main__':
    train_and_predict()
