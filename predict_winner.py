import fastf1
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV

fastf1.Cache.enable_cache('cache')


def load_data(years=(2022, 2023, 2024)):
    """Load race results for multiple seasons using FastF1.

    If a cached CSV file exists it will be reused to avoid excessive
    network access.
    """
    csv_path = Path("data.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    records = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for rnd in schedule["RoundNumber"]:
            try:
                session = fastf1.get_session(year, int(rnd), "R")
                session.load(laps=False, telemetry=False)
            except Exception as exc:  # pragma: no cover - network errors
                print(f"Could not load {year} round {rnd}: {exc}")
                continue

            res = session.results[
                [
                    "DriverNumber",
                    "Abbreviation",
                    "TeamName",
                    "GridPosition",
                    "Position",
                    "Points",
                ]
            ].copy()
            res["Year"] = year
            res["Round"] = int(rnd)
            records.append(res)

    df = pd.concat(records, ignore_index=True)
    df = df.dropna(subset=["GridPosition", "DriverNumber", "Position"])
    df.to_csv(csv_path, index=False)
    return df


def build_model():
    """Return a logistic regression model with basic preprocessing."""

    categorical = ["Abbreviation", "TeamName"]
    numeric = ["GridPosition", "DriverNumber"]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", pre),
            (
                "classifier",
                LogisticRegressionCV(
                    Cs=10,
                    cv=5,
                    max_iter=5000,
                    scoring="accuracy",
                    n_jobs=None,
                ),
            ),
        ]
    )
    return model


def train_and_predict():
    df = load_data()
    df["Winner"] = (df["Position"] == 1).astype(int)

    last_year = df["Year"].max()
    last_round = df[df["Year"] == last_year]["Round"].max()

    train_df = df[~((df["Year"] == last_year) & (df["Round"] == last_round))]
    test_df = df[(df["Year"] == last_year) & (df["Round"] == last_round)]


        "Abbreviation",
        "TeamName",
        "GridPosition",
        "DriverNumber",

    y_train = train_df['Winner']

    model = build_model()
    model.fit(X_train, y_train)


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
    acc = model.score(X_te, y_te)
    print("Overall accuracy", f"{acc:.3f}")


if __name__ == '__main__':
    train_and_predict()
