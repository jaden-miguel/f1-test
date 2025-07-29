import fastf1
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


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
    df.sort_values(["Year", "Round"], inplace=True)
    df["DriverPointsBefore"] = (
        df.groupby(["Year", "DriverNumber"])["Points"]
        .cumsum()
        .shift(fill_value=0)
    )
    df["TeamPointsBefore"] = (
        df.groupby(["Year", "TeamName"])["Points"]
        .cumsum()
        .shift(fill_value=0)
    )
    df.to_csv(csv_path, index=False)
    return df


def build_model():
    """Return a cross-validated random forest model."""

    categorical = ["Abbreviation", "TeamName"]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )

    clf = RandomForestClassifier(random_state=42, class_weight="balanced")
    search = RandomizedSearchCV(
        clf,
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        n_iter=10,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocess", pre),
        ("classifier", search),
    ])
    return model


def train_and_predict():
    df = load_data()
    df["Winner"] = (df["Position"] == 1).astype(int)

    last_year = df["Year"].max()
    last_round = df[df["Year"] == last_year]["Round"].max()

    train_df = df[~((df["Year"] == last_year) & (df["Round"] == last_round))]
    test_df = df[(df["Year"] == last_year) & (df["Round"] == last_round)]

    X_train = train_df[
        [
            "Abbreviation",
            "TeamName",
            "GridPosition",
            "DriverNumber",
            "DriverPointsBefore",
            "TeamPointsBefore",
        ]
    ]
    y_train = train_df['Winner']

    model = build_model()
    model.fit(X_train, y_train)
    best_params = model.named_steps["classifier"].best_params_
    print("Best parameters", best_params)
    best_clf = RandomForestClassifier(
        random_state=42, class_weight="balanced", **best_params
    )
    model = Pipeline([
        ("preprocess", model.named_steps["preprocess"]),
        ("classifier", best_clf),
    ])
    model.fit(X_train, y_train)

    X_test = test_df[
        [
            "Abbreviation",
            "TeamName",
            "GridPosition",
            "DriverNumber",
            "DriverPointsBefore",
            "TeamPointsBefore",
        ]
    ]
    probs = model.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df['WinProbability'] = probs
    pred = test_df.sort_values('WinProbability', ascending=False).iloc[0]
    print("Predicted winner for round", last_round, "is", pred['Abbreviation'],
          "with probability", f"{pred['WinProbability']:.3f}")
    actual_winner = test_df[test_df['Winner'] == 1].iloc[0]
    print("Actual winner was", actual_winner['Abbreviation'])

    # compute accuracy on full dataset via train/test split
    X = df[
        [
            "Abbreviation",
            "TeamName",
            "GridPosition",
            "DriverNumber",
            "DriverPointsBefore",
            "TeamPointsBefore",
        ]
    ]
    y = df['Winner']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_model()
    model.fit(X_tr, y_tr)
    best_params = model.named_steps["classifier"].best_params_
    best_clf = RandomForestClassifier(
        random_state=42, class_weight="balanced", **best_params
    )
    final_model = Pipeline([
        ("preprocess", model.named_steps["preprocess"]),
        ("classifier", best_clf),
    ])
    final_model.fit(X_tr, y_tr)
    acc = final_model.score(X_te, y_te)
    print("Overall accuracy", f"{acc:.3f}")


if __name__ == '__main__':
    train_and_predict()
