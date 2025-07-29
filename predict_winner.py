import fastf1
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_data(years=(2022, 2023, 2024)):
    """Load race results for multiple seasons using FastF1 and compute
    cumulative points so the model can learn from season progress.

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
    df.sort_values(["Year", "Round"], inplace=True)
    df["DriverPointsBefore"] = (
        df.groupby("DriverNumber")["Points"].cumsum() - df["Points"]
    )
    df["TeamPointsBefore"] = (
        df.groupby("TeamName")["Points"].cumsum() - df["Points"]
    )
    df = df.dropna(subset=["GridPosition", "DriverNumber", "Position"])
    df.to_csv(csv_path, index=False)
    return df


def build_model():
    """Return a randomized-search random forest pipeline."""

    categorical = ["Abbreviation", "TeamName"]
    numeric = [
        "GridPosition",
        "DriverNumber",
        "DriverPointsBefore",
        "TeamPointsBefore",
    ]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    pipe = Pipeline(
        [
            ("preprocess", pre),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
        random_state=42,
    )
    return search


def train_and_predict():
    df = load_data()
    df["Winner"] = (df["Position"] == 1).astype(int)

    last_year = df["Year"].max()
    last_round = df[df["Year"] == last_year]["Round"].max()

    features = [
        "Abbreviation",
        "TeamName",
        "GridPosition",
        "DriverNumber",
        "DriverPointsBefore",
        "TeamPointsBefore",
    ]

    X = df[features]
    y = df["Winner"]

    search = build_model()
    search.fit(X, y)
    print("Best parameters:", search.best_params_)

    model = search.best_estimator_

    X_train = train_df[features]
    y_train = train_df["Winner"]

    search = build_model()
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)

    best_model = search.best_estimator_

    X_test = test_df[features]
    probs = best_model.predict_proba(X_test)[:, 1]


    probs = model.predict_proba(X_test)[:, 1]

    test_df = test_df.copy()
    test_df['WinProbability'] = probs
    pred = test_df.sort_values('WinProbability', ascending=False).iloc[0]
    print("Predicted winner for round", last_round, "is", pred['Abbreviation'],
          "with probability", f"{pred['WinProbability']:.3f}")
    actual_winner = test_df[test_df['Winner'] == 1].iloc[0]
    print("Actual winner was", actual_winner['Abbreviation'])

    # predict winner for the next scheduled race
    schedule = fastf1.get_event_schedule(last_year, include_testing=False)
    max_round = schedule["RoundNumber"].max()
    if last_round < max_round:
        next_year = last_year
        next_round = last_round + 1
    else:
        next_year = last_year + 1
        next_schedule = fastf1.get_event_schedule(
            next_year, include_testing=False
        )
        next_round = int(next_schedule["RoundNumber"].min())

    lineup = (
        df[df["Year"] == last_year]
        .groupby(["DriverNumber", "Abbreviation", "TeamName"])
        .tail(1)
        [["DriverNumber", "Abbreviation", "TeamName"]]
    )

    driver_totals = (
        df.groupby("DriverNumber")["Points"].sum()
    )
    team_totals = df.groupby("TeamName")["Points"].sum()

    lineup["DriverPointsBefore"] = lineup["DriverNumber"].map(driver_totals)
    lineup["TeamPointsBefore"] = lineup["TeamName"].map(team_totals)
    lineup["GridPosition"] = 0

    X_next = lineup[features]
    next_probs = model.predict_proba(X_next)[:, 1]
    lineup["WinProbability"] = next_probs
    pred_next = lineup.sort_values("WinProbability", ascending=False).iloc[0]
    print(
        "Predicted P1 for the next race (round",
        next_round,
        "in",
        next_year,
        ") is",
    next_probs = best_model.predict_proba(X_next)[:, 1]
    lineup["WinProbability"] = next_probs
    pred_next = lineup.sort_values("WinProbability", ascending=False).iloc[0]
    print(
        "Predicted winner for next round",
        next_round,
        "is",
        pred_next["Abbreviation"],
        "with probability",
        f"{pred_next['WinProbability']:.3f}",
    )

    # compute accuracy on full dataset via train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print("Overall accuracy", f"{acc:.3f}")

    # refit on all data before predicting
    model.fit(X, y)


    # compute accuracy on full dataset via train/test split
    X = df[features]
    y = df["Winner"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    best_model.fit(X_tr, y_tr)
    acc = best_model.score(X_te, y_te)


    y = df['Winner']
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_model()
    model.fit(X_tr, y_tr)


    print("Overall accuracy", f"{acc:.3f}")


if __name__ == '__main__':
    train_and_predict()
