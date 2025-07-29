# F1 Winner Predictor

This project downloads Formula 1 race results with the
[FastF1](https://github.com/theOehrly/Fast-F1) library and trains a random
forest model to estimate each driver's chance of winning the next race. The
training data includes cumulative driver and team points before each round.
`RandomizedSearchCV` selects the best hyperparameters for the forest so the
predictions are as accurate as possible.

## Setup

Install the dependencies with pip:

```bash
pip install -r requirements.txt
```

The first run will download timing data from the official F1 API and cache it in
`cache/`.

If `data.csv` is not present, race results from the past three seasons will also
be downloaded automatically to build the training dataset.

## Usage

1. Run `predict_winner.py` to fetch race results, train a model and output the
   driver most likely to win the next scheduled race (P1).

```bash
python predict_winner.py
```

The script also prints the overall accuracy of the model using a random
train/test split before predicting the winner of the next race based on the
latest standings.

## Data Source

Race and timing data is retrieved via `fastf1`, which accesses the official F1
live timing API. No deprecated Ergast data is used.
