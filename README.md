# F1 Winner Predictor

This project downloads the latest Formula 1 race results using the
[FastF1](https://github.com/theOehrly/Fast-F1) library and trains a simple
logistic regression model to estimate the probability of each driver winning the
next race.

## Setup

Install the dependencies with pip:

```bash
pip install -r requirements.txt
```

The first run will download timing data from the official F1 API and cache it in
`cache/`.

## Usage

1. Run `predict_winner.py` to fetch race results, train a model and display the
   predicted winner for the most recent round.

```bash
python predict_winner.py
```

The script also prints the overall accuracy of the model using a random train
/test split.

## Data Source

Race and timing data is retrieved via `fastf1`, which accesses the official F1
live timing API. No deprecated Ergast data is used.
