# F1 Winner Predictor

This project downloads Formula 1 race results with the
[FastF1](https://github.com/theOehrly/Fast-F1) library and trains a logistic
regression model to estimate each driver's chance of winning the next race. The
model is trained on data from the last few seasons and uses cross-validation to
improve accuracy.


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

1. Run `predict_winner.py` to fetch race results, train a model and display the
   predicted winner for the most recent round.

```bash
python predict_winner.py
```


## Data Source

Race and timing data is retrieved via `fastf1`, which accesses the official F1
live timing API. No deprecated Ergast data is used.
