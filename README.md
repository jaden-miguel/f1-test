# F1 Winner Predictor


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


## Data Source

Race and timing data is retrieved via `fastf1`, which accesses the official F1
live timing API. No deprecated Ergast data is used.
