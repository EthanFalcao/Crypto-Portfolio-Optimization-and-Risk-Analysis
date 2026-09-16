"""Run the full pipeline: fetch -> engineer features -> train/evaluate the LSTM."""
from src import data_fetch, features, model


def run():
    data_fetch.run()
    features.run()
    metrics = model.run()
    return metrics


if __name__ == "__main__":
    run()
