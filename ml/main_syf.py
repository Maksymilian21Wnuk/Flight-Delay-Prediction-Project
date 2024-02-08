from EDA_plottings import run_plots
import pandas as pd







DATASET_PATH = "/home/meks/Desktop/data_cale.csv"


def main():
    dataset = pd.read_csv(DATASET_PATH)
    run_plots(dataset)


if __name__=="__main__":
    main()
