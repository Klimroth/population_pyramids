from typing import Tuple

import pandas as pd
from tqdm import tqdm

import config
import src.io_data as io_data
import src.visualize as visualize
from src.generate_statistics import GenerateStatistics
from src.helper import ensure_diretory

OUTPUT_FILE_BIOLOGICAL_DATA = config.INPUT_FILE_BIOLOGICAL_DATA[:-4] + "_modified.xlsx"


def read_rosetta_stone() -> pd.DataFrame | None:
    try:
        df = pd.read_excel(config.INPUT_FILE_BIOLOGICAL_DATA)
    except:
        print("ERROR: file not found.")
        return None

    df["adult_threshold_female"] = 0
    df["adult_threshold_male"] = 0
    df["senior_threshold_female"] = 0
    df["senior_threshold_male"] = 0

    return df


def add_fecundity_values(df: pd.DataFrame) -> pd.DataFrame:

    def get_fecundity_thresholds(species: str) -> Tuple[float, float, float, float]:
        output_name_fecundity: str = (
            f"{config.OUTPUT_FOLDER_STATISTICS}{species}_fecundity.csv"
        )
        config.initialise_config_variables()

        try:
            df: pd.DataFrame = pd.read_csv(output_name_fecundity)
            fr_m: float = config.MINIMUM_AGE_AT_BIRTH_MALE[species] / 365.2425
            fr_f: float = config.MINIMUM_AGE_AT_BIRTH_FEMALE[species] / 365.2425
            max_age: int = max(df[df["fecundity probability"] > 0]["age"].values)
            df = df[df["age"] <= max_age]
            lr_m: float = df[df["sex"] == "male"]["last reproduction threshold"].values[
                0
            ]
            lr_f: float = df[df["sex"] == "female"][
                "last reproduction threshold"
            ].values[0]
        except Exception as e:
            print(
                "Fecundity has not been calculated yet, need to re-trigger pipeline",
                species,
            )
            fr_m = 0
            fr_f = 0
            lr_m = 999
            lr_f = 999

        return fr_f, fr_m, lr_f, lr_m

    for row in tqdm(df.index):
        species = df.loc[row, "Species"]
        adult_thresh_female, adult_thresh_male, sen_thresh_female, sen_thresh_male = (
            get_fecundity_thresholds(species)
        )
        df.loc[row, "adult_threshold_female"] = adult_thresh_female
        df.loc[row, "adult_threshold_male"] = adult_thresh_male
        df.loc[row, "senior_threshold_female"] = sen_thresh_female
        df.loc[row, "senior_threshold_male"] = sen_thresh_male

    return df


rosetta_stone = read_rosetta_stone()
rosetta_stone = add_fecundity_values(rosetta_stone)
rosetta_stone.to_excel(OUTPUT_FILE_BIOLOGICAL_DATA)
