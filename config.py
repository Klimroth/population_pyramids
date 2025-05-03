import os
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

USE_REGIONS = ["Europe", "North America", "South America", "Africa", "Oceania", "Asia"]
USE_REGIONS_PYRAMIDS = ["North America"]  # has to be a subset of USE_REGIONS
REGION_STRING_PYRAMID_NAME = "-".join(USE_REGIONS_PYRAMIDS)


INPUT_FILE_BIOLOGICAL_DATA = ""

# novel data source
BASE_DATA_FILE = ""

# manually annotated data source
BASE_DATA_STORAGE = (
    ""
)

OUTPUT_FOLDER_DATA_CONSISTENCY = (
    ""
)
OUTPUT_FOLDER_TMP = (
    ""
)
OUTPUT_FOLDER_STATISTICS = f""
OUTPUT_FOLDER_FECUNDITY_INFORMATION = (
    ""
)
OUTPUT_FOLDER_VISUALISATION = f""
OUTPUT_FOLDER_PYRAMID_OVERVIEW = f""
OUTPUT_FOLDER_PYRAMID_TMP_STAT = (
    f""
)
OUTPUT_FOLDER_LOG = (
    ""
)

EVALUATED_SPECIES = [""]

DISMISSED_SPECIES = []

SKIP_DATA_SANITY_CHECK = 1
SKIP_SUMMARISE_PROBLEMS = 1

SKIP_STATISTIC_GENERATION = 0
ONLY_PYRAMID_STATISTIC_GENERATION = 0
SKIP_PYRAMID_STATISTIC_GENERATION = 0

SKIP_AGE_DISTRIBUTION_PLOT = 1
SKIP_BIRTH_PLOT = 1
SKIP_DEATH_PLOT = 1
SKIP_LOST_PLOT = 1
SKIP_ALIVE_PLOT = 1
SKIP_MEDIAN_AGE_PLOT = 1
SKIP_PROPORTION_NEWBORN_PLOT = 1
SKIP_LITTER_SIZE_PLOT = 1
SKIP_REPRODUCTIVE_AGE_PLOT = 1
SKIP_FECUNDITY_PLOT = 1

SKIP_PYRAMID_PLOT_ABSOLUTE = 0
SKIP_PYRAMID_SHAPE_STATS = 0

SKIP_PYRAMID_PLOT_RELATIVE = 1
SAVE_ALL_TMP_FILES = 0

DATA_MODE = (
    "AFTER_SANITY_DATA"  # "AFTER_SANITY_DATA" # AFTER_SANITY_DATA oder PRE_SANITY_DATA
)
SKIP_PYRAMIDS_BEFORE_YEAR = 1970

"""
*********************************************************
Relative population pyramid properties:
*********************************************************
"""
RELATIVE_PYRAMID_NUMBER_AGES = 21
PYRAMID_SEM_DISTINGUISH_FACTOR = 0.5

"""
*********************************************************
Visualisation properties:
- figure width and DPI
- Colors used in the figures
*********************************************************
"""

SAVE_ADDITIONALLY_AS_VECTORGRAPHIC = False
FIGURE_DPI = 1200
GRIDCOLOR = "grey"

WIDTH_AGE_DISTRIBUTION_IMAGE_MM = 90
AGE_DISTRIBUTION_COLOR_MAP = {
    "fraction_neonates": "rgb(214,234,248)",
    "fraction_subadult": "rgb(36,113,168)",
    "fraction_adult": "rgb(33,47,61)",
}

SEX_COLOR_MAP = {
    "male": "rgb(3,169,244)",
    "female": "rgb(216,27,96)",
    "undetermined": "rgb(128,139,150)",
    "all": "rgb(0,0,0)",
}

SEX_AGE_COLOR_MAP = {
    "male_juvenile": "rgb(135,206,235)",
    "male_adult": "rgb(30,144,255)",
    "male_senior": "rgb(0,0,205)",
    "female_juvenile": "rgb(255,69,0)",
    "female_adult": "rgb(205,0,0)",
    "female_senior": "rgb(139,0,0)",
}

WIDTH_BIRTH_IMAGE_MM = 90
WIDTH_DEATH_IMAGE_MM = 90
WIDTH_LOST_IMAGE_MM = 90
WIDTH_ALIVE_IMAGE_MM = 90
WIDTH_MEDIAN_AGE_IMAGE_MM = 90
WIDTH_PROPORTION_NEWBORN_IMAGE_MM = 90
WIDTH_PYRAMID_IMAGE_MM = 140


"""
*********************************************************
Basic properties:
- Name mapping of species names
- Names reported in figures
- Age boundary decision: neonate -> subadult -> adult
*********************************************************
"""

# species name in raw data : species name in cleaned data
SPECIES_NAME_MAPPER: Dict[str, str] = {}

# species name in cleaned data : name as shown on graphics
SPECIES_NAME_MAPPER_GRAPHICS: Dict[str, str] = {}

# species name in cleaned data: (age of becoming subadult, age of becoming adult)
SPECIES_AGE_BOUNDARY: Dict[str, Tuple[float, float]] = {}

# start year and end year for making statistics per species
# pipeline will use max(species_min, actual_min) - min(species_max, actual_max) as the year range
# thus we can write (1800, 2100) if everything should be included
SPECIES_MINIMUM_YEAR: Dict[str, Tuple[int, int]] = {}


"""
*********************************************************
Parallelization
*********************************************************
"""
MAX_WORKERS = 4

"""
*********************************************************
Data correction
- MAXIMUM_PLAUSIBLE_AGE in years
- EXPECTED_BIRTH_INTERVAL in years (e.g., 9 month = 0.75)
- MINIMUM_DAM_SIRE_CERTAINTY: ignore dam/sire relation if the certainty of parenthood is below this threshold
*********************************************************
"""
MAXIMUM_PLAUSIBLE_AGE: int = 50
REPORT_INDIVIDUALS_OVER_AGE: Dict[str, float] = {}
NUMBER_OLDEST_INDIVIDUALS: int = 25
NUMBER_YOUNGEST_INDIVIDUALS_AT_BIRTH: int = 25
EXPECTED_BIRTH_INTERVAL: Dict[str, float] = {}
MINIMUM_FRACTION_OF_BIRTH_INTERVAL: float = 1.0
EXPECTED_LITTER_SIZE: Dict[str, float] = {}
MINIMUM_DAM_SIRE_CERTAINTY = 0.0
DISMISS_BIRTH_ESTIMATION_CATEGORY = ["ApproxAfter", "ApproxBefore", "Undetermined"]
DISMISS_DEATH_ESTIMATION_CATEGORY = ["ApproxAfter", "ApproxBefore", "Undetermined"]
ROLLING_AVERAGE_WINDOW = 3
MINIMUM_AGE_AT_BIRTH_MALE: Dict[str, float] = {}
MINIMUM_AGE_AT_BIRTH_FEMALE: Dict[str, float] = {}

MAXIMUM_LONGEVITY_BY_SPECIES: Dict[str, Dict[str, float]] = {}
TAXONOMIC_GROUP_BY_SPECIES: Dict[str, float] = {}

THRESHOLD_AGE_PYRAMID: Dict[str, Dict[str, float]] = {}


def initialise_config_variables(input_file: str = INPUT_FILE_BIOLOGICAL_DATA):
    if not os.path.exists(input_file):
        print("ERROR: file not found ", input_file)
        return

    add_to_evaluation = False
    if len(EVALUATED_SPECIES) == 0:
        add_to_evaluation = True

    try:
        df = pd.read_csv(input_file)
    except:
        df = pd.read_excel(input_file)

    for row in df.index:
        species_name = df.loc[row, "Species"]
        gestation_period = float(
            df.loc[row, "Interbirth Interval  (d) (0.9*Gestation)"]
        )
        maximum_plausible_age = (
            max(
                float(df.loc[row, "Max Lifespan (Females)"]),
                float(df.loc[row, "Max Lifespan (Males)"]),
            )
            / 365.25
        )
        maximum_age_male = float(df.loc[row, "Max Lifespan (Males)"]) / 365.25
        maximum_age_female = float(df.loc[row, "Max Lifespan (Females)"]) / 365.25
        litter_size = float(df.loc[row, "Litter Size"])

        minimum_age_at_birth_male = float(
            df.loc[row, "Age of 1st Repro - at birth (Males)"]
        )  # days
        minimum_age_at_birth_female = float(
            df.loc[row, "Age of 1st Reproduction (Females)"]
        )  # days

        border_subadult_adult = (
            0.5 * (minimum_age_at_birth_male + minimum_age_at_birth_female) / 365.2425
        )

        MAXIMUM_LONGEVITY_BY_SPECIES[species_name] = {
            "all": maximum_plausible_age,
            "male": maximum_age_male,
            "female": maximum_age_female,
        }

        TAXONOMIC_GROUP_BY_SPECIES[species_name] = df.loc[row, "Group"]

        # todo get border_adult_senior
        if not species_name in MINIMUM_AGE_AT_BIRTH_MALE:
            MINIMUM_AGE_AT_BIRTH_MALE[species_name] = minimum_age_at_birth_male
        if not species_name in MINIMUM_AGE_AT_BIRTH_FEMALE:
            MINIMUM_AGE_AT_BIRTH_FEMALE[species_name] = minimum_age_at_birth_female
        if not species_name in REPORT_INDIVIDUALS_OVER_AGE:
            REPORT_INDIVIDUALS_OVER_AGE[species_name] = maximum_plausible_age
        if not species_name in EXPECTED_BIRTH_INTERVAL:
            EXPECTED_BIRTH_INTERVAL[species_name] = gestation_period
        if not species_name in SPECIES_NAME_MAPPER:
            SPECIES_NAME_MAPPER[species_name] = species_name
        if not species_name in SPECIES_NAME_MAPPER_GRAPHICS:
            SPECIES_NAME_MAPPER_GRAPHICS[species_name] = species_name
        if not species_name in SPECIES_AGE_BOUNDARY:
            SPECIES_AGE_BOUNDARY[species_name] = (1, border_subadult_adult)
        if not species_name in SPECIES_MINIMUM_YEAR:
            SPECIES_MINIMUM_YEAR[species_name] = (1800, 2100)
        if not species_name in EXPECTED_LITTER_SIZE:
            EXPECTED_LITTER_SIZE[species_name] = litter_size
        if add_to_evaluation:
            EVALUATED_SPECIES.append(species_name)

        if not species_name in THRESHOLD_AGE_PYRAMID:
            THRESHOLD_AGE_PYRAMID[species_name] = {
                "adult_threshold_female": df.loc[row, "adult_threshold_female"],
                "senior_threshold_female": df.loc[row, "senior_threshold_female"],
                "adult_threshold_male": df.loc[row, "adult_threshold_male"],
                "senior_threshold_male": df.loc[row, "senior_threshold_male"],
            }
