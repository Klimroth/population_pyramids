import warnings
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
from pandas.errors import SettingWithCopyWarning
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import test.sanity_check as sanity_check

import config
import src.io_data as io_data
import src.visualize as visualize
from src.generate_statistics import GenerateStatistics
from src.helper import ensure_diretory
from src.simple_logger import SimpleLogger

LOG: SimpleLogger = SimpleLogger()

SPECIES_WITH_POPULATION_DECREASE = []


def generate_sanity_check_one_species(species_name: str) -> None:

    if config.DATA_MODE == "PRE_SANITY_DATA":
        population_overview: Optional[pd.DataFrame] = io_data.read_species_information(
            src=config.BASE_DATA_FILE, species=species_name
        )
    else:
        population_overview = io_data.read_species_information(
            src=config.BASE_DATA_STORAGE, species=species_name
        )
    if population_overview is None:
        return None

    san_check: sanity_check.SanityChecker = sanity_check.SanityChecker()
    info_sheet = san_check.add_sanity_data_to_dataframe(
        population_overview, species_name, LOG
    )

    output_name: str = (
        f"{config.OUTPUT_FOLDER_DATA_CONSISTENCY}{species_name}_overview.xlsx"
    )
    ensure_diretory(output_name)
    info_sheet.to_excel(output_name, index=False)


def get_fecundity_threshold(species: str) -> Tuple[float, float, float, float]:
    output_name_fecundity: str = (
        f"{config.OUTPUT_FOLDER_STATISTICS}{species}_fecundity.csv"
    )
    try:
        df: pd.DataFrame = pd.read_csv(output_name_fecundity)
        fr_m: float = config.MINIMUM_AGE_AT_BIRTH_MALE[species] / 365.2425
        fr_f: float = config.MINIMUM_AGE_AT_BIRTH_FEMALE[species] / 365.2425
        max_age: int = max(df[df["fecundity probability"] > 0]["age"].values)
    except Exception as e:
        print(e)
        return

    lr_m: float = df[df["sex"] == "male"]["last reproduction threshold"].values[0]
    lr_f: float = df[df["sex"] == "female"]["last reproduction threshold"].values[0]

    return fr_m, lr_m, fr_f, lr_f


def generate_statistics_one_species(species: str) -> None:

    if not config.DATA_MODE == "AFTER_SANITY_DATA":
        population_overview: Optional[pd.DataFrame] = io_data.read_species_information(
            src=config.BASE_DATA_FILE, species=species
        )
    else:
        population_overview = io_data.read_species_information(
            src=config.BASE_DATA_STORAGE, species=species
        )

    if population_overview is None:
        return None

    output_population_overview: str = (
        f"{config.OUTPUT_FOLDER_TMP}{species}_population_overview.csv"
    )
    ensure_diretory(output_population_overview)
    population_overview.to_csv(output_population_overview, index=False)

    stat_gen: GenerateStatistics = GenerateStatistics(
        LOG, config.SPECIES_MINIMUM_YEAR, config.SPECIES_AGE_BOUNDARY
    )
    basic_stats, pyramid_stats, fecundity_stats = stat_gen.gather_statistics(
        population_overview, species
    )

    output_name_stats: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
    output_name_fecundity: str = (
        f"{config.OUTPUT_FOLDER_STATISTICS}{species}_fecundity.csv"
    )
    output_name_pyramids: str = (
        f"{config.OUTPUT_FOLDER_STATISTICS}{species}_pyramids.csv"
    )
    ensure_diretory(output_name_stats)
    ensure_diretory(output_name_fecundity)
    ensure_diretory(output_name_pyramids)

    basic_stats.to_csv(output_name_stats, index=False)
    fecundity_stats.to_csv(output_name_fecundity, index=False)

    if not config.SKIP_PYRAMID_STATISTIC_GENERATION:
        pyramid_stats.to_csv(output_name_pyramids, index=False)


def conduct_one_species(species):
    if not config.SKIP_DATA_SANITY_CHECK:
        generate_sanity_check_one_species(species_name=species)

    if not config.SKIP_STATISTIC_GENERATION:
        generate_statistics_one_species(species=species)

    visualize_stats = visualize.DrawStatistics(LOG)
    if not config.SKIP_AGE_DISTRIBUTION_PLOT:
        LOG.info("Start: visualize age distribution")
        visualize_stats.generate_image_fraction_age(species=species)

    if not config.SKIP_BIRTH_PLOT:
        LOG.info("Start: visualize annual births.")
        visualize_stats.generate_image_number_birth(species=species)

    if not config.SKIP_DEATH_PLOT:
        LOG.info("Start: visualize annual deaths.")
        visualize_stats.generate_image_number_death(species=species)

    if not config.SKIP_LOST_PLOT:
        LOG.info("Start: visualize annual lost tracks.")
        visualize_stats.generate_image_number_lost(species=species)

    if not config.SKIP_LOST_PLOT:
        LOG.info("Start: visualize annual population size.")
        visualize_stats.generate_image_number_alive(species=species)

    if not config.SKIP_MEDIAN_AGE_PLOT:
        LOG.info("Start: visualize median age of population.")
        visualize_stats.generate_image_median_age(species=species)

    if not config.SKIP_PROPORTION_NEWBORN_PLOT:
        LOG.info("Start: visualize sex proportion newborn.")
        visualize_stats.generate_image_proportion_newborn(species=species)

    if not config.SKIP_LITTER_SIZE_PLOT:
        LOG.info("Start: visualize litter size.")
        visualize_stats.generate_image_litter_size(species=species)

    if not config.SKIP_REPRODUCTIVE_AGE_PLOT:
        LOG.info("Start: visualize first reproduction age.")
        visualize_stats.generate_first_reproductive_image(species=species)

    visualize_pyramid = visualize.DrawPopulationPyramid(LOG)
    if not config.SKIP_PYRAMID_PLOT_ABSOLUTE:
        LOG.info("Start: visualize absolute population pyramid.")
        species_count_drops_below_threshold = visualize_pyramid.draw_pyramid(
            species=species, mode="absolute"
        )
        if species_count_drops_below_threshold:
            SPECIES_WITH_POPULATION_DECREASE.append(species)

    if not config.SKIP_PYRAMID_PLOT_RELATIVE:
        LOG.info("Start: visualize relative population pyramid.")
        visualize_pyramid.draw_pyramid(species=species, mode="relative")

    if not config.SKIP_FECUNDITY_PLOT:
        LOG.info("Start: visualize fecundity.")
        visualize_stats.draw_fecundity_plot(species=species)

    if not config.SKIP_PYRAMID_SHAPE_STATS:
        LOG.info("Start: pyramid shape information")
        visualize_pyramid.visualize_pyramid_classification(species=species)


def concat_problematic_parent_ids():
    dfs: List[pd.DataFrame] = []
    for species in config.EVALUATED_SPECIES:
        print(species)
        df_spec: pd.DataFrame = pd.read_excel(
            f"{config.OUTPUT_FOLDER_DATA_CONSISTENCY}{species}_overview.xlsx"
        )
        df = df_spec[df_spec["Problematic_Parent_ID"].notna()]
        dfs.append(df)
    pd.concat(dfs).to_excel(
        f"{config.OUTPUT_FOLDER_DATA_CONSISTENCY}00_problematic_parent_id_summary.xlsx",
        index=False,
    )


if __name__ == "__main__":
    now: datetime = datetime.now()
    config.initialise_config_variables()

    species_to_conduct = sorted(
        [
            spec
            for spec in config.EVALUATED_SPECIES
            if not spec in config.DISMISSED_SPECIES
        ]
    )
    j = 1
    exceptions: List[str] = []
    for species in species_to_conduct:
        print(f"*** {species} ({j} / {len(species_to_conduct)}) ***")
        try:
            conduct_one_species(species)
        except Exception as e:
            print(f"Species {species} has exception {e}.")
            exceptions.append(f"Species {species} has exception {e}.")
        j += 1
        # conduct_one_species(species)

    print(exceptions)

    if not config.SKIP_SUMMARISE_PROBLEMS:
        concat_problematic_parent_ids()
    LOG.dump_log(f"{config.OUTPUT_FOLDER_LOG}{now.strftime('%Y%m%d_%H%M%S')}_log.json")
    pd.DataFrame({"Species": SPECIES_WITH_POPULATION_DECREASE}).to_excel(
        f"{config.OUTPUT_FOLDER_PYRAMID_OVERVIEW}species_with_too_small_population.xlsx",
        index=False,
    )
