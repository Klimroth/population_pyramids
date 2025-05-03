from datetime import date, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd

import config
from src.generate_statistics import GenerateStatistics
from src.simple_logger import SimpleLogger


class SanityChecker:

    @staticmethod
    def add_sanity_data_to_dataframe(
        df: pd.DataFrame, species: str, logger: SimpleLogger
    ) -> pd.DataFrame:
        birth_age_dict_female, birth_age_dict_male = SanityChecker.get_reproduction_age(
            df, logger
        )
        sibling_info = SanityChecker.get_litter_by_dam(birth_age_dict_female)

        female_individual_first_last_birth = (
            SanityChecker.get_age_of_first_and_last_birth(birth_age_dict_female)
        )
        male_individual_first_last_birth = (
            SanityChecker.get_age_of_first_and_last_birth(birth_age_dict_male)
        )
        age_per_individual = SanityChecker.get_maximum_individual_age_in_days(df)
        birthday_mapping = SanityChecker.get_birthday_mapping(df)

        zooborn_animals: List[int] = df[df["birthType"] == "Captive Birth/Hatch"][
            "anonID"
        ].to_list()

        df["first_reproduction_threshold_female [d]"] = None
        df["first_reproduction_threshold_male [d]"] = None
        df["age_first_rep_male [y]"] = None
        df["age_first_rep_male [d]"] = None
        df["age_first_rep_female [y]"] = None
        df["age_first_rep_female [d]"] = None
        df["is_exceptionally_young_at_fertilization"] = None

        df["sire_age_at_birth [y]"] = None
        df["sire_age_at_birth [d]"] = None
        df["dam_age_at_birth [y]"] = None
        df["dam_age_at_birth [d]"] = None
        df["has_exceptionally_young_dam"] = None
        df["has_exceptionally_young_sire"] = None

        df["longevity_cutoff [y]"] = None
        df["longevity_cutoff [d]"] = None
        df["longevity_cutoff_male [y]"] = None
        df["longevity_cutoff_male [d]"] = None
        df["longevity_cutoff_female [y]"] = None
        df["longevity_cutoff_female [d]"] = None
        df["longevity [y]"] = None
        df["longevity [d]"] = None
        df["is_exceptionally_old"] = None
        df["is_zooborn_and_dead"] = None

        df["gestation_period [d]"] = None
        df["gestation_period_threshold [d]"] = None
        df["rank_of_this_birth"] = None
        df["prev_birth_of_dam"] = None
        df["CorrectedBirthDate"] = df["DateOfBirth"]
        df["next_birth_of_dam"] = None
        df["interbirth_interval_to_previous_birth_of_dam [d]"] = None
        df["is_exceptionally_short_interval"] = None
        df["is_last_birth_of_dam"] = None
        df["is_first_birth_of_dam"] = None

        df["average_litter_size"] = None
        df["current_litter_size"] = None
        df["litter_contained_individuals"] = None
        df["is_exceptionally_large_litter_size"] = None

        for row in df.index:
            anon_id = int(df.loc[row, "anonID"])
            sire_id = df.loc[row, "Sire_ID"]
            dam_id = df.loc[row, "Dam_ID"]
            age_of_first_reproduction = 0
            df.loc[row, "first_reproduction_threshold_female [d]"] = (
                config.MINIMUM_AGE_AT_BIRTH_FEMALE[species]
            )
            df.loc[row, "first_reproduction_threshold_male [d]"] = (
                config.MINIMUM_AGE_AT_BIRTH_MALE[species]
            )

            df.loc[row, "longevity [y]"] = round(
                age_per_individual[anon_id] / 365.2425, 2
            )
            df.loc[row, "longevity [d]"] = int(age_per_individual[anon_id])

            if df.loc[row, "Sex"] == "Male":
                df.loc[row, "age_first_rep_male [y]"] = (
                    round(
                        male_individual_first_last_birth[anon_id]["age_first_birth"]
                        / 365.2425,
                        2,
                    )
                    if anon_id in male_individual_first_last_birth
                    else None
                )
                df.loc[row, "age_first_rep_male [d]"] = (
                    int(male_individual_first_last_birth[anon_id]["age_first_birth"])
                    if anon_id in male_individual_first_last_birth
                    else None
                )
                age_of_first_reproduction = df.loc[row, "age_first_rep_male [d]"]
            else:
                df.loc[row, "age_first_rep_female [y]"] = (
                    round(
                        female_individual_first_last_birth[anon_id]["age_first_birth"]
                        / 365.2425,
                        2,
                    )
                    if anon_id in female_individual_first_last_birth
                    else None
                )
                df.loc[row, "age_first_rep_female [d]"] = (
                    int(female_individual_first_last_birth[anon_id]["age_first_birth"])
                    if anon_id in female_individual_first_last_birth
                    else None
                )
                age_of_first_reproduction = df.loc[row, "age_first_rep_female [d]"]

            if df.loc[row, "Sex"] == "Male":
                df.loc[row, "is_exceptionally_young_at_fertilization"] = (
                    (
                        age_of_first_reproduction
                        < df.loc[row, "first_reproduction_threshold_male [d]"]
                    )
                    if age_of_first_reproduction
                    else False
                )
            else:
                df.loc[row, "is_exceptionally_young_at_fertilization"] = (
                    (
                        age_of_first_reproduction
                        < df.loc[row, "first_reproduction_threshold_female [d]"]
                    )
                    if age_of_first_reproduction
                    else False
                )

            if sire_id is not None and sire_id in zooborn_animals:
                df.loc[row, "sire_age_at_birth [y]"] = round(
                    birth_age_dict_male[sire_id][anon_id] / 365.2425, 2
                )
                df.loc[row, "sire_age_at_birth [d]"] = int(
                    birth_age_dict_male[sire_id][anon_id]
                )
                df.loc[row, "has_exceptionally_young_sire"] = (
                    birth_age_dict_male[sire_id][anon_id]
                    - config.EXPECTED_BIRTH_INTERVAL[species]
                ) < df.loc[row, "first_reproduction_threshold_male [d]"]

            if dam_id is not None:
                if dam_id in zooborn_animals:
                    df.loc[row, "dam_age_at_birth [y]"] = round(
                        birth_age_dict_female[dam_id][anon_id] / 365.2425, 2
                    )
                    df.loc[row, "dam_age_at_birth [d]"] = int(
                        birth_age_dict_female[dam_id][anon_id]
                    )
                    df.loc[row, "has_exceptionally_young_dam"] = (
                        birth_age_dict_female[dam_id][anon_id]
                        - config.EXPECTED_BIRTH_INTERVAL[species]
                    ) < df.loc[row, "first_reproduction_threshold_female [d]"]

                interbirth_interval: int | None = (
                    int(
                        birth_age_dict_female[dam_id][anon_id]
                        - sibling_info[anon_id]["previous_birth_dam_age"]
                    )
                    if sibling_info[anon_id]["previous_birth_dam_age"]
                    else None
                )

                df.loc[row, "prev_birth_of_dam"] = (
                    birthday_mapping[anon_id]
                    - timedelta(
                        days=birth_age_dict_female[dam_id][anon_id]
                        - sibling_info[anon_id]["previous_birth_dam_age"]
                    )
                    if sibling_info[anon_id]["previous_birth_dam_age"]
                    and birth_age_dict_female[dam_id][anon_id]
                    else None
                )
                df.loc[row, "next_birth_of_dam"] = (
                    birthday_mapping[anon_id]
                    + timedelta(
                        days=sibling_info[anon_id]["next_birth_dam_age"]
                        - birth_age_dict_female[dam_id][anon_id]
                    )
                    if sibling_info[anon_id]["next_birth_dam_age"]
                    and birth_age_dict_female[dam_id][anon_id]
                    else None
                )
                df.loc[row, "interbirth_interval_to_previous_birth_of_dam [d]"] = (
                    interbirth_interval
                )
                df.loc[row, "is_exceptionally_short_interval"] = (
                    True
                    if interbirth_interval
                    and interbirth_interval
                    < config.MINIMUM_FRACTION_OF_BIRTH_INTERVAL
                    * config.EXPECTED_BIRTH_INTERVAL[species]
                    else False
                )
                df.loc[row, "is_last_birth_of_dam"] = sibling_info[anon_id][
                    "is_last_birth_of_dam"
                ]
                df.loc[row, "is_first_birth_of_dam"] = sibling_info[anon_id][
                    "is_first_birth_of_dam"
                ]
                df.loc[row, "current_litter_size"] = sibling_info[anon_id][
                    "litter_size"
                ]
                df.loc[row, "litter_contained_individuals"] = "; ".join(
                    [str(sibling_id) for sibling_id in sibling_info[anon_id]["litter"]]
                )
                df.loc[row, "is_exceptionally_large_litter_size"] = (
                    True
                    if sibling_info[anon_id]["litter_size"]
                    > config.EXPECTED_LITTER_SIZE[species]
                    else False
                )
                df.loc[row, "rank_of_this_birth"] = sibling_info[anon_id][
                    "rank_of_birth_by_dam"
                ]

            df.loc[row, "longevity_cutoff [y]"] = round(
                config.REPORT_INDIVIDUALS_OVER_AGE[species], 2
            )
            df.loc[row, "longevity_cutoff [d]"] = int(
                config.REPORT_INDIVIDUALS_OVER_AGE[species] * 365.2425
            )
            df.loc[row, "longevity_cutoff_male [y]"] = round(
                config.REPORT_INDIVIDUALS_OVER_AGE[species], 2
            )
            df.loc[row, "longevity_cutoff_male [d]"] = int(
                config.REPORT_INDIVIDUALS_OVER_AGE[species] * 365.2425
            )
            df.loc[row, "longevity_cutoff_female [y]"] = round(
                config.REPORT_INDIVIDUALS_OVER_AGE[species], 2
            )
            df.loc[row, "longevity_cutoff_female [d]"] = int(
                config.REPORT_INDIVIDUALS_OVER_AGE[species] * 365.2425
            )

            df.loc[row, "is_exceptionally_old"] = int(
                age_per_individual[anon_id]
            ) > int(config.REPORT_INDIVIDUALS_OVER_AGE[species] * 365.2425)

            df.loc[row, "is_zooborn_and_dead"] = (
                True
                if df.loc[row, "globStat"] == "Dead"
                and df.loc[row, "birthType"] == "Captive Birth/Hatch"
                else False
            )
            df.loc[row, "gestation_period [d]"] = int(
                config.EXPECTED_BIRTH_INTERVAL[species] * 365.2425
            )
            df.loc[row, "gestation_period_threshold [d]"] = int(
                config.EXPECTED_BIRTH_INTERVAL[species]
                * 365.2425
                * config.MINIMUM_FRACTION_OF_BIRTH_INTERVAL
            )
            df.loc[row, "average_litter_size"] = config.EXPECTED_LITTER_SIZE[species]

        return df

    @staticmethod
    def get_age_of_first_and_last_birth(
        individual_reproduction_age: Dict[int, Dict[int, float]]
    ) -> Dict[int, Dict[str, float]]:

        ret: Dict[int, Dict[str, float]] = {}
        for individual_id in individual_reproduction_age:
            ret[individual_id] = {
                "age_first_birth": min(
                    individual_reproduction_age[individual_id].values()
                ),
                "age_last_birth": max(
                    individual_reproduction_age[individual_id].values()
                ),
            }

        return ret

    @staticmethod
    def get_birthday_mapping(df: pd.DataFrame) -> Dict[int, date]:
        ret = {}
        for row in df.index:
            ret[df.loc[row, "anonID"]] = df.loc[row, "DateOfBirth"]
        return ret

    @staticmethod
    def get_maximum_individual_age_in_days(
        df: pd.DataFrame, reference_day: date = date(2024, 5, 1)
    ) -> Dict[int, float]:

        ret: Dict[int, float] = {}
        for row in df.index:
            if df.loc[row, "DateOfDeath"] == date(4000, 1, 1):
                max_age: int = (reference_day - df.loc[row, "DateOfBirth"]).days
            else:
                max_age = (df.loc[row, "DateOfDeath"] - df.loc[row, "DateOfBirth"]).days
            ret[int(df.loc[row, "anonID"])] = max_age

        return ret

    @staticmethod
    def get_litter_by_dam(
        individual_reproduction_age: Dict[int, Dict[int, float]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        called with the output of get_age_of_first_reproduction
        """
        sibling_info: Dict[int, Dict[str, Any]] = {}
        for dam_id in individual_reproduction_age:

            for child_id, dam_age in individual_reproduction_age[dam_id].items():
                litter: List[int] = [
                    sibling_id
                    for sibling_id in individual_reproduction_age[dam_id]
                    if individual_reproduction_age[dam_id][sibling_id] == dam_age
                ]
                first_birth_of_dam: bool = dam_age == min(
                    individual_reproduction_age[dam_id].values()
                )
                last_birth_of_dam: bool = dam_age == max(
                    individual_reproduction_age[dam_id].values()
                )

                dam_ages: List[float] = sorted(
                    list(set(list(individual_reproduction_age[dam_id].values())))
                )
                previous_dam_age = -1
                next_dam_age = -1
                previous_dam_age_index = -1
                next_dam_age_index = len(dam_ages)

                for j in range(len(dam_ages)):
                    if dam_ages[j] == dam_age:
                        previous_dam_age_index = j - 1
                        next_dam_age_index = j + 1
                if previous_dam_age_index >= 0:
                    previous_dam_age = dam_ages[previous_dam_age_index]
                if next_dam_age_index < len(dam_ages):
                    next_dam_age = dam_ages[next_dam_age_index]

                sibling_info[child_id] = {
                    "litter": litter,
                    "litter_size": len(litter),
                    "is_first_birth_of_dam": first_birth_of_dam,
                    "is_last_birth_of_dam": last_birth_of_dam,
                    "previous_birth_dam_age": (
                        previous_dam_age if not first_birth_of_dam else None
                    ),
                    "next_birth_dam_age": (
                        next_dam_age if not last_birth_of_dam else None
                    ),
                    "rank_of_birth_by_dam": previous_dam_age_index + 2,
                }

        return sibling_info

    @staticmethod
    def get_reproduction_age(
        df: pd.DataFrame, logger: SimpleLogger
    ) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        """
        :returns female and male reproduction ages parent_id: { newborn_id : age_of_parent }
        """
        statistic_generator: GenerateStatistics = GenerateStatistics(
            logger, config.SPECIES_MINIMUM_YEAR, config.SPECIES_AGE_BOUNDARY
        )
        year_range: Tuple[int, int] = statistic_generator.get_minimum_maximum_year(
            df, species="sanity_check"
        )

        birth_age_dict_female: Dict[int, Dict[int, float]] = {}
        birth_age_dict_male: Dict[int, Dict[int, float]] = {}
        for year in range(year_range[0], year_range[1] + 1):
            df_newborn: pd.DataFrame = df[df["YearOfBirth"] == year]
            for row in df_newborn.index:
                newborn_birthdate: date = df_newborn.loc[row, "DateOfBirth"]
                newborn_id: int = int(df_newborn.loc[row, "anonID"])
                dam_id: int = df_newborn.loc[row, "Dam_ID"]
                sire_id: int = df_newborn.loc[row, "Sire_ID"]

                if not dam_id is None:
                    if not len(df[df["anonID"] == dam_id]) == 1:
                        logger.error(
                            f"Unexpected error. dam_id {dam_id} not known but in data."
                        )
                    else:
                        dam_age: float = (
                            newborn_birthdate
                            - df[df["anonID"] == dam_id]["DateOfBirth"].to_list()[0]
                        ).days
                        if not dam_id in birth_age_dict_female:
                            birth_age_dict_female[dam_id] = {}

                        birth_age_dict_female[dam_id][newborn_id] = dam_age

                if not sire_id is None:
                    if not len(df[df["anonID"] == sire_id]) == 1:
                        logger.error(
                            f"Unexpected error. sire_id {sire_id} not known but in data."
                        )
                    else:
                        sire_age: float = (
                            newborn_birthdate
                            - df[df["anonID"] == sire_id]["DateOfBirth"].to_list()[0]
                        ).days
                        if not sire_id in birth_age_dict_male:
                            birth_age_dict_male[sire_id] = {}

                        birth_age_dict_male[sire_id][newborn_id] = sire_age

        return birth_age_dict_female, birth_age_dict_male
