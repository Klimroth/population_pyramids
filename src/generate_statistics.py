from datetime import date, datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gamma
from tqdm.contrib.concurrent import process_map

import config
from src.helper import ensure_diretory
from src.simple_logger import SimpleLogger


class GenerateStatistics:

    SPECIES_MINIMUM_YEAR = None
    SPECIES_AGE_BOUNDARY = None

    def __init__(
        self, logger: SimpleLogger, species_minimum_year, species_age_boundary
    ):
        self.logger: SimpleLogger = logger
        self.SPECIES_MINIMUM_YEAR = species_minimum_year
        self.SPECIES_AGE_BOUNDARY = species_age_boundary
        config.initialise_config_variables()

    def get_minimum_maximum_year(
        self, df: pd.DataFrame, species: str
    ) -> Tuple[int, int]:
        try:
            min_birth_year: float = min(
                [x for x in df["YearOfBirth"].to_list() if not x is None]
            )
        except:
            min_birth_year: float = 1900

        try:
            min_death_year: float = min(
                [x for x in df["YearOfDeath"].to_list() if not x is None]
            )
        except:
            min_death_year = 1900
        try:
            max_birth_year: float = max(
                [x for x in df["YearOfBirth"].to_list() if not x is None]
            )
        except:
            max_birth_year = date.today().year

        try:
            max_death_year: float = max(
                [
                    year
                    for year in df["YearOfDeath"].to_list()
                    if not year is None and year < 4000
                ]
            )
        except:
            max_death_year: float = date.today().year

        minimum_data_year = int(min(min_birth_year, min_death_year))
        maximum_data_year = int(max(max_birth_year, max_death_year))

        if species != "sanity_check":
            try:
                return (
                    max(minimum_data_year, self.SPECIES_MINIMUM_YEAR[species][0]),
                    min(maximum_data_year, self.SPECIES_MINIMUM_YEAR[species][1]),
                )
            except:
                self.logger.warning(
                    f"No year range given for {species}. All available years will be used."
                )

        return (minimum_data_year, maximum_data_year)

    def fit_gaussian_pdf(self, x: float, mu: float, sigma: float):
        f = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return f

    def gamma_pdf(self, x: float, alpha: float, theta: float):
        if theta <= 0:
            return np.inf
        return x ** (alpha - 1) * np.exp(-x / theta) / (gamma(alpha) * theta**alpha)

    def get_fecundity_threshold(
        self,
        df: pd.DataFrame,
        max_age: float,
        year_range: Tuple[int, int],
        sex: str,
        species_name: str,
    ) -> Tuple[
        List[int], np.array, float, float, float, float, int, pd.DataFrame | None
    ]:

        df = df[
            (df["birthType"] == "Captive Birth/Hatch")
            & (df["use_for_pyramids"] != "no")
            & (df["use_for_pyramids"] != "0")
            & (df["use_for_pyramids"] != 0)
        ]
        sex_col: str = "Dam_ID" if sex == "Female" else "Sire_ID"

        age_overview: Dict[int, Dict[str, int | float]] = {}
        for age in range(0, int(max_age) + 1):
            age_overview[age] = {
                "alive": 0,
                "alive_known_breeder": 0,
                "reproductive": 0,
                "gamma_fit": 0,
            }

        # get reproduction age
        # partly duplicate to sanity_check.py
        reproduction_dict: Dict[int, Set[int]] = {}
        for year in range(year_range[0], year_range[1] + 1):
            df_newborn: pd.DataFrame = df[df["YearOfBirth"] == year]
            for row in df_newborn.index:
                newborn_birthdate: date = df_newborn.loc[row, "DateOfBirth"]
                parent_id: int = df_newborn.loc[row, sex_col]
                if not parent_id is None:
                    if (
                        not len(df[df["anonID"] == parent_id]["DateOfBirth"].values)
                        == 0
                    ):
                        try:
                            parent_age: int = int(
                                np.round(
                                    (
                                        newborn_birthdate
                                        - df[df["anonID"] == parent_id][
                                            "DateOfBirth"
                                        ].values[0]
                                    ).days
                                    / 365.2425,
                                    0,
                                )
                            )
                        except:
                            continue
                        if not parent_id in reproduction_dict:
                            reproduction_dict[parent_id] = set([])
                        if parent_age < 0:
                            print(
                                f"SANITY ERROR: {df_newborn.loc[row, 'binSpecies']} - Individual {parent_id} has negative age ({df[df['anonID'] == parent_id]['DateOfBirth'].values[0]} {parent_age}) at birth of {df_newborn.loc[row, 'anonID']} ({newborn_birthdate})"
                            )
                            continue
                        reproduction_dict[parent_id].add(parent_age)

        df_known_breeders: pd.DataFrame = df[
            df["anonID"].isin(reproduction_dict.keys())
        ]
        known_breeder_ids = list(reproduction_dict.keys())
        df_sex = df[df["Sex"] == sex]
        for row in df_sex.index:
            individual_max_age: int = min(
                int(
                    (
                        min(df_sex.loc[row, "DateOfDeath"], date(2023, 12, 31))
                        - df_sex.loc[row, "DateOfBirth"]
                    ).days
                    / 365.2425
                ),
                int(max_age),
            )
            for age in range(individual_max_age + 1):
                age_overview[age]["alive"] += 1
                if df_sex.loc[row, "anonID"] in known_breeder_ids:
                    age_overview[age]["alive_known_breeder"] += 1

        for breeder in df_known_breeders["anonID"].values:
            if not breeder in reproduction_dict:
                continue
            for age in reproduction_dict[breeder]:
                if age > max_age:
                    print(
                        f"SANITY ERROR: {df_sex['binSpecies'].values[0]} - Individual {breeder} gives birth at age {age} but the species max age is {max_age}."
                    )
                else:
                    age_overview[age]["reproductive"] += 1

        for age in range(0, int(max_age) + 1):
            fecundity_probability: float = (
                age_overview[age]["reproductive"]
                / age_overview[age]["alive_known_breeder"]
                if age_overview[age]["alive_known_breeder"] > 3
                else 0
            )
            if np.isnan(fecundity_probability) or np.isinf(fecundity_probability):
                fecundity_probability = 0
            age_overview[age]["fecundity_probability"] = fecundity_probability

        ages = []
        probs = []

        for age in age_overview:
            ages.append(age)
            probs.append(age_overview[age]["fecundity_probability"])

        tmp_probs = []
        for j in range(len(probs)):
            if j == 0:
                tmp_probs.append(probs[j])
            elif j == 1:
                tmp_probs.append(0.5 * (probs[0] + probs[1]))
            else:
                tmp_probs.append(1 / 3 * (probs[j - 2] + probs[j - 1] + probs[j]))
        probs = np.array(tmp_probs)
        ages = ages[: len(probs)]
        ages[0] += 0.000001

        normalisation: float = np.sum(probs)

        if normalisation == 0:
            print(
                "ERROR: normalisation cannot be zero in fecundity probability array.",
                sex_col,
            )
            return ages, probs, 0, 0, 0, 0, 0, None

        normalised_probs: np.array = probs / normalisation
        loc = ages[ages.index(max(ages))]
        opt_params, _ = curve_fit(
            f=self.gamma_pdf,
            xdata=ages,
            ydata=normalised_probs,
            p0=[loc, 0.5],
            full_output=False,
        )

        alpha, theta = opt_params[0], opt_params[1]
        age_steps_fit = [y / 5000 for y in range(5000 * int(ages[-1]))]
        gamma_fitted_values = [self.gamma_pdf(x, alpha, theta) for x in age_steps_fit]

        max_gamma = max(gamma_fitted_values)
        max_index = gamma_fitted_values.index(max_gamma)

        ages[0] = int(ages[0])

        j = max_index
        while j < len(gamma_fitted_values) - 1:
            if gamma_fitted_values[j] < 0.75 * max_gamma:
                break
            j += 1

        threshold = age_steps_fit[j]

        for age in age_overview:
            used_age = age if age > 0 else 0.000001
            age_overview[age]["gamma_fit"] = normalisation * self.gamma_pdf(
                used_age, alpha, theta
            )

        df_fecundity_dict = {
            "age": [age for age in age_overview],
            "number_alive": [age_overview[age]["alive"] for age in age_overview],
            "number_breeders_alive": [
                age_overview[age]["alive_known_breeder"] for age in age_overview
            ],
            "number_reproductive": [
                age_overview[age]["reproductive"] for age in age_overview
            ],
            "fecundity_probability": [
                age_overview[age]["fecundity_probability"] for age in age_overview
            ],
            "fecundity_fit": [age_overview[age]["gamma_fit"] for age in age_overview],
        }

        df_fecundity = pd.DataFrame(df_fecundity_dict)
        df_fecundity["species"] = species_name
        df_fecundity["sex"] = sex
        df_fecundity["threshold_age"] = threshold
        df_fecundity["age_of_maximum_probability"] = age_steps_fit[max_index]

        return (
            ages,
            probs,
            alpha,
            theta,
            normalisation,
            threshold,
            len(df_known_breeders["anonID"].values),
            df_fecundity,
        )

    def get_first_reproduction_ages_year(
        self, df: pd.DataFrame, year: int, sex_col: str
    ) -> Dict[str, float]:
        df_year = df[(df["YearOfBirth"] == year) & (df[sex_col].notna())].copy()

        first_reproduction_ages: Dict[str, float] = {}
        for row in df_year.index:
            child_id = df_year.at[row, "anonID"]
            parent_id = df_year.at[row, sex_col]

            if not parent_id:
                continue

            child_current_age = df_year[df_year["anonID"] == child_id][
                "CurrentAge"
            ].values[0]

            # check whether dam already had a child
            df_all_born_children = df[
                (df[sex_col] == parent_id) & (df["CurrentAge"] > child_current_age)
            ]
            if len(df_all_born_children) >= 1:
                # is not the first ever born individual
                continue

            if len(df[df["anonID"] == parent_id]["CurrentAge"].values) > 0:
                parent_age = df[df["anonID"] == parent_id]["CurrentAge"].values[0]
                first_reproduction_ages[parent_id] = parent_age

        return first_reproduction_ages

    def get_last_reproduction_ages_year(
        self, df: pd.DataFrame, year: int, sex_col: str
    ) -> Dict[str, float]:
        df_year = df[(df["YearOfBirth"] == year) & (df[sex_col].notna())].copy()

        last_reproduction_ages: Dict[str, float] = {}
        for row in df_year.index:
            child_id = df_year.at[row, "anonID"]
            parent_id = df_year.at[row, sex_col]

            if not parent_id:
                continue

            child_current_age = df_year[df_year["anonID"] == child_id][
                "CurrentAge"
            ].values[0]

            # check whether dam has a later born child
            df_all_born_children = df[
                (df[sex_col] == parent_id) & (df["CurrentAge"] < child_current_age)
            ]
            if len(df_all_born_children) >= 1:
                # is not the last ever born individual
                continue

            parent_age = df[df["anonID"] == parent_id]["CurrentAge"].values[0]
            last_reproduction_ages[parent_id] = parent_age

        return last_reproduction_ages

    def get_average_age_first_female_reproduction(
        self, df: pd.DataFrame, year: int
    ) -> Optional[float]:

        first_reproduction_ages = self.get_first_reproduction_ages_year(
            df, year, "Dam_ID"
        )
        if len(first_reproduction_ages) == 0:
            return None
        return np.median(list(first_reproduction_ages.values()))

    def get_average_age_first_male_reproduction(
        self, df: pd.DataFrame, year: int
    ) -> Optional[float]:

        first_reproduction_ages = self.get_first_reproduction_ages_year(
            df, year, "Sire_ID"
        )
        if len(first_reproduction_ages) == 0:
            return None
        return np.median(list(first_reproduction_ages.values()))

    def get_average_age_last_female_reproduction(
        self, df: pd.DataFrame, year: int
    ) -> Optional[float]:

        last_reproduction_ages = self.get_last_reproduction_ages_year(
            df, year, "Dam_ID"
        )
        if len(last_reproduction_ages) == 0:
            return None
        return np.median(list(last_reproduction_ages.values()))

    def get_average_age_last_male_reproduction(
        self, df: pd.DataFrame, year: int
    ) -> Optional[float]:
        df_newborn = df[(df["YearOfBirth"] == year) & (df["Sire_ID"].notna())]
        df_newborn["Sire_Age"] = -1.0

        last_reproduction_ages = self.get_last_reproduction_ages_year(
            df, year, "Sire_ID"
        )
        if len(last_reproduction_ages) == 0:
            return None
        return np.median(list(last_reproduction_ages.values()))

    def get_litter_size(self, df: pd.DataFrame, year: int) -> Optional[float]:
        df2 = (
            df[(df["YearOfBirth"] == year) & (df["Dam_ID"] is not None)]
            .groupby(["Dam_ID", "DateOfBirth"])
            .count()
            .mean(axis=0, numeric_only=True)["anonID"]
        )
        if pd.isna(df2):
            return None
        else:
            return df2

    def get_number_breeders(
        self,
        df: pd.DataFrame,
        year: int,
        senior_thresh_m: float,
        senior_thresh_f: float,
    ) -> Tuple[int, int, int, int]:
        df_dams = df[(df["YearOfBirth"] == year) & (df["Dam_ID"] is not None)]
        df_sires = df[(df["YearOfBirth"] == year) & (df["Sire_ID"] is not None)]

        dams = set(df_dams["Dam_ID"].to_list())
        sires = set(df_sires["Sire_ID"].to_list())

        adult_dams, senior_dams = 0, 0
        adult_sires, senior_sires = 0, 0

        for dam in dams:
            ages = df[df["anonID"] == dam]["CurrentAge"].to_list()
            if len(ages) < 1:
                continue
            age = ages[0]
            if age <= senior_thresh_f:
                adult_dams += 1
            else:
                senior_dams += 1

        for sir in sires:
            ages = df[df["anonID"] == sir]["CurrentAge"].to_list()
            if len(ages) < 1:
                continue
            age = ages[0]
            if age <= senior_thresh_m:
                adult_sires += 1
            else:
                senior_sires += 1

        return adult_dams, senior_dams, adult_sires, senior_sires

    def get_current_age_row(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        df["CurrentAge"] = -1.0
        for row in df.index:
            df.loc[row, "CurrentAge"] = float(
                (date(year, 12, 31) - df.loc[row, "DateOfBirth"]).days / 365.2425
            )
        return df[df["CurrentAge"] >= 0]

    def get_current_rounded_age_row(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        df["CurrentFlooredAge"] = -1
        for row in df.index:
            curr_age: float = float(
                (date(year, 12, 31) - df.loc[row, "DateOfBirth"]).days / 365.2425
            )
            df.loc[row, "CurrentFlooredAge"] = int(curr_age) if curr_age >= 0 else -1

        return df[df["CurrentFlooredAge"] >= 0]

    def get_number_alive(self, df: pd.DataFrame, year: int, sex: List[str]) -> int:
        df_alive: pd.DataFrame = df[
            (df["YearOfDeath"] > year)
            & (df["Sex"].isin(sex))
            & (df["YearOfBirth"] <= year)
        ]
        return len(df_alive)

    def get_number_deaths(self, df: pd.DataFrame, year: int, sex: List[str]) -> int:
        df_death_cases: pd.DataFrame = df[
            (df["YearOfDeath"] == year)
            & (df["Sex"].isin(sex))
            & (df["globStat"] != "Unknown")
        ]
        return len(df_death_cases)

    def get_number_lost_individuals(
        self, df: pd.DataFrame, year: int, sex: List[str]
    ) -> int:
        df_lost_cases: pd.DataFrame = df[
            (df["YearOfDeath"] == year)
            & (df["Sex"].isin(sex))
            & (df["globStat"] == "Unknown")
        ]
        return len(df_lost_cases)

    def get_number_births(self, df: pd.DataFrame, year: int, sex: List[str]) -> int:
        df_birth_cases: pd.DataFrame = df[
            (df["YearOfBirth"] == year) & (df["Sex"].isin(sex))
        ]
        return len(df_birth_cases)

    def get_fraction_neonates(
        self, df: pd.DataFrame, year: int, sex: List[str]
    ) -> float:
        current_species_name: str = df["binSpecies"].to_list()[0]
        df_newborn: pd.DataFrame = df[
            (df["CurrentAge"] < self.SPECIES_AGE_BOUNDARY[current_species_name][0])
            & (df["CurrentAge"] >= 0)
            & (df["Sex"].isin(sex))
            & (df["YearOfDeath"] > year)
        ]
        num_newborn: int = len(df_newborn)
        num_alive: int = self.get_number_alive(df, year, sex)

        if num_alive == 0:
            return 0.0

        return num_newborn / num_alive

    def get_fraction_subadult(
        self, df: pd.DataFrame, year: int, sex: List[str]
    ) -> float:
        current_species_name: str = df["binSpecies"].to_list()[0]
        df_subadult: pd.DataFrame = df[
            (df["CurrentAge"] >= self.SPECIES_AGE_BOUNDARY[current_species_name][0])
            & (df["CurrentAge"] < self.SPECIES_AGE_BOUNDARY[current_species_name][1])
            & (df["YearOfDeath"] > year)
            & (df["Sex"].isin(sex))
        ]
        num_subadult: int = len(df_subadult)
        num_alive: int = self.get_number_alive(df, year, sex)

        if num_alive == 0:
            return 0.0

        return num_subadult / num_alive

    def get_fraction_adult(self, df: pd.DataFrame, year: int, sex: List[str]) -> float:
        current_species_name: str = df["binSpecies"].to_list()[0]
        df_adult: pd.DataFrame = df[
            (df["CurrentAge"] >= self.SPECIES_AGE_BOUNDARY[current_species_name][1])
            & (df["Sex"].isin(sex))
            & (df["YearOfDeath"] > year)
        ]
        num_adult: int = len(df_adult)
        num_alive: int = self.get_number_alive(df, year, sex)

        if num_alive == 0:
            return 0.0

        return num_adult / num_alive

    def get_median_population_age(
        self, df: pd.DataFrame, year: int, sex: List[str]
    ) -> float:
        df_alive: pd.DataFrame = df[
            (df["YearOfBirth"] <= year)
            & (df["Sex"].isin(sex))
            & (df["YearOfDeath"] > year)
        ]
        return df_alive.median(axis=0, numeric_only=True)["CurrentAge"]

    def get_maximum_population_age(
        self, df: pd.DataFrame, year: int, sex: List[str]
    ) -> float:
        df_alive: pd.DataFrame = df[
            (df["YearOfBirth"] <= year)
            & (df["Sex"].isin(sex))
            & (df["YearOfDeath"] > year)
        ]
        if len(df_alive) == 0:
            return 0
        return df_alive.max(axis=0, numeric_only=True)["CurrentAge"]

    def get_sex_proportion_newborn(
        self, df: pd.DataFrame, year: int
    ) -> Tuple[float, float, float]:
        df_birth_cases: pd.DataFrame = df[df["YearOfBirth"] == year]
        all_cases: int = len(df_birth_cases)
        if all_cases == 0:
            return 0.0, 0.0, 0.0

        male_cases: int = len(df_birth_cases[df_birth_cases["Sex"] == "Male"])
        female_cases: int = len(df_birth_cases[df_birth_cases["Sex"] == "Female"])

        return (
            male_cases / all_cases,
            female_cases / all_cases,
            1 - (male_cases + female_cases) / all_cases,
        )

    def gather_stats_one_year(self, args: Tuple[pd.DataFrame, int]) -> pd.DataFrame:

        def append_row(
            ret: Dict[str, List[Any]], cur_year, quantity, sex, value
        ) -> None:
            ret["Year"].append(cur_year)
            ret["Quantity"].append(quantity)
            ret["Sex"].append(sex)
            ret["Value"].append(value)

        df: pd.DataFrame = args[0]
        year: int = args[1]

        self.logger.info(f"Gathering statistics for year {year}.")

        df_used: pd.DataFrame = self.get_current_age_row(df.copy(), year)

        if config.SAVE_ALL_TMP_FILES:
            species = df_used["binSpecies"].to_list()[0]
            output_population_overview: str = (
                f"{config.OUTPUT_FOLDER_TMP}{species}/{species}_population_overview_exact_age_{year}.csv"
            )
            ensure_diretory(output_population_overview)
            df_used.to_csv(output_population_overview, index=False)

        ret: Dict[str, List[Any]] = {
            "Year": [],
            "Quantity": [],
            "Sex": [],
            "Value": [],
        }

        frac_neonates_male: float = self.get_fraction_neonates(df_used, year, ["Male"])
        frac_neonates_female: float = self.get_fraction_neonates(
            df_used, year, ["Female"]
        )
        frac_neonates_undetermined: float = self.get_fraction_neonates(
            df_used, year, ["Undet"]
        )
        frac_neonates_population: float = self.get_fraction_neonates(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "fraction_neonates", "male", frac_neonates_male)
        append_row(ret, year, "fraction_neonates", "female", frac_neonates_female)
        append_row(
            ret, year, "fraction_neonates", "undetermined", frac_neonates_undetermined
        )
        append_row(ret, year, "fraction_neonates", "all", frac_neonates_population)

        frac_subadults_male: float = self.get_fraction_subadult(df_used, year, ["Male"])
        frac_subadults_female: float = self.get_fraction_subadult(
            df_used, year, ["Female"]
        )
        frac_subadults_undetermined: float = self.get_fraction_subadult(
            df_used, year, ["Undet"]
        )
        frac_subadults_population: float = self.get_fraction_subadult(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "fraction_subadult", "male", frac_subadults_male)
        append_row(ret, year, "fraction_subadult", "female", frac_subadults_female)
        append_row(
            ret, year, "fraction_subadult", "undetermined", frac_subadults_undetermined
        )
        append_row(ret, year, "fraction_subadult", "all", frac_subadults_population)

        frac_adults_male: float = self.get_fraction_adult(df_used, year, ["Male"])
        frac_adults_female: float = self.get_fraction_adult(df_used, year, ["Female"])
        frac_adults_undetermined: float = self.get_fraction_adult(
            df_used, year, ["Undet"]
        )
        frac_adults_population: float = self.get_fraction_adult(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "fraction_adult", "male", frac_adults_male)
        append_row(ret, year, "fraction_adult", "female", frac_adults_female)
        append_row(
            ret, year, "fraction_adult", "undetermined", frac_adults_undetermined
        )
        append_row(ret, year, "fraction_adult", "all", frac_adults_population)

        number_birth_male: int = self.get_number_births(df_used, year, ["Male"])
        number_birth_female: int = self.get_number_births(df_used, year, ["Female"])
        number_birth_undetermined: int = self.get_number_births(
            df_used, year, ["Undet"]
        )
        number_birth_population: int = self.get_number_births(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "number_births", "male", number_birth_male)
        append_row(ret, year, "number_births", "female", number_birth_female)
        append_row(
            ret, year, "number_births", "undetermined", number_birth_undetermined
        )
        append_row(ret, year, "number_births", "all", number_birth_population)

        number_death_male: int = self.get_number_deaths(df_used, year, ["Male"])
        number_death_female: int = self.get_number_deaths(df_used, year, ["Female"])
        number_death_undetermined: int = self.get_number_deaths(
            df_used, year, ["Undet"]
        )
        number_death_population: int = self.get_number_deaths(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "number_deaths", "male", number_death_male)
        append_row(ret, year, "number_deaths", "female", number_death_female)
        append_row(
            ret, year, "number_deaths", "undetermined", number_death_undetermined
        )
        append_row(ret, year, "number_deaths", "all", number_death_population)

        number_lost_male: int = self.get_number_lost_individuals(
            df_used, year, ["Male"]
        )
        number_lost_female: int = self.get_number_lost_individuals(
            df_used, year, ["Female"]
        )
        number_lost_undetermined: int = self.get_number_lost_individuals(
            df_used, year, ["Undet"]
        )
        number_lost_population: int = self.get_number_lost_individuals(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "number_lost", "male", number_lost_male)
        append_row(ret, year, "number_lost", "female", number_lost_female)
        append_row(ret, year, "number_lost", "undetermined", number_lost_undetermined)
        append_row(ret, year, "number_lost", "all", number_lost_population)

        number_alive_male: int = self.get_number_alive(df_used, year, ["Male"])
        number_alive_female: int = self.get_number_alive(df_used, year, ["Female"])
        number_alive_undetermined: int = self.get_number_alive(df_used, year, ["Undet"])
        number_alive_population: int = self.get_number_alive(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "number_alive", "male", number_alive_male)
        append_row(ret, year, "number_alive", "female", number_alive_female)
        append_row(ret, year, "number_alive", "undetermined", number_alive_undetermined)
        append_row(ret, year, "number_alive", "all", number_alive_population)

        median_age_male: float = self.get_median_population_age(df_used, year, ["Male"])
        median_age_female: float = self.get_median_population_age(
            df_used, year, ["Female"]
        )
        median_age_undetermined: float = self.get_median_population_age(
            df_used, year, ["Undet"]
        )
        median_age_population: float = self.get_median_population_age(
            df_used, year, ["Male", "Female", "Undet"]
        )

        append_row(ret, year, "median_age", "male", median_age_male)
        append_row(ret, year, "median_age", "female", median_age_female)
        append_row(ret, year, "median_age", "undetermined", median_age_undetermined)
        append_row(ret, year, "median_age", "all", median_age_population)

        maximum_age_population: float = self.get_maximum_population_age(
            df_used, year, ["Male", "Female", "Undet"]
        )
        append_row(ret, year, "maximum_age", "all", maximum_age_population)

        frac_male_newborn, frac_female_newborn, frac_undet_newborn = (
            self.get_sex_proportion_newborn(df_used, year)
        )

        append_row(ret, year, "proportion_newborn", "male", frac_male_newborn)
        append_row(ret, year, "proportion_newborn", "female", frac_female_newborn)
        append_row(ret, year, "proportion_newborn", "undetermined", frac_undet_newborn)
        append_row(ret, year, "proportion_newborn", "all", 1.0)

        average_litter_size = self.get_litter_size(df_used, year)
        append_row(ret, year, "average_litter_size", "all", average_litter_size)

        mean_ffr = self.get_average_age_first_female_reproduction(df_used, year)
        mean_fmr = self.get_average_age_first_male_reproduction(df_used, year)
        append_row(ret, year, "mean_age_first_reproduction", "female", mean_ffr)
        append_row(ret, year, "mean_age_first_reproduction", "male", mean_fmr)

        return pd.DataFrame(ret)

    def get_pyramid_information(
        self, df: pd.DataFrame, year: int, sex: List[str], max_age: float
    ) -> Dict[int, int]:
        df_alive: pd.DataFrame = df[
            (df["YearOfDeath"] > year)
            & (df["Sex"].isin(sex))
            & (df["YearOfBirth"] <= year)
        ]

        num_all_individuals: int = len(df_alive)
        if num_all_individuals == 0:
            return {a: 0 for a in range(int(max_age) + 1)}

        ret: Dict[int, int] = {}
        for age in range(int(max_age) + 1):
            individuals_of_age: int = len(
                df_alive[df_alive["CurrentFlooredAge"] == age]
            )
            ret[age] = individuals_of_age

        return ret

    def get_scaled_pyramid_information(
        self, df: pd.DataFrame, year: int, sex: List[str], max_age: float
    ) -> Dict[float, float]:
        def get_bucket_number(boundaries: List[float], value: float):
            j = 0
            for boundary in boundaries:
                if value <= boundary:
                    return j
                j += 1
            return j

        age_boundaries: List[float] = [
            round(j / config.RELATIVE_PYRAMID_NUMBER_AGES, 4)
            for j in range(1, config.RELATIVE_PYRAMID_NUMBER_AGES + 1)
        ]
        number_all_individuals: int = len(
            df[(df["YearOfDeath"] > year) & (df["YearOfBirth"] <= year)]
        )
        df_alive: pd.DataFrame = df[
            (df["YearOfDeath"] > year)
            & (df["Sex"].isin(sex))
            & (df["YearOfBirth"] <= year)
        ]

        num_all_individuals: int = len(df_alive)
        if num_all_individuals == 0:
            return {}

        ret: Dict[float, float] = {}
        bucket_numbers = [j for j in range(len(age_boundaries))]
        for age_bucket in bucket_numbers:
            ret[
                age_bucket / config.RELATIVE_PYRAMID_NUMBER_AGES
                + 1 / (2 * config.RELATIVE_PYRAMID_NUMBER_AGES)
            ] = 0

        for row in df_alive.index:
            scaled_age = max(0, min(df_alive.loc[row, "CurrentAge"] / max_age, 1))
            age_bucket = get_bucket_number(age_boundaries, scaled_age)
            scaled_age_bucket = age_bucket / config.RELATIVE_PYRAMID_NUMBER_AGES + 1 / (
                2 * config.RELATIVE_PYRAMID_NUMBER_AGES
            )
            ret[scaled_age_bucket] += 1 / number_all_individuals

        return ret

    def gather_pyramid_values_one_year(
        self,
        args: Tuple[pd.DataFrame, int, float, str, List, List],
    ) -> pd.DataFrame:

        def append_row(
            result,
            cur_year: int,
            cur_type: str,
            age_val: int | float,
            sex: str,
            ind: int | float,
            median_age: float,
            mean_age: float,
            proven_breeders: float,
            num_births: int,
            num_proven_breeders: int,
            num_birth_givers_adult: int,
            num_birth_givers_senior: int,
            num_proven_breeders_adult: int,
            num_proven_breeders_senior: int,
            proven_breeders_adult: float,
            proven_breeders_senior: float,
            prop_birth_givers: float,
            prop_birth_givers_adult: float,
            prop_birth_givers_senior: float,
            median_age_all_sex: float,
            num_proven_breeders_adultSenior: int,
            proven_breeders_adultSenior: float,
            num_proven_breeders_all_sex: float,
            num_proven_breeders_adult_all_sex: float,
            num_proven_breeders_senior_all_sex: float,
            num_proven_breeders_adultSenior_all_sex: float,
            proven_breeders_all_sex: float,
            proven_breeders_adult_all_sex: float,
            proven_breeders_senior_all_sex: float,
            proven_breeders_adultSenior_all_sex: float,
            num_juvenile: int,
            num_adult: int,
            num_senior: int,
        ):
            result["Year"].append(cur_year)
            result["Type"].append(cur_type)
            result["Age"].append(age_val)
            result["Sex"].append(sex)
            result["Individuals"].append(ind)

            result["median age"].append(median_age)
            result["median age all sex"].append(median_age_all_sex)
            result["mean age"].append(mean_age)

            result["num proven breeders"].append(num_proven_breeders)
            result["num proven breeders adult"].append(num_proven_breeders_adult)
            result["num proven breeders senior"].append(num_proven_breeders_senior)
            result["num proven breeders adultSenior"].append(
                num_proven_breeders_adultSenior
            )

            result["num proven breeders all sex"].append(num_proven_breeders_all_sex)
            result["num proven breeders adult all sex"].append(
                num_proven_breeders_adult_all_sex
            )
            result["num proven breeders senior all sex"].append(
                num_proven_breeders_senior_all_sex
            )
            result["num proven breeders adultSenior all sex"].append(
                num_proven_breeders_adultSenior_all_sex
            )

            result["proven breeders"].append(proven_breeders)
            result["proven breeders adult"].append(proven_breeders_adult)
            result["proven breeders senior"].append(proven_breeders_senior)
            result["proven breeders adultSenior"].append(proven_breeders_adultSenior)
            result["proven breeders all sex"].append(proven_breeders_all_sex)
            result["proven breeders adult all sex"].append(
                proven_breeders_adult_all_sex
            )
            result["proven breeders senior all sex"].append(
                proven_breeders_senior_all_sex
            )
            result["proven breeders adultSenior all sex"].append(
                proven_breeders_adultSenior_all_sex
            )

            result["number of births"].append(num_births)
            result["number of birth-givers"].append(
                num_birth_givers_adult + num_birth_givers_senior
            )
            result["number of birth-givers adult"].append(num_birth_givers_adult)
            result["number of birth-givers senior"].append(num_birth_givers_senior)
            result["proportion birth-givers"].append(prop_birth_givers)
            result["proportion birth-givers adult"].append(prop_birth_givers_adult)
            result["proportion birth-givers senior"].append(prop_birth_givers_senior)

            result["num juvenile"].append(num_juvenile)
            result["num adult"].append(num_adult)
            result["num senior"].append(num_senior)

        df: pd.DataFrame = args[0]
        year: int = args[1]
        max_age: float = args[2]
        species: str = args[3]
        proven_breeders_female_all: List = args[4]
        proven_breeders_male_all: List = args[5]

        df = df[(df["YearOfDeath"] > year) & (df["YearOfBirth"] <= year)]

        self.logger.info(f"Gathering pyramid statistics for year {year}.")
        df_used: pd.DataFrame = self.get_current_rounded_age_row(df.copy(), year)
        df_used = self.get_current_age_row(df_used.copy(), year)

        if config.SAVE_ALL_TMP_FILES:
            species = df_used["binSpecies"].to_list()[0]
            output_population_overview: str = (
                f"{config.OUTPUT_FOLDER_TMP}{species}/{species}_population_overview_rounded_age_{year}.csv"
            )
            ensure_diretory(output_population_overview)
            df_used.to_csv(output_population_overview, index=False)

        ret: Dict[str, List[Any]] = {
            "Year": [],
            "Type": [],
            "Age": [],
            "Sex": [],
            "Individuals": [],
            "median age": [],
            "median age all sex": [],
            "mean age": [],
            "num proven breeders": [],
            "num proven breeders adult": [],
            "num proven breeders senior": [],
            "num proven breeders adultSenior": [],
            "num proven breeders all sex": [],
            "num proven breeders adult all sex": [],
            "num proven breeders senior all sex": [],
            "num proven breeders adultSenior all sex": [],
            "proven breeders": [],
            "proven breeders adult": [],
            "proven breeders senior": [],
            "proven breeders adultSenior": [],
            "proven breeders all sex": [],
            "proven breeders adult all sex": [],
            "proven breeders senior all sex": [],
            "proven breeders adultSenior all sex": [],
            "number of births": [],
            "number of birth-givers": [],
            "number of birth-givers adult": [],
            "number of birth-givers senior": [],
            "proportion birth-givers": [],
            "proportion birth-givers adult": [],
            "proportion birth-givers senior": [],
            "num juvenile": [],
            "num adult": [],
            "num senior": [],
        }

        age_quant_m = self.get_pyramid_information(df_used, year, ["Male"], max_age)
        age_quant_f = self.get_pyramid_information(df_used, year, ["Female"], max_age)

        num_birth_male = self.get_number_births(df_used, year, ["Male"])
        num_birth_female = self.get_number_births(df_used, year, ["Female"])

        mean_age_male = (
            np.mean(df_used[df_used["Sex"] == "Male"]["CurrentAge"].to_list())
            if len(df_used[df_used["Sex"] == "Male"]["CurrentAge"])
            else None
        )
        mean_age_female = (
            np.mean(df_used[df_used["Sex"] == "Female"]["CurrentAge"].to_list())
            if len(df_used[df_used["Sex"] == "Female"]["CurrentAge"])
            else None
        )

        median_age_male = (
            np.median(df_used[df_used["Sex"] == "Male"]["CurrentAge"].to_list())
            if len(df_used[df_used["Sex"] == "Male"]["CurrentAge"])
            else None
        )
        median_age_female = (
            np.median(df_used[df_used["Sex"] == "Female"]["CurrentAge"].to_list())
            if len(df_used[df_used["Sex"] == "Female"]["CurrentAge"])
            else None
        )

        median_age = (
            np.median(df_used["CurrentAge"].to_list())
            if len(df_used["CurrentAge"])
            else None
        )
        mean_age = (
            np.median(df_used["CurrentAge"].to_list())
            if len(df_used["CurrentAge"])
            else None
        )
        config.initialise_config_variables()

        fr_f = config.THRESHOLD_AGE_PYRAMID[species]["adult_threshold_female"]
        fr_m = config.THRESHOLD_AGE_PYRAMID[species]["adult_threshold_male"]
        lr_f = config.THRESHOLD_AGE_PYRAMID[species]["senior_threshold_female"]
        lr_m = config.THRESHOLD_AGE_PYRAMID[species]["senior_threshold_male"]

        if (fr_f, fr_m, lr_f, lr_m) == (0, 0, 999, 999):
            print(
                "Fecundity has not been calculated yet, need to re-trigger pipeline",
                species,
            )

        # proven breeders
        num_proven_breeders_adult_female: int = len(
            df_used[
                (df_used["anonID"].isin(proven_breeders_female_all))
                & (df_used["CurrentAge"] <= lr_f)
                & (df_used["CurrentAge"] >= fr_f)
            ]
        )
        num_proven_breeders_senior_female: int = len(
            df_used[
                (df_used["anonID"].isin(proven_breeders_female_all))
                & (df_used["CurrentAge"] > lr_f)
            ]
        )
        num_proven_breeders_female = (
            num_proven_breeders_adult_female + num_proven_breeders_senior_female
        )

        num_adult_female = len(
            df_used[
                (df_used["CurrentAge"] >= fr_f)
                & (df_used["CurrentAge"] <= lr_f)
                & (df_used["Sex"] == "Female")
            ]
        )
        num_senior_female = len(
            df_used[(df_used["CurrentAge"] > lr_f) & (df_used["Sex"] == "Female")]
        )
        num_female = len(df_used[df_used["Sex"] == "Female"])
        num_juvenile_female = max(num_female - num_adult_female - num_senior_female, 0)

        num_proven_breeders_adult_male: int = len(
            df_used[
                (df_used["anonID"].isin(proven_breeders_male_all))
                & (df_used["CurrentAge"] <= lr_m)
                & (df_used["CurrentAge"] >= fr_m)
            ]
        )
        num_proven_breeders_senior_male: int = len(
            df_used[
                (df_used["anonID"].isin(proven_breeders_male_all))
                & (df_used["CurrentAge"] > lr_m)
            ]
        )
        num_proven_breeders_male = (
            num_proven_breeders_adult_male + num_proven_breeders_senior_male
        )

        num_adult_male = len(
            df_used[
                (df_used["CurrentAge"] >= fr_m)
                & (df_used["CurrentAge"] <= lr_m)
                & (df_used["Sex"] == "Male")
            ]
        )
        num_senior_male = len(
            df_used[(df_used["CurrentAge"] > lr_m) & (df_used["Sex"] == "Male")]
        )
        num_male = len(df_used[df_used["Sex"] == "Male"])
        num_juvenile_male = max(num_male - num_adult_male - num_senior_male, 0)

        proven_breeders_male = (
            num_proven_breeders_male / num_male if num_male > 0 else None
        )
        proven_breeders_female = (
            num_proven_breeders_female / num_female if num_female > 0 else None
        )
        proven_breeders_male_adult_senior = (
            num_proven_breeders_male / (num_adult_male + num_senior_male)
            if num_adult_male + num_senior_male > 0
            else None
        )
        proven_breeders_female_adult_senior = (
            num_proven_breeders_female / (num_adult_female + num_senior_female)
            if num_adult_female + num_senior_female > 0
            else None
        )

        proven_breeders_adult_male = (
            num_proven_breeders_adult_male / num_adult_male
            if num_adult_male > 0
            else None
        )
        proven_breeders_adult_female = (
            num_proven_breeders_adult_female / num_adult_female
            if num_adult_female > 0
            else None
        )
        proven_breeders_senior_male = (
            num_proven_breeders_senior_male / num_senior_male
            if num_senior_male > 0
            else None
        )
        proven_breeders_senior_female = (
            num_proven_breeders_senior_female / num_senior_female
            if num_senior_female > 0
            else None
        )

        num_proven_breeders_all_sex = (
            num_proven_breeders_female + num_proven_breeders_male
        )
        num_proven_breeders_all_sex_adult = (
            num_proven_breeders_adult_male + num_proven_breeders_adult_female
        )
        num_proven_breeders_all_sex_senior = (
            num_proven_breeders_senior_male + num_proven_breeders_senior_female
        )

        proven_breeders_all_sex = (
            num_proven_breeders_all_sex / (num_male + num_female)
            if num_male + num_female > 0
            else None
        )
        proven_breeders_all_sex_adult = (
            num_proven_breeders_all_sex_adult / (num_adult_male + num_adult_female)
            if num_adult_male + num_adult_female > 0
            else None
        )
        proven_breeders_all_sex_senior = (
            num_proven_breeders_all_sex_senior / (num_senior_male + num_senior_female)
            if num_senior_male + num_senior_female > 0
            else None
        )
        proven_breeders_all_sex_adultsenior = (
            num_proven_breeders_all_sex
            / (num_adult_male + num_senior_male + num_adult_female + num_senior_female)
            if num_adult_male + num_senior_male + num_adult_female + num_senior_female
            > 0
            else None
        )

        # birth givers
        adult_dams, senior_dams, adult_sires, senior_sires = self.get_number_breeders(
            df_used, year, lr_m, lr_f
        )
        prop_birth_givers_male = (
            (adult_sires + senior_sires) / (num_adult_male + num_senior_male)
            if num_adult_male + num_senior_male > 0
            else None
        )
        prop_birth_givers_male_adult = (
            adult_sires / num_adult_male if num_adult_male > 0 else None
        )
        prop_birth_givers_male_senior = (
            senior_sires / num_senior_male if num_senior_male > 0 else None
        )
        prop_birth_givers_female = (
            (adult_dams + senior_dams) / (num_adult_female + num_senior_female)
            if num_adult_female + num_senior_female > 0
            else None
        )
        prop_birth_givers_female_adult = (
            adult_dams / num_adult_female if num_adult_female > 0 else None
        )
        prop_birth_givers_female_senior = (
            senior_dams / num_senior_female if num_senior_female > 0 else None
        )

        for age, num_ind in age_quant_m.items():
            append_row(
                ret,
                year,
                "absolute",
                age,
                "male",
                num_ind,
                median_age_male,
                mean_age_male,
                proven_breeders_male,
                num_birth_male,
                num_proven_breeders_male,
                adult_sires,
                senior_sires,
                num_proven_breeders_adult_male,
                num_proven_breeders_senior_male,
                proven_breeders_adult_male,
                proven_breeders_senior_male,
                prop_birth_givers_male,
                prop_birth_givers_male_adult,
                prop_birth_givers_male_senior,
                median_age,
                num_proven_breeders_male,
                proven_breeders_male_adult_senior,
                num_proven_breeders_all_sex,
                num_proven_breeders_all_sex_adult,
                num_proven_breeders_all_sex_senior,
                num_proven_breeders_all_sex,
                proven_breeders_all_sex,
                proven_breeders_all_sex_adult,
                proven_breeders_all_sex_senior,
                proven_breeders_all_sex_adultsenior,
                num_juvenile_male,
                num_adult_male,
                num_senior_male,
            )
        for age, num_ind in age_quant_f.items():
            append_row(
                ret,
                year,
                "absolute",
                age,
                "female",
                num_ind,
                median_age_female,
                mean_age_female,
                proven_breeders_female,
                num_birth_female,
                num_proven_breeders_female,
                adult_dams,
                senior_dams,
                num_proven_breeders_adult_female,
                num_proven_breeders_senior_female,
                proven_breeders_adult_female,
                proven_breeders_senior_female,
                prop_birth_givers_female,
                prop_birth_givers_female_adult,
                prop_birth_givers_female_senior,
                median_age,
                num_proven_breeders_female,
                proven_breeders_female_adult_senior,
                num_proven_breeders_all_sex,
                num_proven_breeders_all_sex_adult,
                num_proven_breeders_all_sex_senior,
                num_proven_breeders_all_sex,
                proven_breeders_all_sex,
                proven_breeders_all_sex_adult,
                proven_breeders_all_sex_senior,
                proven_breeders_all_sex_adultsenior,
                num_juvenile_female,
                num_adult_female,
                num_senior_female,
            )

        age_scaled_m = self.get_scaled_pyramid_information(
            df_used, year, ["Male"], max_age
        )
        age_scaled_f = self.get_scaled_pyramid_information(
            df_used, year, ["Female"], max_age
        )

        for age, num_ind in age_scaled_m.items():
            append_row(
                ret,
                year,
                "relative",
                age,
                "male",
                num_ind,
                median_age_male,
                mean_age_male,
                proven_breeders_male,
                num_birth_male,
                num_proven_breeders_male,
                adult_sires,
                senior_sires,
                num_proven_breeders_adult_male,
                num_proven_breeders_senior_male,
                proven_breeders_adult_male,
                proven_breeders_senior_male,
                prop_birth_givers_male,
                prop_birth_givers_male_adult,
                prop_birth_givers_male_senior,
                median_age,
                num_proven_breeders_male,
                proven_breeders_male_adult_senior,
                num_proven_breeders_all_sex,
                num_proven_breeders_all_sex_adult,
                num_proven_breeders_all_sex_senior,
                num_proven_breeders_all_sex,
                proven_breeders_all_sex,
                proven_breeders_all_sex_adult,
                proven_breeders_all_sex_senior,
                proven_breeders_all_sex_adultsenior,
                num_juvenile_male,
                num_adult_male,
                num_senior_male,
            )
        for age, num_ind in age_scaled_f.items():
            append_row(
                ret,
                year,
                "relative",
                age,
                "female",
                num_ind,
                median_age_female,
                mean_age_female,
                proven_breeders_female,
                num_birth_female,
                num_proven_breeders_female,
                adult_dams,
                senior_dams,
                num_proven_breeders_adult_female,
                num_proven_breeders_senior_female,
                proven_breeders_adult_female,
                proven_breeders_senior_female,
                prop_birth_givers_female,
                prop_birth_givers_female_adult,
                prop_birth_givers_female_senior,
                median_age,
                num_proven_breeders_female,
                proven_breeders_female_adult_senior,
                num_proven_breeders_all_sex,
                num_proven_breeders_all_sex_adult,
                num_proven_breeders_all_sex_senior,
                num_proven_breeders_all_sex,
                proven_breeders_all_sex,
                proven_breeders_all_sex_adult,
                proven_breeders_all_sex_senior,
                proven_breeders_all_sex_adultsenior,
                num_juvenile_female,
                num_adult_female,
                num_senior_female,
            )

        return pd.DataFrame(ret)

    def gather_statistics(
        self, df: pd.DataFrame, species_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
        year_range: Tuple[int, int] = self.get_minimum_maximum_year(df, species_name)
        year_range_list: List[Tuple[pd.DataFrame, int]] = [
            (df.copy(), year) for year in range(year_range[0] + 1, year_range[1] + 1)
        ]

        config.initialise_config_variables()

        proven_breeders_female = list(set(df["Dam_ID"].to_list()))
        proven_breeders_male = list(set(df["Sire_ID"].to_list()))

        if not config.ONLY_PYRAMID_STATISTIC_GENERATION:

            now: datetime = datetime.now()
            basic_stat_dfs: List[pd.DataFrame] = process_map(
                self.gather_stats_one_year,
                year_range_list,
                max_workers=config.MAX_WORKERS,
                desc=f"{now.strftime('%Y-%m-%d %H:%M:%S')} - Gathering basic statistics per year.",
            )

            basic_stats: pd.DataFrame = self.add_species_information_to_output(
                pd.concat(basic_stat_dfs, ignore_index=True), species_name, None
            )

            maximum_age = max(
                basic_stats[basic_stats["Quantity"] == "maximum_age"]["Value"]
            )

            (
                ages_f,
                probs_f,
                alpha_f,
                theta_f,
                norm_f,
                threshold_f,
                samplesize_f,
                fecundity_df_f,
            ) = self.get_fecundity_threshold(
                df, maximum_age, year_range, "Female", species_name
            )
            (
                ages_m,
                probs_m,
                alpha_m,
                theta_m,
                norm_m,
                threshold_m,
                samplesize_m,
                fecundity_df_m,
            ) = self.get_fecundity_threshold(
                df, maximum_age, year_range, "Male", species_name
            )

            ensure_diretory(config.OUTPUT_FOLDER_FECUNDITY_INFORMATION)
            pd.concat([fecundity_df_m, fecundity_df_f]).to_excel(
                f"{config.OUTPUT_FOLDER_FECUNDITY_INFORMATION}{species_name}-fecundity_statistics.xlsx",
                index=False,
            )

            fecundity_information: Dict[str, Any] = {
                "age": [],
                "sex": [],
                "species": [],
                "fitted_alpha": [],
                "fitted_theta": [],
                "fitted_norm": [],
                "fecundity probability": [],
                "last reproduction threshold": [],
                "sample size": [],
            }

            for j in range(len(ages_f)):
                fecundity_information["age"].append(ages_f[j])
                fecundity_information["sex"].append("female")
                fecundity_information["species"].append(species_name)
                fecundity_information["fecundity probability"].append(probs_f[j])
                fecundity_information["fitted_alpha"].append(alpha_f)
                fecundity_information["fitted_theta"].append(theta_f)
                fecundity_information["fitted_norm"].append(norm_f)
                fecundity_information["last reproduction threshold"].append(threshold_f)
                fecundity_information["sample size"].append(samplesize_f)

            for j in range(len(ages_m)):
                fecundity_information["age"].append(ages_m[j])
                fecundity_information["sex"].append("male")
                fecundity_information["species"].append(species_name)
                fecundity_information["fecundity probability"].append(probs_m[j])
                fecundity_information["fitted_alpha"].append(alpha_m)
                fecundity_information["fitted_theta"].append(theta_m)
                fecundity_information["fitted_norm"].append(norm_m)
                fecundity_information["last reproduction threshold"].append(threshold_m)
                fecundity_information["sample size"].append(samplesize_m)

            fecundity_df = pd.DataFrame(fecundity_information)

        else:
            basic_stats = pd.read_csv(
                f"{config.OUTPUT_FOLDER_STATISTICS}{species_name}_basics.csv"
            )
            fecundity_df = pd.read_csv(
                f"{config.OUTPUT_FOLDER_STATISTICS}{species_name}_fecundity.csv"
            )
            maximum_age = max(
                basic_stats[basic_stats["Quantity"] == "maximum_age"]["Value"]
            )

        if not config.SKIP_PYRAMID_STATISTIC_GENERATION:
            df_pyramid = df[
                (df["FirstRegion"].isin(config.USE_REGIONS_PYRAMIDS))
                & (df["use_for_pyramids"] != "no")
                & (df["use_for_pyramids"] != "0")
                & (df["use_for_pyramids"] != 0)
            ]
            year_range_list: List[Tuple[pd.DataFrame, int, float, str, List, List]] = [
                (
                    df_pyramid,
                    year,
                    maximum_age,
                    species_name,
                    proven_breeders_female,
                    proven_breeders_male,
                )
                for year in range(year_range[0] + 1, year_range[1] + 1)
            ]
            now: datetime = datetime.now()
            pyramid_stat_dfs: List[pd.DataFrame] = process_map(
                self.gather_pyramid_values_one_year,
                year_range_list,
                max_workers=config.MAX_WORKERS,
                desc=f"{now.strftime('%Y-%m-%d %H:%M:%S')} - Gathering pyramid build information per year.",
            )

            pyramid_stats: pd.DataFrame | None = self.add_species_information_to_output(
                pd.concat(pyramid_stat_dfs, ignore_index=True), species_name, None
            )
        else:
            pyramid_stats = None

        return basic_stats, pyramid_stats, fecundity_df

    def add_species_information_to_output(
        self, df: pd.DataFrame, species: str, species_information_path: Optional[str]
    ):

        add_information: Dict[str, str] = {"species": species}
        if species_information_path is not None:
            pass

        for key, value in add_information.items():
            df[key] = value

        return df
