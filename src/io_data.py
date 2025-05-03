import os
from datetime import date, datetime, timedelta
from typing import Any, List, Optional, Tuple

import pandas as pd

import config
from main import LOG


def check_configuration_for_species_information(species: str) -> bool:
    allowed_species: List[str] = [
        orig for orig, mapped in config.SPECIES_NAME_MAPPER.items() if mapped == species
    ]
    if len(allowed_species) == 0:
        LOG.error(
            f"Species {species} is not contained as a value in SPECIES_NAME_MAPPER. Skipped."
        )
        return False
    if species not in config.SPECIES_AGE_BOUNDARY:
        LOG.error(
            f"Species {species} is not contained as a key in SPECIES_AGE_BOUNDARY. Skipped."
        )
        return False

    return True


def remove_invalid_dam_sire(df_in: pd.DataFrame) -> pd.DataFrame:

    df = df_in.copy()

    now: datetime = datetime.now()
    df["Dam_ID"] = None
    df["Sire_ID"] = None
    df["Problematic_Parent_ID"] = None

    for row in df.index:

        current_individual = df.loc[row, "anonID"]
        original_dam = (
            df.loc[row, "Dam_AnonID"] if pd.notna(df.loc[row, "Dam_AnonID"]) else None
        )
        original_sire = (
            df.loc[row, "Sire_AnonID"] if pd.notna(df.loc[row, "Sire_AnonID"]) else None
        )
        new_dam = None
        new_sire = None
        if pd.notna(df.loc[row, "Dam_AnonID"]):
            dam_id = int(df.loc[row, "Dam_AnonID"])
        else:
            dam_id = None
        if pd.notna(df.loc[row, "Sire_AnonID"]):
            sire_id = int(df.loc[row, "Sire_AnonID"])
        else:
            sire_id = None

        set_dam = False
        set_sire = False
        # dam id existiert und ist weiblich -> übernehmen
        if dam_id and len(df[(df["anonID"] == dam_id) & (df["Sex"] == "Female")]) == 1:
            df.loc[row, "Dam_ID"] = dam_id
            new_dam = dam_id
            set_dam = True
        # sire id existiert und ist männlich -> übernehmen
        if sire_id and len(df[(df["anonID"] == sire_id) & (df["Sex"] == "Male")]) == 1:
            df.loc[row, "Sire_ID"] = sire_id
            new_sire = sire_id
            set_sire = True
        # dam id existiert (ist aber männlich) und sire id ist nicht gültig -> nehme dam_id als Sire
        if (
            not set_dam
            and dam_id
            and not sire_id
            and len(df[(df["anonID"] == dam_id) & (df["Sex"] == "Male")]) == 1
        ):
            df.loc[row, "Sire_ID"] = dam_id
            new_sire = dam_id
        # sire id existiert (ist aber weiblich) und dam id ist nicht gültig -> nehme sire_id als dam
        if (
            not set_sire
            and sire_id
            and not dam_id
            and len(df[(df["anonID"] == sire_id) & (df["Sex"] == "Female")]) == 1
        ):
            df.loc[row, "Dam_ID"] = sire_id
            new_dam = sire_id

        if (new_dam != original_dam) or (new_sire != original_sire):
            df.loc[row, "Problematic_Parent_ID"] = True

        if (
            set_dam
            and pd.notna(df.loc[row, "Dam_Probability"])
            and float(df.loc[row, "Dam_Probability"].replace(",", ".")) / 100
            < config.MINIMUM_DAM_SIRE_CERTAINTY
        ):
            df.loc[row, "Dam_ID"] = None
        if (
            set_sire
            and pd.notna(df.loc[row, "Sire_Probability"])
            and float(df.loc[row, "Sire_Probability"].replace(",", ".")) / 100
            < config.MINIMUM_DAM_SIRE_CERTAINTY
        ):
            df.loc[row, "Sire_ID"] = None

        """
        
        try:
            current_individual = df.loc[row, "anonID"]
            if not pd.notna(df.loc[row, "Dam_AnonID"]):
                df.loc[row, "Dam_ID"] = None
            elif (
                pd.notna(df.loc[row, "Dam_Probability"])
                and float(df.loc[row, "Dam_Probability"].replace(",", ".")) / 100
                < config.MINIMUM_DAM_SIRE_CERTAINTY
            ):
                dam_id: int = int(df.loc[row, "Dam_AnonID"])
                df.loc[row, "Dam_ID"] = None
                LOG.warning(
                    f"Dam ID {dam_id} is not certain enough (individual {current_individual}). Set to None."
                )
            else:
                dam_id: int = int(df.loc[row, "Dam_AnonID"])
                if len(df[df["anonID"] == dam_id]) == 0:
                    LOG.error(
                        f"Dam ID {dam_id} is not existent as an individual itself. Set to None."
                    )
                    df.loc[row, "Dam_ID"] = None
                    df.loc[row, "Dam_AnonID"] = None
                elif len(df[df["anonID"] == dam_id]) > 1:
                    LOG.error(
                        f"Dam ID {dam_id} has multiple rows. Remove this row (individual {current_individual})."
                    )
                    keep_index = False
                elif not df[df["anonID"] == dam_id]["Sex"].tolist()[0] == "Female":
                    LOG.error(f"Dam ID {dam_id} is not a female individual itself.")
                    df.loc[row, "Dam_ID"] = None
                    df.loc[row, "Dam_AnonID"] = None
                else:
                    df.loc[row, "Dam_ID"] = dam_id

            if not pd.notna(df.loc[row, "Sire_AnonID"]):
                df.loc[row, "Sire_ID"] = None
            elif (
                pd.notna(df.loc[row, "Sire_Probability"])
                and float(df.loc[row, "Sire_Probability"].replace(",", ".")) / 100
                < config.MINIMUM_DAM_SIRE_CERTAINTY
            ):
                sire_id: int = int(df.loc[row, "Sire_AnonID"])
                df.loc[row, "Sire_ID"] = None
                LOG.warning(
                    f"Sire ID {sire_id} is not certain enough (individual {current_individual}). Set to None."
                )
            else:
                sire_id: int = int(df.loc[row, "Sire_AnonID"])
                if len(df[df["anonID"] == sire_id]) == 0:
                    LOG.error(
                        f"Dam ID {sire_id} is not existent as an individual itself. Set to None."
                    )
                    df.loc[row, "Sire_ID"] = None
                    df.loc[row, "Sire_AnonID"] = None
                elif len(df[df["anonID"] == sire_id]) > 1:
                    LOG.error(
                        f"Sire ID {sire_id} has multiple rows. Remove this row (individual {current_individual})."
                    )
                    keep_index = False
                elif not df[df["anonID"] == sire_id]["Sex"].tolist()[0] == "Male":
                    LOG.error(f"Sire ID {sire_id} is not a male individual itself.")
                    df.loc[row, "Sire_ID"] = None
                    df.loc[row, "Sire_AnonID"] = None
                else:
                    df.loc[row, "Sire_ID"] = sire_id
        except Exception as e:
            LOG.error(str(e))
            keep_index = False
    
    """

    return df


def read_species_information(src: str, species: str) -> Optional[pd.DataFrame]:
    """
    :return:
        ret[0]: clean variant of the data frame which is used for data extraction
    """
    config.initialise_config_variables()
    if not check_configuration_for_species_information(species):
        print(f"Configuration for {species} not ready.")
        return None

    if config.DATA_MODE == "PRE_SANITY_DATA":
        LOG.info(f"Start to read species information file {src} for species {species}.")
        if not os.path.exists(src):
            LOG.error(f"Datafile {src} not found.")
            return None
        allowed_species: List[str] = [
            orig
            for orig, mapped in config.SPECIES_NAME_MAPPER.items()
            if mapped == species
        ]
        LOG.info(f"The species names {allowed_species} are used.")

        df: pd.DataFrame = pd.read_csv(src, sep=";")
        if "Subspecies" in df.keys():
            df["binSpecies"] = df["Subspecies"]
        df = df[
            [
                "anonID",
                "binSpecies",
                "globStat",
                "Sex",
                "BirthDate",
                "DeathDate",
                "LastTXDate",
                "FirstRegion",
                "Dam_AnonID",
                "Sire_AnonID",
                "birthEst",
                "deathEst",
                "birthType",
                "Dam_Probability",
                "Sire_Probability",
            ]
        ]

        df = df[
            (df["binSpecies"].isin(allowed_species))
            & (df["FirstRegion"].isin(config.USE_REGIONS))
        ].reset_index()
    else:
        src_tmp = f"{src}{species}_overview_done.xlsx"
        LOG.info(f"Start to read species information file {src} for species {species}.")
        if not os.path.exists(f"{src_tmp}"):
            src_tmp = f"{src}{species}_overview.xlsx"
        if not os.path.exists(f"{src_tmp}"):
            LOG.error(f"Datafile {src_tmp} not found.")
            return None
        src = src_tmp
        allowed_species: List[str] = [
            orig
            for orig, mapped in config.SPECIES_NAME_MAPPER.items()
            if mapped == species
        ]
        LOG.info(f"The species names {allowed_species} are used.")

        df: pd.DataFrame = pd.read_excel(src)
        if "Subspecies" in df.keys():
            df["binSpecies"] = df["Subspecies"]
        df.rename(columns={"Individual_remove": "individual_remove"}, inplace=True)
        df.rename(columns={"Individuals_remove": "individual_remove"}, inplace=True)
        df.rename(columns={"individuals_remove": "individual_remove"}, inplace=True)
        df = df[
            [
                "anonID",
                "binSpecies",
                "globStat",
                "Sex",
                "LastTXDate",
                "FirstRegion",
                "birthEst",
                "deathEst",
                "birthType",
                "Dam_Probability",
                "Sire_Probability",
                "individual_remove",
                "Dam_ID",
                "Sire_ID",
                "use_for_pyramids",
                "CorrectedBirthDate",
                "DateOfDeath",
            ]
        ]

        df.rename(
            columns={
                "Dam_AnonID": "Original_Dam_AnonID",
                "Sire_AnonID": "Original_Sire_AnonID",
                "BirthDate": "Original_BirthDate",
                "DeathDate": "OriginalDeathDate",
            },
            inplace=True,
        )
        df.rename(
            columns={
                "Dam_ID": "Dam_AnonID",
                "Sire_ID": "Sire_AnonID",
                "CorrectedBirthDate": "BirthDate",
                "DateOfDeath": "DeathDate",
            },
            inplace=True,
        )

        df.replace("NA/A", None, inplace=True)

        df = df[
            (df["binSpecies"].isin(allowed_species))
            & (df["FirstRegion"].isin(config.USE_REGIONS))
            & (df["individual_remove"] != "yes")
        ].reset_index()

    if len(df) == 0:
        return None

    df.to_csv(f"{src[:-4]}-{allowed_species[0]}.csv", index=False)

    # check first date to see date format
    date_mode: str = "excel_days"
    if config.DATA_MODE == "AFTER_SANITY_DATA":
        date_mode = "timestamp_object"
    elif "-" in str(df["LastTXDate"].to_list()[0]):
        date_mode = "yyyymmdd"
    elif "." in str(df["LastTXDate"].to_list()[0]):
        date_mode = "ddmmyyyy"

    # identify year of birth and year of death
    df["DateOfBirth"] = None
    df["DateOfDeath"] = None

    df["YearOfBirth"] = None
    df["MonthOfBirth"] = None
    df["DayOfBirth"] = None
    df["YearOfDeath"] = None
    df["MonthOfDeath"] = None
    df["DayOfDeath"] = None

    changes_conducted: bool = True
    while changes_conducted:

        # remove double anonId
        df = df.drop_duplicates(subset="anonID", keep="last").reset_index(drop=True)

        now: datetime = datetime.now()
        keep_indices: List[int] = []
        current_len: int = len(df)
        for row in df.index:
            if (
                df.loc[row, "birthEst"] in config.DISMISS_BIRTH_ESTIMATION_CATEGORY
                or df.loc[row, "deathEst"] in config.DISMISS_DEATH_ESTIMATION_CATEGORY
            ):
                LOG.warning(
                    f"Individual {df.loc[row, 'anonID']} of species {df.loc[row, 'binSpecies']} has cannot be used due to uncertain birth/death date."
                )
                continue
            try:
                birth_date_csv: Any = (
                    df.loc[row, "BirthDate"]
                    if pd.notna(df.loc[row, "BirthDate"])
                    else None
                )
                last_tx_date_csv: Any = (
                    df.loc[row, "LastTXDate"]
                    if pd.notna(df.loc[row, "LastTXDate"])
                    else None
                )
                death_date_csv: Any = (
                    df.loc[row, "DeathDate"]
                    if pd.notna(df.loc[row, "DeathDate"])
                    else None
                )
                if birth_date_csv is None:
                    LOG.warning(
                        f"Individual {df.loc[row, 'anonID']} of species {df.loc[row, 'binSpecies']} has no BirthDate."
                    )
                    continue

                if last_tx_date_csv is None:
                    LOG.warning(
                        f"Individual {df.loc[row, 'anonID']} of species {df.loc[row, 'binSpecies']} has no LastTXDate."
                    )
                    continue

                if death_date_csv is None and df.loc[row, "globStat"] == "Dead":
                    LOG.warning(
                        f"Individual {df.loc[row, 'anonID']} of species {df.loc[row, 'binSpecies']} has no DeathDate but is dead."
                    )
                    continue

                if death_date_csv is None and df.loc[row, "globStat"] != "Alive":
                    LOG.info(
                        f"Individual {df.loc[row, 'anonID']} of species {df.loc[row, 'binSpecies']} was lost."
                    )
                    df.loc[row, "globStat"] = "Unknown"
                    death_date_csv = last_tx_date_csv

                keep_indices.append(row)

                birth_date: date = date(1900, 1, 1)
                last_tx_date: date = date(1900, 1, 1)
                death_date: date = date(4000, 1, 1)

                birth_year: float = 0.0
                death_year: float = 4000
                if date_mode == "excel_days":
                    birth_date = date(year=1899, month=12, day=30) + timedelta(
                        days=int(birth_date_csv)
                    )
                    last_tx_date = date(year=1899, month=12, day=30) + timedelta(
                        days=int(last_tx_date_csv)
                    )
                    if death_date_csv is not None:
                        death_date = date(year=1899, month=12, day=30) + timedelta(
                            days=int(death_date_csv)
                        )
                elif date_mode == "yyyymmdd":
                    y, m, d = birth_date_csv.split("-")
                    y, m, d = int(y), int(m), int(d)
                    birth_date = date(y, m, d)
                    y, m, d = last_tx_date_csv.split("-")
                    y, m, d = int(y), int(m), int(d)
                    last_tx_date = date(y, m, d)
                    if death_date_csv is not None:
                        y, m, d = death_date_csv.split("-")
                        y, m, d = int(y), int(m), int(d)
                        death_date = date(y, m, d)
                elif date_mode == "timestamp_object":
                    try:
                        birth_date = birth_date_csv.date()
                        if death_date_csv is not None:
                            death_date = death_date_csv.date()
                    except Exception as e:
                        # print(f"{species} - correcting date {birth_date_csv} on row {row}", e)

                        y, m, d = birth_date_csv.split("-")
                        y, m, d = int(y), int(m), int(d)
                        birth_date = date(y, m, d)

                        if death_date_csv is not None:

                            try:
                                death_date = death_date_csv.date()
                            except:
                                y, m, d = death_date_csv.split("-")
                                y, m, d = int(y), int(m), int(d)
                                death_date = date(y, m, d)

                else:
                    d, m, y = birth_date_csv.split("-")
                    y, m, d = int(y), int(m), int(d)
                    birth_date = date(y, m, d)
                    d, m, y = last_tx_date_csv.split("-")
                    y, m, d = int(y), int(m), int(d)
                    if death_date_csv is not None:
                        d, m, y = death_date_csv.split("-")
                        y, m, d = int(y), int(m), int(d)
                        death_date = date(y, m, d)

                birth_year = float(birth_date.year)
                death_year = float(death_date.year)

                df.loc[row, "DateOfBirth"] = birth_date
                df.loc[row, "DateOfDeath"] = death_date

                df.loc[row, "YearOfBirth"] = birth_year
                df.loc[row, "MonthOfBirth"] = float(birth_date.month)
                df.loc[row, "DayOfBirth"] = float(birth_date.day)
                df.loc[row, "YearOfDeath"] = death_year
                df.loc[row, "MonthOfDeath"] = float(death_date.month)
                df.loc[row, "DayOfDeath"] = float(death_date.day)

            except Exception as e:
                print(f"{species} - Error while reading line {row}", e)
        df = df.iloc[keep_indices].reset_index(drop=True)

        # clean invalid species names
        now: datetime = datetime.now()
        keep_indices: List[int] = []
        for row in df.index:
            if not df.loc[row, "binSpecies"] in config.SPECIES_NAME_MAPPER:
                LOG.warning(f"Unknown species name in row {row} for species {species}.")
                continue
            df.loc[row, "binSpecies"] = config.SPECIES_NAME_MAPPER[
                df.loc[row, "binSpecies"]
            ]
            keep_indices.append(row)
        df = df.iloc[keep_indices].reset_index(drop=True)

        df = remove_invalid_dam_sire(df)

        if len(df) == current_len:
            changes_conducted = False

    df["CorrectedBirthDate"] = df["DateOfBirth"]
    df.to_csv(f"{src[:-4]}-{allowed_species[0]}-clean.csv", index=False)

    rows = [  # used original data
        "anonID",
        "binSpecies",
        "globStat",
        "Sex",
        # not used original data
        "BirthDate",
        "DeathDate",
        "LastTXDate",
        "FirstRegion",
        "Dam_AnonID",
        "Sire_AnonID",
        "birthEst",
        "deathEst",
        "birthType",
        "Dam_Probability",
        "Sire_Probability",
        # new data
        "YearOfBirth",
        "MonthOfBirth",
        "DayOfBirth",
        "YearOfDeath",
        "MonthOfDeath",
        "DayOfDeath",
        "DateOfBirth",
        "CorrectedBirthDate",
        "DateOfDeath",
        "Dam_ID",
        "Sire_ID",
        "Problematic_Parent_ID",
    ]
    if config.DATA_MODE == "AFTER_SANITY_DATA":
        rows.append("individual_remove")
        rows.append("use_for_pyramids")

    return df[rows]
