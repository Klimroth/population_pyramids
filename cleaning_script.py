#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import pandas as pd
from tqdm import tqdm


def split_birthdate(dataset_filepath: str, dismiss_csv_path: str, keep_csv_path: str):
    df = pd.read_csv(dataset_filepath, sep=";")
    dismiss = ["ApproxAfter", "ApproxBefore", "Undetermined"]
    df_dismiss = df[
        (df["birthEst"].isin(dismiss)) | (df["deathEst"].isin(dismiss))
    ].reset_index(drop=True)
    print(f"We dismiss {len(df_dismiss)} many individuals overall.")
    df_keep = df[
        (~df["birthEst"].isin(dismiss)) & (~df["deathEst"].isin(dismiss))
    ].reset_index(drop=True)
    print(f"We keep {len(df_keep)} many individuals overall.")

    df_dismiss.to_csv(dismiss_csv_path, sep=";", index=False)
    df_keep.to_csv(keep_csv_path, sep=";", index=False)


def map_species(input_csv_path: str, output_csv_path: str):
    df = pd.read_csv(input_csv_path, sep=";")

    index_special_case = df[
        (
            df["binSpecies"].isin(
                ["Babyrousa sp.", "Babyrousa celebensis", "Babyrousa babyrousa"]
            )
        )
        & (df["BirthDate"] > "1970")
    ].index
    df.loc[index_special_case, "binSpecies"] = "Babyrousa celebensis"

    df["speciesOriginal"] = df["binSpecies"]
    species_map = pd.read_csv(p, index_col="species_given")
    species_mapping = {}
    for spec in species_map.index:
        species_mapping[spec] = species_map.loc[spec, "species_desired"]

    df["binSpecies"] = df["binSpecies"].replace(species_mapping)
    df.to_csv(output_csv_path, sep=";")


def get_species_count(dataset_csv_file_path: str, sample_size_excel_path: str):
    df = pd.read_csv(dataset_csv_file_path, sep=";")
    regions = ["Europe", "North America", "South America", "Africa", "Oceania", "Asia"]

    d_count = {"species": []}
    for region in regions:
        d_count[region] = []

    for species in tqdm(set(df["binSpecies"].to_list())):
        d_count["species"].append(species)
        for region in regions:
            df_reg = df[(df["binSpecies"] == species) & (df["FirstRegion"] == region)]
            d_count[region].append(len(df_reg))

    pd.DataFrame(d_count).to_excel(sample_size_excel_path, index=False)


def get_reduced_dataset(
    csv_path_all_species: str,
    csv_path_large_species: str,
    sample_size_all_excel_path: str,
    sample_size_large_excel_path: str,
    species_information_excel_path: str,
):
    df = pd.read_csv(csv_path_all_species, sep=";")
    df_samplesize = pd.read_excel(sample_size_all_excel_path)

    species_kept = list(
        set(
            df_samplesize[
                (df_samplesize["Europe"] >= 150)
                | (df_samplesize["North America"] >= 150)
            ]["species"].to_list()
        )
    )
    df = df[df["binSpecies"].isin(species_kept)]
    df.to_csv(csv_path_large_species, sep=";")
    print("number individuals remaining", len(df.index))

    df_samplesize[
        (df_samplesize["Europe"] >= 150) | (df_samplesize["North America"] >= 150)
    ].to_excel(sample_size_large_excel_path, index=False)

    pd.DataFrame({"species": [s for s in species_kept]}).to_excel(
        species_information_excel_path,
        index=False,
    )


def clean_dam_sire(csv_path_large_species: str, csv_path_cleaned_large_species: str):
    df = pd.read_csv(csv_path_large_species, sep=";")
    df = remove_invalid_dam_sire(df)
    print(
        "After cleaning DAM and SIRE we have still this many individuals: ",
        len(df.index),
    )
    df.to_csv(
        csv_path_cleaned_large_species,
        sep=";",
    )


def remove_invalid_dam_sire(df_in: pd.DataFrame) -> pd.DataFrame:

    df = df_in.copy()

    keep_indices: List[int] = []
    df["Dam_ID"] = None
    df["Sire_ID"] = None
    for row in tqdm(df.index):
        keep_index = True
        current_individual = df.loc[row, "anonID"]
        if not pd.notna(df.loc[row, "Dam_AnonID"]):
            df.loc[row, "Dam_ID"] = None
        elif (
            pd.notna(df.loc[row, "Dam_Probability"])
            and float(df.loc[row, "Dam_Probability"].replace(",", ".")) / 100 < 1
        ):
            df.loc[row, "Dam_ID"] = None
        else:
            dam_id: int = int(df.loc[row, "Dam_AnonID"])
            if len(df[df["anonID"] == dam_id]) == 0:
                df.loc[row, "Dam_ID"] = None
                df.loc[row, "Dam_AnonID"] = None
            elif len(df[df["anonID"] == dam_id]) > 1:
                keep_index = False
            elif not df[df["anonID"] == dam_id]["Sex"].tolist()[0] == "Female":
                df.loc[row, "Dam_ID"] = None
                df.loc[row, "Dam_AnonID"] = None
            else:
                df.loc[row, "Dam_ID"] = dam_id

        if not pd.notna(df.loc[row, "Sire_AnonID"]):
            df.loc[row, "Sire_ID"] = None
        elif (
            pd.notna(df.loc[row, "Sire_Probability"])
            and float(df.loc[row, "Sire_Probability"].replace(",", ".")) / 100 < 1
        ):
            df.loc[row, "Sire_ID"] = None
        else:
            sire_id: int = int(df.loc[row, "Sire_AnonID"])
            if len(df[df["anonID"] == sire_id]) == 0:
                df.loc[row, "Sire_ID"] = None
                df.loc[row, "Sire_AnonID"] = None
            elif len(df[df["anonID"] == sire_id]) > 1:
                keep_index = False
            elif not df[df["anonID"] == sire_id]["Sex"].tolist()[0] == "Male":
                df.loc[row, "Sire_ID"] = None
                df.loc[row, "Sire_AnonID"] = None
            else:
                df.loc[row, "Sire_ID"] = sire_id

        if keep_index:
            keep_indices.append(row)
    df = df.iloc[keep_indices].reset_index(drop=True)
    return df
