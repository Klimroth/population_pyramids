import enum
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.special import gamma
from tqdm import tqdm

import config
from src.helper import ensure_diretory
from src.simple_logger import SimpleLogger


class DrawStatistics:

    def __init__(self, log: SimpleLogger):
        self.logger = log
        config.initialise_config_variables()

    def collect_data_image_fraction_age(
        self, df: pd.DataFrame, sex: str
    ) -> pd.DataFrame:
        return df[
            (df["Sex"] == sex)
            & (
                df["Quantity"].isin(
                    ["fraction_neonates", "fraction_subadult", "fraction_adult"]
                )
            )
        ]

    def collect_data_image_birth(
        self, df: pd.DataFrame, include_all: bool = False
    ) -> pd.DataFrame:
        df_local = df.copy()
        included_sex: List[str] = ["male", "female", "undetermined"]
        if include_all:
            included_sex: List[str] = ["male", "female", "all"]
        return df_local[
            (df_local["Sex"].isin(included_sex))
            & (df_local["Quantity"] == "number_births")
        ]

    def collect_data_image_death(
        self, df: pd.DataFrame, include_all: bool = False
    ) -> pd.DataFrame:
        included_sex: List[str] = ["male", "female", "undetermined"]
        if include_all:
            included_sex: List[str] = ["male", "female", "all"]
        return df[(df["Sex"].isin(included_sex)) & (df["Quantity"] == "number_deaths")]

    def collect_data_image_lost(
        self, df: pd.DataFrame, include_all: bool = False
    ) -> pd.DataFrame:
        included_sex: List[str] = ["male", "female", "undetermined"]
        if include_all:
            included_sex: List[str] = ["male", "female", "all"]
        return df[(df["Sex"].isin(included_sex)) & (df["Quantity"] == "number_lost")]

    def collect_data_image_alive(
        self, df: pd.DataFrame, include_all: bool = False
    ) -> pd.DataFrame:
        included_sex: List[str] = ["male", "female", "undetermined"]
        if include_all:
            included_sex: List[str] = ["male", "female", "all"]
        return df[(df["Sex"].isin(included_sex)) & (df["Quantity"] == "number_alive")]

    def collect_data_image_median_age(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df["Sex"].isin(["male", "female", "all"]))
            & (df["Quantity"] == "median_age")
        ]

    def collect_data_image_litter_size(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[(df["Quantity"] == "average_litter_size")]

    def collect_data_image_sex_proportion_newborn(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        return df[
            (df["Sex"].isin(["male", "female", "undetermined"]))
            & (df["Quantity"] == "proportion_newborn")
        ]

    def _generate_barplot_by_sex(
        self,
        df: pd.DataFrame,
        fig_title: str,
        y_label: str,
        sex_color: Dict[str, str],
        stacked: bool = True,
        year_label: str = "year",
    ):
        barmode = "group" if not stacked else "stack"
        fig = px.bar(
            df,
            x="Year",
            y="Value",
            color="Sex",
            title=fig_title,
            labels={
                "Year": year_label,
                "Value": y_label,
                "Sex": "sex",
            },
            color_discrete_map=sex_color,
            barmode=barmode,
        )
        fig = fig.update_layout(
            {
                "plot_bgcolor": "rgb(255, 255, 255)",
                "paper_bgcolor": "rgb(255, 255, 255)",
                "title_x": 0.5,
            }
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
            )
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=config.GRIDCOLOR, linecolor=config.GRIDCOLOR
        )
        fig.update_xaxes(showgrid=False, showline=True, linecolor=config.GRIDCOLOR)

        return fig

    def _generate_lineplot_by_sex(
        self,
        df: pd.DataFrame,
        fig_title: str,
        y_label: str,
        sex_color: Dict[str, str],
        year_label: str = "year",
    ):

        fig = px.line(
            df,
            x="Year",
            y="Value",
            color="Sex",
            title=fig_title,
            labels={
                "Year": year_label,
                "Value": y_label,
                "Sex": "sex",
            },
            color_discrete_map=sex_color,
            markers=True,
        )
        fig = fig.update_layout(
            {
                "plot_bgcolor": "rgb(255, 255, 255)",
                "paper_bgcolor": "rgb(255, 255, 255)",
                "title_x": 0.5,
            }
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
            )
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=config.GRIDCOLOR, linecolor=config.GRIDCOLOR
        )
        fig.update_xaxes(showgrid=False, showline=True, linecolor=config.GRIDCOLOR)

        return fig

    def _generate_lineplot_without_sex(
        self,
        df: pd.DataFrame,
        fig_title: str,
        y_label: str,
        sex_color: Dict[str, str],
        year_label: str = "year",
    ):

        fig = px.line(
            df,
            x="Year",
            y="Value",
            title=fig_title,
            labels={
                "Year": year_label,
                "Value": y_label,
                "Sex": "sex",
            },
            markers=True,
        )
        fig = fig.update_layout(
            {
                "plot_bgcolor": "rgb(255, 255, 255)",
                "paper_bgcolor": "rgb(255, 255, 255)",
                "title_x": 0.5,
            }
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
            )
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=config.GRIDCOLOR, linecolor=config.GRIDCOLOR
        )
        fig.update_xaxes(showgrid=False, showline=True, linecolor=config.GRIDCOLOR)

        return fig

    def _check_basic_statistic_file(
        self, df_location: str, species: str, image: str
    ) -> bool:
        if not species in config.SPECIES_NAME_MAPPER_GRAPHICS:
            self.logger.error(
                f"Cannot create {image} because {species} is not contained in SPECIES_NAME_MAPPER_GRAPHICS."
            )
            return False

        if not os.path.exists(df_location):
            self.logger.error(
                f"Cannot create {image} because {df_location} is not existent."
            )
            return False

        return True

    def _write_imagefile(self, fig, save_path, save_path_svg, scale):
        ensure_diretory(save_path)
        fig.write_image(save_path, scale=scale)
        if config.SAVE_ADDITIONALLY_AS_VECTORGRAPHIC:
            fig.write_image(
                save_path_svg,
                scale=scale,
            )

    def draw_fecundity_plot(self, species: str):
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

        df = df[df["age"] <= max_age]
        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        lr_m: float = df[df["sex"] == "male"]["last reproduction threshold"].values[0]
        lr_f: float = df[df["sex"] == "female"]["last reproduction threshold"].values[0]

        alpha_f: float = df[df["sex"] == "female"]["fitted_alpha"].values[0]
        alpha_m: float = df[df["sex"] == "male"]["fitted_alpha"].values[0]
        theta_f: float = df[df["sex"] == "female"]["fitted_theta"].values[0]
        theta_m: float = df[df["sex"] == "male"]["fitted_theta"].values[0]
        norm_f: float = df[df["sex"] == "female"]["fitted_norm"].values[0]
        norm_m: float = df[df["sex"] == "male"]["fitted_norm"].values[0]

        n_female: int = df[df["sex"] == "female"]["sample size"].values[0]
        n_male: int = df[df["sex"] == "male"]["sample size"].values[0]

        # draw lineplot with color by sex and two vertical lines (the threshold values)
        fig_title: str = f"Fecundity probability of <i>{species_name}</i>"
        scale = (config.WIDTH_BIRTH_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)

        fig = px.line(
            df,
            x="age",
            y="fecundity probability",
            color="sex",
            title=fig_title,
            color_discrete_map=config.SEX_COLOR_MAP,
            markers=True,
        )
        fig.update_traces(line=dict(width=0.75, dash="dash"))

        ages_kernel_plot = np.linspace(0, max_age, 5000)
        female_kernel = norm_f * np.array(
            [
                x ** (alpha_f - 1)
                * np.exp(-x / theta_f)
                / (gamma(alpha_f) * theta_f**alpha_f)
                for x in ages_kernel_plot
            ]
        )
        male_kernel = norm_m * np.array(
            [
                x ** (alpha_m - 1)
                * np.exp(-x / theta_m)
                / (gamma(alpha_m) * theta_m**alpha_m)
                for x in ages_kernel_plot
            ]
        )

        max_value: float = max(
            max(female_kernel),
            max(male_kernel),
            max(df["fecundity probability"].values.tolist()),
        )

        fig.add_scatter(
            x=ages_kernel_plot,
            y=female_kernel,
            mode="lines",
            line_dash="solid",
            line_width=1,
            line_color=config.SEX_COLOR_MAP["female"],
            showlegend=False,
        )
        fig.add_scatter(
            x=ages_kernel_plot,
            y=male_kernel,
            mode="lines",
            line_dash="solid",
            line_width=1,
            line_color=config.SEX_COLOR_MAP["male"],
            showlegend=False,
        )

        fig.add_vline(
            x=fr_m,
            line_dash="dot",
            line_color=config.SEX_COLOR_MAP["male"],
            line_width=1,
        )
        fig.add_vline(
            x=lr_m,
            line_dash="dot",
            line_color=config.SEX_COLOR_MAP["male"],
            line_width=1,
        )
        fig.add_vline(
            x=fr_f,
            line_dash="dot",
            line_color=config.SEX_COLOR_MAP["female"],
            line_width=1,
        )
        fig.add_vline(
            x=lr_f,
            line_dash="dot",
            line_color=config.SEX_COLOR_MAP["female"],
            line_width=1,
        )

        fig = fig.update_layout(
            {
                "plot_bgcolor": "rgb(255, 255, 255)",
                "paper_bgcolor": "rgb(255, 255, 255)",
                "title_x": 0.5,
            }
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
            )
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=config.GRIDCOLOR,
            linecolor=config.GRIDCOLOR,
        )
        fig.update_xaxes(showgrid=False, showline=True, linecolor=config.GRIDCOLOR)

        fig.add_annotation(
            x=0.9 * max_age,
            y=1.075 * max_value,
            text=f"<b>{n_male + n_female}</b> <br> {n_male}.{n_female}",
            showarrow=False,
        )

        save_path: str = (
            f"{config.OUTPUT_FOLDER_FECUNDITY_INFORMATION}/{species}_fecundity-plot.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_FECUNDITY_INFORMATION}/{species}_fecundity-plot.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)

    def generate_image_number_birth(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "annual birth plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df_bar = self.collect_data_image_birth(df.copy())
        df_line = self.collect_data_image_birth(df.copy(), True)

        fig_title: str = f"Annual number of births of <i>{species_name}</i>"
        y_label: str = "number of births"

        scale = (config.WIDTH_BIRTH_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
        fig = self._generate_barplot_by_sex(
            df=df_bar,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_birth.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_birth.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)

        fig_line = self._generate_lineplot_by_sex(
            df=df_line,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_birth-line.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_birth-line.svg"
        )
        self._write_imagefile(fig_line, save_path, save_path_svg, scale)

    def generate_image_proportion_newborn(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "sex proportion newborn plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df = self.collect_data_image_sex_proportion_newborn(df)

        fig_title: str = f"Sex proportion of newborn <i>{species_name}</i>"
        y_label: str = "proportion"

        scale = (config.WIDTH_PROPORTION_NEWBORN_IMAGE_MM / 25.4) / (
            700 / config.FIGURE_DPI
        )
        fig = self._generate_barplot_by_sex(
            df=df, fig_title=fig_title, y_label=y_label, sex_color=config.SEX_COLOR_MAP
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_proportion_newborn.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_proportion_newborn.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)

    def generate_image_number_death(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "annual death plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df_bar = self.collect_data_image_death(df.copy())
        df_line = self.collect_data_image_death(df.copy(), True)

        fig_title: str = f"Annual number of deaths of <i>{species_name}</i>"
        y_label: str = "number of deaths"

        scale = (config.WIDTH_DEATH_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
        fig = self._generate_barplot_by_sex(
            df=df_bar,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_death.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_death.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)

        fig_line = self._generate_lineplot_by_sex(
            df=df_line,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_death-line.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_death-line.svg"
        )
        self._write_imagefile(fig_line, save_path, save_path_svg, scale)

    def generate_image_number_lost(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "lost track plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df_bar = self.collect_data_image_lost(df.copy())
        df_line = self.collect_data_image_lost(df.copy(), True)

        fig_title: str = f"Lost individuals of <i>{species_name}</i>"
        y_label: str = "number of lost individuals"

        scale = (config.WIDTH_LOST_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
        fig = self._generate_barplot_by_sex(
            df=df_bar,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_lost.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_lost.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)

        fig_line = self._generate_lineplot_by_sex(
            df=df_line,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_lost-line.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_lost-line.svg"
        )
        self._write_imagefile(fig_line, save_path, save_path_svg, scale)

    def generate_image_number_alive(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(df_location, species, "alive plot"):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df_bar = self.collect_data_image_alive(df.copy())
        df_line = self.collect_data_image_alive(df.copy(), True)

        fig_title: str = f"Population size of <i>{species_name}</i>"
        y_label: str = "number of living individuals"

        scale = (config.WIDTH_LOST_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
        fig = self._generate_barplot_by_sex(
            df=df_bar,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_alive.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_alive.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)

        fig_line = self._generate_lineplot_by_sex(
            df=df_line,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_alive-line.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_annual_alive-line.svg"
        )
        self._write_imagefile(fig_line, save_path, save_path_svg, scale)

    def generate_image_median_age(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "median age plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df = self.collect_data_image_median_age(df)

        fig_title: str = f"Median age of <i>{species_name}</i>"
        y_label: str = "median age [years]"

        scale = (config.WIDTH_MEDIAN_AGE_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
        fig = self._generate_barplot_by_sex(
            df=df,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
            stacked=False,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_median_age.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_median_age.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)

        fig_line = self._generate_lineplot_by_sex(
            df=df,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_median_age-line.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_median_age-line.svg"
        )
        self._write_imagefile(fig_line, save_path, save_path_svg, scale)

    def generate_image_fraction_age(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "age fraction plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)

        for sex in ["male", "female", "all"]:
            df_sex = self.collect_data_image_fraction_age(df, sex)
            if sex == "all":
                fig_title: str = f"Age distribution of <i>{species_name}</i>"
            else:
                fig_title: str = f"Age distribution of {sex} <i>{species_name}</i>"

            fig_sex = px.bar(
                df_sex,
                x="Year",
                y="Value",
                color="Quantity",
                title=fig_title,
                labels={
                    "Year": "year",
                    "Value": "proportion of age group",
                    "Quantity": "",
                },
                color_discrete_map=config.AGE_DISTRIBUTION_COLOR_MAP,
            )
            fig_sex = fig_sex.update_layout(
                {
                    "plot_bgcolor": "rgb(255, 255, 255)",
                    "paper_bgcolor": "rgb(255, 255, 255)",
                    "title_x": 0.5,
                }
            )
            fig_sex.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
                )
            )

            fig_sex.update_traces(
                {"name": "neonate"}, selector={"name": "fraction_neonates"}
            )
            fig_sex.update_traces(
                {"name": "subadult"}, selector={"name": "fraction_subadult"}
            )
            fig_sex.update_traces(
                {"name": "adult"}, selector={"name": "fraction_adult"}
            )

            save_path: str = (
                f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_age_distribution_{sex}.png"
            )
            save_path_svg: str = (
                f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_age_distribution_{sex}.svg"
            )
            scale = (config.WIDTH_AGE_DISTRIBUTION_IMAGE_MM / 25.4) / (
                700 / config.FIGURE_DPI
            )
            self._write_imagefile(fig_sex, save_path, save_path_svg, scale)

    def generate_image_litter_size(self, species: str):

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "litter size plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df = self.collect_data_image_litter_size(df)[["Year", "Value"]]
        df = df.rolling(
            min_periods=1,
            window=config.ROLLING_AVERAGE_WINDOW,
            on="Year",
            closed="both",
        ).mean()

        fig_title: str = f"Rolling mean litter size of <i>{species_name}</i>"
        y_label: str = "litter size"

        scale = (config.WIDTH_MEDIAN_AGE_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
        fig_line = self._generate_lineplot_without_sex(
            df=df,
            fig_title=fig_title,
            y_label=y_label,
            sex_color=config.SEX_COLOR_MAP,
        )
        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_mean_litter_size-line.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_mean_litter_size-line.svg"
        )
        self._write_imagefile(fig_line, save_path, save_path_svg, scale)

    def generate_first_reproductive_image(self, species):
        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_basics.csv"
        if not self._check_basic_statistic_file(
            df_location, species, "litter size plot"
        ):
            return

        species_name = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df: pd.DataFrame = pd.read_csv(df_location)
        df_male = df[
            (df["Quantity"] == "mean_age_first_reproduction") & (df["Sex"] == "male")
        ][["Year", "Value"]]
        df_male = df_male.rolling(
            min_periods=1,
            window=config.ROLLING_AVERAGE_WINDOW,
            on="Year",
            closed="both",
        ).mean()
        df_male["Sex"] = "male"

        df_female = df[
            (df["Quantity"] == "mean_age_first_reproduction") & (df["Sex"] == "female")
        ][["Year", "Value"]]
        df_female = df_female.rolling(
            min_periods=1,
            window=config.ROLLING_AVERAGE_WINDOW,
            on="Year",
            closed="both",
        ).mean()
        df_female["Sex"] = "female"
        df = pd.concat([df_male, df_female])

        fig_title: str = (
            f"Rolling average first reproduction age of <i>{species_name}</i>"
        )
        scale = (config.WIDTH_MEDIAN_AGE_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
        y_max = int(np.ceil(1.05 * np.nanmax(df["Value"])))

        fig = px.line(
            df,
            x="Year",
            y="Value",
            color="Sex",
            range_y=(0, y_max),
            title=fig_title,
            labels={
                "Year": "year",
                "Value": "age [years]",
                "Sex": "sex",
            },
            color_discrete_map=config.SEX_COLOR_MAP,
            markers=True,
        )
        fig = fig.update_layout(
            {
                "plot_bgcolor": "rgb(255, 255, 255)",
                "paper_bgcolor": "rgb(255, 255, 255)",
                "title_x": 0.5,
            }
        )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
            )
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=config.GRIDCOLOR, linecolor=config.GRIDCOLOR
        )
        fig.update_xaxes(showgrid=False, showline=True, linecolor=config.GRIDCOLOR)

        save_path: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_average_first_reproduction_age.png"
        )
        save_path_svg: str = (
            f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/{species}_average_first_reproduction_age.svg"
        )
        self._write_imagefile(fig, save_path, save_path_svg, scale)


class PyramidTypes(str, enum.Enum):
    def __str__(self):
        return str(self.value)

    PYRAMID: str = ("pyramid",)
    I_PYRAMID: str = ("inverted pyramid",)
    HOURGLASS: str = ("hourglass",)
    PLUNGER: str = ("plunger",)
    I_PLUNGER: str = ("inverted plunger",)
    COL: str = ("column",)
    BELL: str = ("bell",)
    I_BELL: str = ("inverted bell",)
    L_DIAMOND: str = ("lower diamond",)
    M_DIAMOND: str = ("middle diamond",)
    U_DIAMOND: str = ("upper diamond",)


class DrawPopulationPyramid:

    def __init__(self, log: SimpleLogger):
        self.log = log

    def concatenate_pyramids(self, species: str, mode: str):
        image_folder: str = f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/pyramids/"
        image_paths: List[str] = [
            image_folder + f
            for f in os.listdir(image_folder)
            if f"{species}_Population-Pyramid_{mode}_" in f
        ]
        pyramid_years: List[int] = sorted(
            [int(img.split(".")[0].split("_")[-2]) for img in image_paths]
        )
        pyramid_years = sorted(
            [y for y in pyramid_years if y >= config.SKIP_PYRAMIDS_BEFORE_YEAR]
        )

        if len(pyramid_years) < 2:
            return

        decade_start: int = int(pyramid_years[0] / 10) * 10
        rounded_decade_end: int = int(pyramid_years[-1] / 10) * 10
        if rounded_decade_end < pyramid_years[-1] / 10:
            decade_end: int = rounded_decade_end + 10
        else:
            decade_end = rounded_decade_end

        # read one image to get height and width in pixel
        reference_img = cv2.imread(image_paths[0])
        reference_img = cv2.resize(
            reference_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC
        )
        white_img = 255 * np.ones(reference_img.shape, dtype=np.uint8)

        row_images = []
        for decade_year in [x for x in range(decade_start, decade_end + 1, 10)]:
            column_images = []
            for exact_year in range(10):
                current_year: int = decade_year + exact_year
                # print(current_year)
                if current_year in pyramid_years:
                    img = cv2.imread(
                        f"{image_folder}{species}_Population-Pyramid_{mode}_{current_year}_{config.REGION_STRING_PYRAMID_NAME}.png"
                    )
                    img = cv2.resize(
                        img,
                        (white_img.shape[1], white_img.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    column_images.append(img)
                    # print(img.shape)
                else:
                    column_images.append(white_img)
                    # print(white_img.shape)
            row_images.append(np.concatenate(column_images, axis=1))
        result_image = np.concatenate(row_images, axis=0)

        taxonomic_group_name = config.TAXONOMIC_GROUP_BY_SPECIES[species]
        output_folder = f"{config.OUTPUT_FOLDER_PYRAMID_OVERVIEW}graphs_pyramids/"
        ensure_diretory(output_folder)
        cv2.imwrite(
            f"{output_folder}{taxonomic_group_name}_{config.REGION_STRING_PYRAMID_NAME}_{species}.png",
            result_image,
        )

    def get_population_evaluation(
        self,
        df: pd.DataFrame,
        max_age: int,
        juvenile_thresh: float,
        senior_thresh: float,
    ) -> Tuple[Tuple[int, int, int, int], str, List[int], List[Tuple[float, float]]]:
        """
        Has to be called on a dataframe containing only one sex and one year, col Age and Individuals are used
        """
        num_ages_adult = int(senior_thresh - juvenile_thresh)
        age_threshold_adult = [
            round(juvenile_thresh) + x for x in np.linspace(0, num_ages_adult, 4)[1:3]
        ]

        age_borders: List[int] = (
            [0, juvenile_thresh] + age_threshold_adult + [senior_thresh, max_age]
        )
        bucket_0: List[float] = df[
            (df["Age"] >= age_borders[0]) & (df["Age"] <= age_borders[1])
        ]["Individuals"].to_list()
        bucket_1: List[float] = df[
            (df["Age"] >= age_borders[1]) & (df["Age"] <= age_borders[2])
        ]["Individuals"].to_list()
        bucket_2: List[float] = df[
            (df["Age"] >= age_borders[2]) & (df["Age"] <= age_borders[3])
        ]["Individuals"].to_list()
        bucket_3: List[float] = df[
            (df["Age"] >= age_borders[3]) & (df["Age"] <= age_borders[4])
        ]["Individuals"].to_list()
        bucket_4: List[float] = df[
            (df["Age"] >= age_borders[4]) & (df["Age"] <= age_borders[5])
        ]["Individuals"].to_list()

        bucket_stats: List[Tuple[float, float]] = [
            (
                (float(np.mean(bucket_0)), np.std(bucket_0) / np.sqrt(len(bucket_0)))
                if len(bucket_0) > 0
                else (0, 0)
            ),
            (
                (float(np.mean(bucket_1)), np.std(bucket_1) / np.sqrt(len(bucket_1)))
                if len(bucket_1) > 0
                else (0, 0)
            ),
            (
                (float(np.mean(bucket_2)), np.std(bucket_2) / np.sqrt(len(bucket_2)))
                if len(bucket_2) > 0
                else (0, 0)
            ),
            (
                (float(np.mean(bucket_3)), np.std(bucket_3) / np.sqrt(len(bucket_3)))
                if len(bucket_3) > 0
                else (0, 0)
            ),
            (
                (float(np.mean(bucket_4)), np.std(bucket_4) / np.sqrt(len(bucket_4)))
                if len(bucket_4) > 0
                else (0, 0)
            ),
        ]

        compare_matrix: np.ndarray = np.zeros(shape=(5, 5))

        for i in range(5):
            for j in range(5):
                current_comparison = 0
                if (
                    bucket_stats[i][0]
                    + config.PYRAMID_SEM_DISTINGUISH_FACTOR * bucket_stats[i][1]
                    < bucket_stats[j][0]
                    - config.PYRAMID_SEM_DISTINGUISH_FACTOR * bucket_stats[j][1]
                ):
                    current_comparison = -1
                    # i < j
                elif (
                    bucket_stats[i][0]
                    - config.PYRAMID_SEM_DISTINGUISH_FACTOR * bucket_stats[i][1]
                    > bucket_stats[j][0]
                    + config.PYRAMID_SEM_DISTINGUISH_FACTOR * bucket_stats[j][1]
                ):
                    current_comparison = 1
                    # i > j
                compare_matrix[i, j] = current_comparison

        # get inc/dec-sequence
        seq: Tuple[int, int, int, int] = (
            int(compare_matrix[1, 0]),
            int(compare_matrix[2, 1]),
            int(compare_matrix[3, 2]),
            int(compare_matrix[4, 3]),
        )

        # implement classification rule
        base_decision_rules = {
            (-1, -1, 1, 1): PyramidTypes.HOURGLASS,
            (-1, -1, 1, 0): PyramidTypes.HOURGLASS,
            (-1, -1, 0, 1): PyramidTypes.HOURGLASS,
            (1, 1, 0, 0): PyramidTypes.I_BELL,
            (1, 0, 0, 0): PyramidTypes.I_BELL,
            (0, 1, 0, 0): PyramidTypes.I_BELL,
            (0, 0, -1, -1): PyramidTypes.BELL,
            (0, 0, -1, 0): PyramidTypes.BELL,
            (0, 0, 0, -1): PyramidTypes.BELL,
            (0, 0, 1, 1): PyramidTypes.I_PLUNGER,
            (0, 0, 1, 0): PyramidTypes.I_PLUNGER,
            (0, 0, 0, 1): PyramidTypes.I_PLUNGER,
            (1, 1, 1, 1): PyramidTypes.I_PYRAMID,
            (1, 1, 1, 0): PyramidTypes.I_PYRAMID,
            (1, 1, 0, 1): PyramidTypes.I_PYRAMID,
            (1, 0, 1, 1): PyramidTypes.I_PYRAMID,
            (0, 1, 1, 1): PyramidTypes.I_PYRAMID,
            (0, 1, 1, 0): PyramidTypes.I_PYRAMID,
            (0, 1, 0, 1): PyramidTypes.I_PYRAMID,
            (1, 0, 1, 0): PyramidTypes.I_PYRAMID,
            (1, 0, 0, 1): PyramidTypes.I_PYRAMID,
            (1, -1, -1, -1): PyramidTypes.L_DIAMOND,
            (1, -1, -1, 0): PyramidTypes.L_DIAMOND,
            (1, -1, 0, -1): PyramidTypes.L_DIAMOND,
            (1, -1, 0, 0): PyramidTypes.L_DIAMOND,
            (1, 0, -1, -1): PyramidTypes.L_DIAMOND,
            (1, 0, -1, 0): PyramidTypes.L_DIAMOND,
            (1, 1, -1, -1): PyramidTypes.M_DIAMOND,
            (1, 1, -1, 0): PyramidTypes.M_DIAMOND,
            (1, 0, 0, -1): PyramidTypes.M_DIAMOND,
            (0, 1, -1, -1): PyramidTypes.M_DIAMOND,
            (0, 1, -1, 0): PyramidTypes.M_DIAMOND,
            (-1, -1, 0, 0): PyramidTypes.PLUNGER,
            (-1, 0, 0, 0): PyramidTypes.PLUNGER,
            (0, -1, 0, 0): PyramidTypes.PLUNGER,
            (-1, -1, -1, -1): PyramidTypes.PYRAMID,
            (-1, -1, -1, 0): PyramidTypes.PYRAMID,
            (-1, -1, 0, -1): PyramidTypes.PYRAMID,
            (-1, 0, -1, -1): PyramidTypes.PYRAMID,
            (-1, 0, -1, 0): PyramidTypes.PYRAMID,
            (-1, 0, 0, -1): PyramidTypes.PYRAMID,
            (0, -1, -1, -1): PyramidTypes.PYRAMID,
            (0, -1, -1, 0): PyramidTypes.PYRAMID,
            (0, -1, 0, -1): PyramidTypes.PYRAMID,
            (1, 1, 1, -1): PyramidTypes.U_DIAMOND,
            (1, 1, 0, -1): PyramidTypes.U_DIAMOND,
            (1, 0, 1, -1): PyramidTypes.U_DIAMOND,
            (0, 1, 1, -1): PyramidTypes.U_DIAMOND,
            (0, 1, 0, -1): PyramidTypes.U_DIAMOND,
            (0, 0, 1, -1): PyramidTypes.U_DIAMOND,
            (0, 0, 0, 0): PyramidTypes.COL,
        }

        classification = ""
        if seq in base_decision_rules:
            classification = base_decision_rules[seq]
        elif seq in [(-1, -1, -1, 1), (0, -1, -1, 1)]:
            if compare_matrix[2, 4] == 1:
                classification = PyramidTypes.PYRAMID
            elif compare_matrix[2, 4] == 0:
                classification = PyramidTypes.HOURGLASS
            else:
                classification = PyramidTypes.HOURGLASS
        elif seq == (1, 1, -1, 1):
            if compare_matrix[2, 4] == 1:
                classification = PyramidTypes.M_DIAMOND
            elif compare_matrix[2, 4] == 0:
                classification = PyramidTypes.I_BELL
            else:
                classification = PyramidTypes.I_PYRAMID
        elif seq == (0, 0, -1, 1):
            if compare_matrix[2, 4] == 1:
                classification = PyramidTypes.BELL
            elif compare_matrix[2, 4] == 0:
                classification = PyramidTypes.COL
            else:
                classification = PyramidTypes.I_PLUNGER
        elif seq == (0, 1, -1, 1):
            if compare_matrix[2, 4] == 1:
                classification = PyramidTypes.M_DIAMOND
            elif compare_matrix[2, 4] == 0:
                classification = PyramidTypes.I_BELL
            else:
                classification = PyramidTypes.I_PYRAMID
        elif seq == (1, 0, -1, 1):
            if compare_matrix[2, 4] == 1:
                classification = PyramidTypes.L_DIAMOND
            elif compare_matrix[2, 4] == 0:
                classification = PyramidTypes.I_BELL
            else:
                classification = PyramidTypes.I_PYRAMID
        elif seq == (0, -1, 1, 0):
            if compare_matrix[1, 3] == 1:
                classification = PyramidTypes.PLUNGER
            elif compare_matrix[1, 3] == 0:
                classification = PyramidTypes.HOURGLASS
            else:
                classification = PyramidTypes.I_PLUNGER
        elif seq == (0, -1, 0, 1):
            if compare_matrix[1, 4] == 1:
                classification = PyramidTypes.PYRAMID
            elif compare_matrix[1, 4] == 0:
                classification = PyramidTypes.HOURGLASS
            else:
                classification = PyramidTypes.I_PLUNGER
        elif seq == (1, -1, 1, 0):
            if compare_matrix[1, 3] == 1:
                classification = PyramidTypes.L_DIAMOND
            elif compare_matrix[1, 3] == 0:
                classification = PyramidTypes.I_BELL
            else:
                classification = PyramidTypes.I_PYRAMID
        elif seq == (0, -1, 1, 1):
            if compare_matrix[1, 3] == 1:
                classification = PyramidTypes.HOURGLASS
            elif compare_matrix[1, 3] == 0:
                classification = PyramidTypes.I_PLUNGER
            else:
                classification = PyramidTypes.I_PLUNGER
        elif seq == (-1, 0, 0, 1):
            if compare_matrix[1, 4] == 1:
                classification = PyramidTypes.PLUNGER
            elif compare_matrix[1, 4] == 0:
                classification = PyramidTypes.HOURGLASS
            else:
                classification = PyramidTypes.I_PLUNGER
        elif seq in [(-1, 0, 1, 1), (-1, 0, 1, 0)]:
            if compare_matrix[0, 3] == 1:
                classification = PyramidTypes.HOURGLASS
            elif compare_matrix[0, 3] == 0:
                classification = PyramidTypes.HOURGLASS
            else:
                classification = PyramidTypes.I_PLUNGER
        elif seq in [(-1, 1, 1, 0)]:
            if compare_matrix[0, 3] == 1:
                classification = PyramidTypes.HOURGLASS
            elif compare_matrix[0, 3] == 0:
                classification = PyramidTypes.HOURGLASS
            else:
                if compare_matrix[0, 2] == -1:
                    classification = PyramidTypes.I_PYRAMID
                else:
                    classification = PyramidTypes.HOURGLASS
        elif seq in [(-1, 1, -1, -1), (-1, 1, -1, 0)]:
            if compare_matrix[0, 2] == 1:
                classification = PyramidTypes.PYRAMID
            elif compare_matrix[0, 2] == 0:
                classification = PyramidTypes.BELL
            else:
                classification = PyramidTypes.M_DIAMOND
        elif seq == (-1, 1, 0, -1):
            if compare_matrix[0, 2] == 1:
                classification = PyramidTypes.PYRAMID
            elif compare_matrix[0, 2] == 0:
                classification = PyramidTypes.BELL
            else:
                classification = PyramidTypes.U_DIAMOND
        elif seq == (-1, 1, 0, 1):
            if compare_matrix[0, 2] == 1:
                classification = PyramidTypes.HOURGLASS
            elif compare_matrix[0, 2] == 0:
                classification = PyramidTypes.I_PLUNGER
            else:
                classification = PyramidTypes.I_PYRAMID
        elif seq == (-1, 1, 1, 1):
            if compare_matrix[0, 2] == 1:
                classification = PyramidTypes.HOURGLASS
            elif compare_matrix[0, 2] == 0:
                classification = PyramidTypes.I_PLUNGER
            else:
                classification = PyramidTypes.I_PYRAMID
        elif seq == (-1, 0, -1, 1):
            if compare_matrix[0, 4] == 0 or compare_matrix[1, 4] == -1:
                classification = PyramidTypes.HOURGLASS
            else:
                if compare_matrix[1, 4] == 0:
                    classification = PyramidTypes.PLUNGER
                else:
                    classification = PyramidTypes.PYRAMID
        elif seq == (-1, 1, 0, 0):
            if compare_matrix[0, 2] == 1:
                classification = PyramidTypes.PLUNGER
            elif compare_matrix[0, 2] == 0:
                classification = PyramidTypes.COL
            else:
                classification = PyramidTypes.I_BELL
        ########## complex rules
        elif seq == (1, -1, -1, 1):
            if compare_matrix[2, 4] == 1:
                classification = PyramidTypes.L_DIAMOND
            elif compare_matrix[2, 4] == 0:
                if compare_matrix[0, 2] == 1:
                    classification = PyramidTypes.PLUNGER
                else:
                    classification = PyramidTypes.L_DIAMOND
            else:
                if compare_matrix[0, 2] >= 0:
                    classification = PyramidTypes.HOURGLASS
                else:
                    classification = PyramidTypes.I_PYRAMID
        elif seq == (-1, 1, 1, -1):
            if compare_matrix[0, 3] == -1:
                classification = PyramidTypes.U_DIAMOND
            else:
                if compare_matrix[0, 2] > 0 and compare_matrix[2, 4] > 0:
                    classification = PyramidTypes.PYRAMID
                elif compare_matrix[2, 4] < 0:
                    classification = PyramidTypes.HOURGLASS
                else:
                    classification = PyramidTypes.BELL

        elif seq == (-1, 0, 1, -1):
            if compare_matrix[0, 4] <= 0:
                classification = PyramidTypes.U_DIAMOND
            else:
                if compare_matrix[0, 3] == 1:
                    classification = PyramidTypes.PYRAMID
                elif compare_matrix[0, 3] == 0:
                    classification = PyramidTypes.BELL
                else:
                    classification = PyramidTypes.HOURGLASS
        elif seq == (1, -1, 1, -1):
            if compare_matrix[0, 2] == 0 and compare_matrix[0, 4] == 0:
                if compare_matrix[1, 3] > 0:
                    classification = PyramidTypes.L_DIAMOND
                elif compare_matrix[1, 3] == 0:
                    classification = PyramidTypes.M_DIAMOND
                else:
                    classification = PyramidTypes.U_DIAMOND
            elif compare_matrix[0, 2] > 0 and compare_matrix[2, 4] > 0:
                classification = PyramidTypes.PYRAMID
            elif compare_matrix[0, 2] < 0 and compare_matrix[2, 4] < 0:
                classification = PyramidTypes.I_PYRAMID
            elif compare_matrix[0, 2] > 0 and compare_matrix[2, 4] < 0:
                if compare_matrix[1, 3] == 0:
                    classification = PyramidTypes.HOURGLASS
                elif compare_matrix[1, 3] > 0:
                    classification = PyramidTypes.L_DIAMOND
                else:
                    classification = PyramidTypes.U_DIAMOND
            else:
                if compare_matrix[1, 3] == 0:
                    classification = PyramidTypes.M_DIAMOND
                elif compare_matrix[1, 3] > 0:
                    classification = PyramidTypes.L_DIAMOND
                else:
                    classification = PyramidTypes.U_DIAMOND
        elif seq == (0, -1, 1, -1):
            if compare_matrix[1, 3] > 0:
                if compare_matrix[2, 4] > 0:
                    classification = PyramidTypes.PYRAMID
                elif compare_matrix[2, 4] == 0:
                    classification = PyramidTypes.PLUNGER
                else:
                    classification = PyramidTypes.HOURGLASS
            elif compare_matrix[1, 3] == 0:
                if compare_matrix[2, 4] < 0:
                    classification = PyramidTypes.HOURGLASS
                else:
                    classification = PyramidTypes.BELL
            else:
                classification = PyramidTypes.U_DIAMOND
        elif seq == (-1, -1, 1, -1):
            if compare_matrix[0, 3] > 0:
                if compare_matrix[2, 4] > 0:
                    classification = PyramidTypes.PYRAMID
                elif compare_matrix[2, 4] == 0:
                    classification = PyramidTypes.PLUNGER
                else:
                    classification = PyramidTypes.HOURGLASS
            elif compare_matrix[0, 3] == 0:
                classification = PyramidTypes.BELL
            else:
                if compare_matrix[0, 4] <= 0:
                    classification = PyramidTypes.U_DIAMOND
                else:
                    classification = PyramidTypes.HOURGLASS
        elif seq == (1, -1, 1, 1):
            if compare_matrix[0, 2] > 0:
                classification = PyramidTypes.HOURGLASS
            elif compare_matrix[0, 2] < 0:
                classification = PyramidTypes.I_PYRAMID
            else:
                if compare_matrix[1, 3] > 0:
                    classification = PyramidTypes.HOURGLASS
                elif compare_matrix[1, 3] == 0:
                    classification = PyramidTypes.I_PYRAMID
                else:
                    classification = PyramidTypes.I_PLUNGER
        elif seq == (-1, 1, -1, 1):
            if compare_matrix[0, 2] > 0:
                if compare_matrix[2, 4] > 0:
                    classification = PyramidTypes.PYRAMID
                elif compare_matrix[2, 4] < 0:
                    classification = PyramidTypes.HOURGLASS
                elif compare_matrix[2, 4] == 0:
                    classification = PyramidTypes.PLUNGER
            elif compare_matrix[0, 2] == 0:
                if compare_matrix[2, 4] == 0:
                    classification = PyramidTypes.COL
                elif compare_matrix[2, 4] > 0:
                    classification = PyramidTypes.BELL
                elif compare_matrix[2, 4] < 0:
                    classification = PyramidTypes.I_PLUNGER
            elif compare_matrix[0, 2] < 0:
                if compare_matrix[2, 4] < 0:
                    classification = PyramidTypes.I_PYRAMID
                elif compare_matrix[2, 4] == 0:
                    classification = PyramidTypes.I_BELL
                else:
                    classification = PyramidTypes.M_DIAMOND
        elif seq == (1, -1, 0, 1):
            if compare_matrix[0, 2] == 0:
                if compare_matrix[1, 4] <= 0:
                    classification = PyramidTypes.I_PLUNGER
                else:
                    classification = PyramidTypes.L_DIAMOND
            elif compare_matrix[0, 2] > 0:
                if compare_matrix[1, 4] < 0:
                    classification = PyramidTypes.HOURGLASS
                elif compare_matrix[1, 4] == 0:
                    classification = PyramidTypes.I_BELL
                else:
                    classification = PyramidTypes.L_DIAMOND
            else:
                if compare_matrix[1, 4] < 0:
                    classification = PyramidTypes.I_PYRAMID
                elif compare_matrix[1, 4] == 0:
                    classification = PyramidTypes.I_BELL
                else:
                    classification = PyramidTypes.L_DIAMOND
        else:
            print(f"ERROR: Sequence {seq} is not defined.")

        if classification == "":
            print(f"ERROR: Could not classify sequence {seq}.")

        return seq, str(classification), age_borders, bucket_stats

    def draw_pyramid(self, species: str, mode: str) -> bool:

        if not species in config.SPECIES_NAME_MAPPER_GRAPHICS:
            self.log.error(
                f"Cannot create population pyramid for {species} as the name is not in SPECIES_NAME_MAPPER_GRAPHICS."
            )
            return

        config.initialise_config_variables()
        fr_f = config.THRESHOLD_AGE_PYRAMID[species]["adult_threshold_female"]
        fr_m = config.THRESHOLD_AGE_PYRAMID[species]["adult_threshold_male"]
        lr_f = config.THRESHOLD_AGE_PYRAMID[species]["senior_threshold_female"]
        lr_m = config.THRESHOLD_AGE_PYRAMID[species]["senior_threshold_male"]

        overview_excel_data: Dict[str, List[Any]] = {
            "species": [],
            "year": [],
            "number_individuals": [],
            "number_male": [],
            "number_female": [],
            "perc_individuals_max_size": [],
            "perc_male_max_size": [],
            "perc_female_max_size": [],
            "number_juvenile": [],
            "number_adult": [],
            "number_senior": [],
            "number_adultSenior": [],
            "number_female_juvenile": [],
            "number_female_adult": [],
            "number_female_senior": [],
            "number_female_adultSenior": [],
            "number_male_juvenile": [],
            "number_male_adult": [],
            "number_male_senior": [],
            "number_male_adultSenior": [],
            "percentage_male": [],
            "percentage_female": [],
            "percentage_juvenile": [],
            "percentage_adult": [],
            "percentage_senior": [],
            "percentage_adultSenior": [],
            "percentage_male_juvenile": [],
            "percentage_male_adult": [],
            "percentage_male_senior": [],
            "percentage_male_adultSenior": [],
            "percentage_female_juvenile": [],
            "percentage_female_adult": [],
            "percentage_female_senior": [],
            "percentage_female_adultSenior": [],
            "mean_age_female": [],
            "mean_age_male": [],
            "mean_age": [],
            "median_age_female": [],
            "median_age_male": [],
            "median_age": [],
            "mean_age_perc_longevity": [],
            "mean_age_female_perc_longevity": [],
            "mean_age_male_perc_longevity": [],
            "median_age_perc_longevity": [],
            "median_age_female_perc_longevity": [],
            "median_age_male_perc_longevity": [],
            "num_proven_breeders": [],
            "proven_breeders": [],
            "num_proven_breeders_female": [],
            "proven_breeders_female": [],
            "num_proven_breeders_male": [],
            "proven_breeders_male": [],
            "num_proven_breeders_adult": [],
            "proven_breeders_adult": [],
            "num_proven_breeders_female_adult": [],
            "proven_breeders_female_adult": [],
            "num_proven_breeders_male_adult": [],
            "proven_breeders_male_adult": [],
            "num_proven_breeders_senior": [],
            "proven_breeders_senior": [],
            "num_proven_breeders_female_senior": [],
            "proven_breeders_female_senior": [],
            "num_proven_breeders_male_senior": [],
            "proven_breeders_male_senior": [],
            "num_proven_breeders_male_adultSenior": [],
            "proven_breeders_male_adultSenior": [],
            "num_proven_breeders_female_adultSenior": [],
            "proven_breeders_female_adultSenior": [],
            "num_proven_breeders_adultSenior": [],
            "proven_breeders_adultSenior": [],
            "num_birth_givers": [],
            "birth_givers": [],
            "num_birth_givers_female": [],
            "birth_givers_female": [],
            "num_birth_givers_male": [],
            "birth_givers_male": [],
            "num_birth_givers_adult": [],
            "birth_givers_adult": [],
            "num_birth_givers_female_adult": [],
            "birth_givers_female_adult": [],
            "num_birth_givers_male_adult": [],
            "birth_givers_male_adult": [],
            "num_birth_givers_senior": [],
            "birth_givers_senior": [],
            "num_birth_givers_female_senior": [],
            "birth_givers_female_senior": [],
            "num_birth_givers_male_senior": [],
            "birth_givers_male_senior": [],
            "num_birth_givers_male_adultSenior": [],
            "birth_givers_male_adultSenior": [],
            "num_birth_givers_female_adultSenior": [],
            "birth_givers_female_adultSenior": [],
            "num_birth_givers_adultSenior": [],
            "birth_givers_adultSenior": [],
            "birth_givers_proven_breeders": [],
            "birth_givers_proven_breeders_male": [],
            "birth_givers_proven_breeders_female": [],
            "number_RAP": [],
            "number_RAP_male": [],
            "number_RAP_female": [],
            "RAP_adultSenior": [],
            "RAP_male_adultSenior": [],
            "RAP_female_adultSenior": [],
            "number_birth_female": [],
            "number_birth_male": [],
            "number_birth": [],
            "pyramid_type_male": [],
            "pyramid_seq_male": [],
            "pyramid_type_female": [],
            "pyramid_seq_female": [],
        }

        df_location: str = f"{config.OUTPUT_FOLDER_STATISTICS}{species}_pyramids.csv"
        if not os.path.exists(df_location):
            self.log.error(
                f"Cannot create population pyramid for {species} as {df_location} is missing."
            )
            return
        species_name: str = config.SPECIES_NAME_MAPPER_GRAPHICS[species]
        df = pd.read_csv(df_location)
        year_range: List[int] = [
            year for year in range(config.SKIP_PYRAMIDS_BEFORE_YEAR, 2024)
        ]  # sorted([y for y in list(set(df["Year"].to_list())) if y >= config.SKIP_PYRAMIDS_BEFORE_YEAR])
        years_with_too_small_population: List[int] = []

        for year in year_range:
            df_year = df[df["Year"] == year]
            n_male = int(
                np.sum(df_year[df_year["Sex"] == "male"]["Individuals"].values)
            )
            n_female = int(
                np.sum(df_year[df_year["Sex"] == "female"]["Individuals"].values)
            )

            if n_male + n_female < 30:
                years_with_too_small_population.append(year)

        min_year = config.SKIP_PYRAMIDS_BEFORE_YEAR
        while min_year in years_with_too_small_population:
            min_year += 1
        graph_year_range = [year for year in year_range if year >= min_year]

        max_age: float = max(
            df[(df["Type"] == mode) & df["Individuals"] > 0]["Age"].to_list()
        )
        max_age_absolute: float = max(
            df[(df["Type"] == "absolute") & df["Individuals"] > 0]["Age"].to_list()
        )

        fr_m_rel = fr_m / max_age_absolute
        fr_f_rel = fr_f / max_age_absolute
        lr_m_rel = lr_m / max_age_absolute
        lr_f_rel = lr_f / max_age_absolute


        number_ages: int = len(df[df["Type"] == "absolute"]["Age"].to_list())
        max_x_value = int(
            1.05 * max(df[df["Type"] == mode]["Individuals"].to_list())
        )
        y_ticks = [np.ceil(x * max_age / 5) for x in range(6)]
        x_ticks = sorted(
            [-1 * x * np.ceil(max_x_value / 4) for x in range(1, 5)]
            + [x * np.ceil(max_x_value / 4) for x in range(0, 5)]
        )
        x_labels = [int(np.abs(x)) for x in x_ticks]
        age_range = [x for x in range(0, int(max_age) + 1)]

        # pre-calculate some maximum values
        population_size_maximum = 0
        male_population_size_maximum = 0
        female_population_size_maximum = 0
        for year in year_range:
            df_year = df[df["Year"] == year]
            n_male = int(
                np.sum(df_year[df_year["Sex"] == "male"]["Individuals"].values)
            )
            n_female = int(
                np.sum(df_year[df_year["Sex"] == "female"]["Individuals"].values)
            )

            population_size_maximum = max(population_size_maximum, n_male + n_female)
            male_population_size_maximum = max(male_population_size_maximum, n_male)
            female_population_size_maximum = max(
                female_population_size_maximum, n_female
            )

        age_categories: List[str] = ["juvenile", "adult", "senior"]
        for year in tqdm(year_range, desc="Prepare pyramid of year"):
            df_year = df[df["Year"] == year]

            n_male = int(
                np.sum(df_year[df_year["Sex"] == "male"]["Individuals"].values)
            )
            n_female = int(
                np.sum(df_year[df_year["Sex"] == "female"]["Individuals"].values)
            )
            df_year = df_year[df_year["Type"] == mode]
            female_bin: Dict[str, List[float]] = {k: [] for k in age_categories}
            male_bin: Dict[str, List[float]] = {k: [] for k in age_categories}
            for age in age_range:
                if (
                    len(
                        df_year[(df_year["Sex"] == "female") & (df_year["Age"] == age)][
                            "Individuals"
                        ].to_list()
                    )
                    == 0
                ):
                    for age_cat in age_categories:
                        female_bin[age_cat].append(0)
                else:
                    val: float = max(
                        df_year[(df_year["Sex"] == "female") & (df_year["Age"] == age)][
                            "Individuals"
                        ].to_list()
                    )
                    if mode == "relative":
                        juv_thresh = fr_f_rel
                        sen_thresh = lr_f_rel
                    else:
                        juv_thresh = fr_f
                        sen_thresh = lr_f
                    if 0 <= age < juv_thresh:
                        female_bin[age_categories[0]].append(val)
                        female_bin[age_categories[1]].append(0)
                        female_bin[age_categories[2]].append(0)
                    elif juv_thresh <= age < sen_thresh:
                        female_bin[age_categories[0]].append(0)
                        female_bin[age_categories[1]].append(val)
                        female_bin[age_categories[2]].append(0)
                    else:
                        female_bin[age_categories[0]].append(0)
                        female_bin[age_categories[1]].append(0)
                        female_bin[age_categories[2]].append(val)
                if (
                    len(
                        df_year[(df_year["Sex"] == "male") & (df_year["Age"] == age)][
                            "Individuals"
                        ].to_list()
                    )
                    == 0
                ):
                    for age_cat in age_categories:
                        male_bin[age_cat].append(0)
                else:
                    val: float = -1 * max(
                        df_year[(df_year["Sex"] == "male") & (df_year["Age"] == age)][
                            "Individuals"
                        ].to_list()
                    )
                    if mode == "relative":
                        juv_thresh = fr_m_rel
                        sen_thresh = lr_m_rel
                    else:
                        juv_thresh = fr_m
                        sen_thresh = lr_m
                    if 0 <= age < juv_thresh:
                        male_bin[age_categories[0]].append(val)
                        male_bin[age_categories[1]].append(0)
                        male_bin[age_categories[2]].append(0)
                    elif juv_thresh <= age < sen_thresh:
                        male_bin[age_categories[0]].append(0)
                        male_bin[age_categories[1]].append(val)
                        male_bin[age_categories[2]].append(0)
                    else:
                        male_bin[age_categories[0]].append(0)
                        male_bin[age_categories[1]].append(0)
                        male_bin[age_categories[2]].append(val)

            # add all data to the excel (everything should be already calculated)
            if mode == "absolute":

                if (
                    len(df_year[df_year["Sex"] == "female"]) > 0
                    and len(df_year[df_year["Sex"] == "male"]) > 0
                ):

                    number_juvenile_female = df_year[df_year["Sex"] == "female"][
                        "num juvenile"
                    ].to_list()[0]
                    number_adult_female = df_year[df_year["Sex"] == "female"][
                        "num adult"
                    ].to_list()[0]
                    number_senior_female = df_year[df_year["Sex"] == "female"][
                        "num senior"
                    ].to_list()[0]
                    number_adultSenior_female = (
                        number_adult_female + number_senior_female
                    )
                    number_juvenile_male = df_year[df_year["Sex"] == "male"][
                        "num juvenile"
                    ].to_list()[0]
                    number_adult_male = df_year[df_year["Sex"] == "male"][
                        "num adult"
                    ].to_list()[0]
                    number_senior_male = df_year[df_year["Sex"] == "male"][
                        "num senior"
                    ].to_list()[0]
                    number_adultSenior_male = number_adult_male + number_senior_male
                    number_juvenile = number_juvenile_female + number_juvenile_male
                    number_adult = number_adult_female + number_adult_male
                    number_senior = number_senior_female + number_senior_male
                    number_adultSenior = (
                        number_adultSenior_female + number_adultSenior_male
                    )

                    n_male = number_juvenile_male + number_adultSenior_male
                    n_female = number_juvenile_female + number_adultSenior_female

                    overview_excel_data["species"].append(species_name)
                    overview_excel_data["year"].append(year)
                    overview_excel_data["number_individuals"].append(n_male + n_female)
                    overview_excel_data["number_male"].append(
                        n_male if year in graph_year_range else None
                    )
                    overview_excel_data["number_female"].append(
                        n_female if year in graph_year_range else None
                    )
                    overview_excel_data["perc_individuals_max_size"].append(
                        (n_male + n_female) / population_size_maximum
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["perc_male_max_size"].append(
                        n_male / male_population_size_maximum
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["perc_female_max_size"].append(
                        n_female / female_population_size_maximum
                        if year in graph_year_range
                        else None
                    )

                    overview_excel_data["number_juvenile"].append(
                        number_juvenile if year in graph_year_range else None
                    )
                    overview_excel_data["number_adult"].append(
                        number_adult if year in graph_year_range else None
                    )
                    overview_excel_data["number_senior"].append(
                        number_senior if year in graph_year_range else None
                    )
                    overview_excel_data["number_adultSenior"].append(
                        number_adultSenior if year in graph_year_range else None
                    )
                    overview_excel_data["number_female_juvenile"].append(
                        number_juvenile_female if year in graph_year_range else None
                    )
                    overview_excel_data["number_female_adult"].append(
                        number_adult_female if year in graph_year_range else None
                    )
                    overview_excel_data["number_female_senior"].append(
                        number_senior_female if year in graph_year_range else None
                    )
                    overview_excel_data["number_female_adultSenior"].append(
                        number_adultSenior_female if year in graph_year_range else None
                    )
                    overview_excel_data["number_male_juvenile"].append(
                        number_juvenile_male if year in graph_year_range else None
                    )
                    overview_excel_data["number_male_adult"].append(
                        number_adult_male if year in graph_year_range else None
                    )
                    overview_excel_data["number_male_senior"].append(
                        number_senior_male if year in graph_year_range else None
                    )
                    overview_excel_data["number_male_adultSenior"].append(
                        number_adultSenior_male if year in graph_year_range else None
                    )

                    percentage_male = (
                        n_male / (n_male + n_female) if n_male + n_female > 0 else 0
                    )
                    percentage_female = (
                        n_female / (n_male + n_female) if n_male + n_female > 0 else 0
                    )
                    percentage_juvenile = (
                        number_juvenile / (n_male + n_female)
                        if n_male + n_female > 0
                        else 0
                    )
                    percentage_adult = (
                        number_adult / (n_male + n_female)
                        if n_male + n_female > 0
                        else 0
                    )
                    percentage_senior = (
                        number_senior / (n_male + n_female)
                        if n_male + n_female > 0
                        else 0
                    )
                    percentage_adultSenior = 1 - percentage_juvenile
                    percentage_male_juvenile = (
                        number_juvenile_male / n_male if n_male > 0 else 0
                    )
                    percentage_male_adult = (
                        number_adult_male / n_male if n_male > 0 else 0
                    )
                    percentage_male_senior = (
                        number_senior_male / n_male if n_male > 0 else 0
                    )
                    percentage_male_adultSenior = 1 - percentage_male_juvenile
                    percentage_female_juvenile = (
                        number_juvenile_female / n_female if n_female > 0 else 0
                    )
                    percentage_female_adult = (
                        number_adult_female / n_female if n_female > 0 else 0
                    )
                    percentage_female_senior = (
                        number_senior_female / n_female if n_female > 0 else 0
                    )
                    percentage_female_adultSenior = 1 - percentage_female_juvenile

                    overview_excel_data["percentage_male"].append(
                        percentage_male if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_female"].append(
                        percentage_female if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_juvenile"].append(
                        percentage_juvenile if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_adult"].append(
                        percentage_adult if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_senior"].append(
                        percentage_senior if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_adultSenior"].append(
                        percentage_adultSenior if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_male_juvenile"].append(
                        percentage_male_juvenile if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_male_adult"].append(
                        percentage_male_adult if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_male_senior"].append(
                        percentage_male_senior if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_male_adultSenior"].append(
                        percentage_male_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["percentage_female_juvenile"].append(
                        percentage_female_juvenile if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_female_adult"].append(
                        percentage_female_adult if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_female_senior"].append(
                        percentage_female_senior if year in graph_year_range else None
                    )
                    overview_excel_data["percentage_female_adultSenior"].append(
                        percentage_female_adultSenior
                        if year in graph_year_range
                        else None
                    )

                    median_age = df_year["median age all sex"].to_list()[0]
                    median_age_female = df_year[df_year["Sex"] == "female"][
                        "median age"
                    ].to_list()[0]
                    mean_age_female = df_year[df_year["Sex"] == "female"][
                        "mean age"
                    ].to_list()[0]
                    median_age_male = df_year[df_year["Sex"] == "male"][
                        "median age"
                    ].to_list()[0]
                    mean_age_male = df_year[df_year["Sex"] == "male"][
                        "mean age"
                    ].to_list()[0]
                    num_birth_female = df_year[df_year["Sex"] == "female"][
                        "number of births"
                    ].to_list()[0]
                    num_birth_male = df_year[df_year["Sex"] == "male"][
                        "number of births"
                    ].to_list()[0]

                    num_breeders_male = df_year[df_year["Sex"] == "male"][
                        "num proven breeders"
                    ].to_list()[0]
                    num_breeders_female = df_year[df_year["Sex"] == "female"][
                        "num proven breeders"
                    ].to_list()[0]
                    num_breeders_all = df_year["num proven breeders all sex"].to_list()[
                        0
                    ]
                    proven_breeders_male = (
                        num_breeders_male / n_male if n_male > 0 else None
                    )
                    proven_breeders_female = (
                        num_breeders_female / n_female if n_female > 0 else None
                    )
                    proven_breeders_all = (
                        num_breeders_all / (n_female + n_male)
                        if n_female + n_male > 0
                        else None
                    )

                    num_breeders_male_senior = df_year[df_year["Sex"] == "male"][
                        "num proven breeders senior"
                    ].to_list()[0]
                    num_breeders_female_senior = df_year[df_year["Sex"] == "female"][
                        "num proven breeders senior"
                    ].to_list()[0]
                    num_proven_breeders_all_senior = df_year[
                        "num proven breeders senior all sex"
                    ].to_list()[0]
                    proven_breeders_male_senior = (
                        num_breeders_male_senior / number_senior_male
                        if number_senior_male > 0
                        else None
                    )
                    proven_breeders_female_senior = (
                        num_breeders_female_senior / number_senior_female
                        if number_senior_female > 0
                        else None
                    )
                    proven_breeders_all_senior = (
                        num_proven_breeders_all_senior
                        / (number_senior_female + number_senior_male)
                        if number_senior_female + number_senior_male > 0
                        else None
                    )

                    num_breeders_male_adult = df_year[df_year["Sex"] == "male"][
                        "num proven breeders adult"
                    ].to_list()[0]
                    num_breeders_female_adult = df_year[df_year["Sex"] == "female"][
                        "num proven breeders adult"
                    ].to_list()[0]
                    num_proven_breeders_all_adult = df_year[
                        "num proven breeders adult all sex"
                    ].to_list()[0]
                    proven_breeders_male_adult = (
                        num_breeders_male_adult / number_adult_male
                        if number_adult_male > 0
                        else None
                    )
                    proven_breeders_female_adult = (
                        num_breeders_female_adult / number_adult_female
                        if number_adult_female > 0
                        else None
                    )
                    proven_breeders_all_adult = (
                        num_proven_breeders_all_adult / number_adult
                        if number_adult > 0
                        else None
                    )

                    num_breeders_male_adultSenior = df_year[df_year["Sex"] == "male"][
                        "num proven breeders adultSenior"
                    ].to_list()[0]
                    num_breeders_female_adultSenior = df_year[
                        df_year["Sex"] == "female"
                    ]["num proven breeders adultSenior"].to_list()[0]
                    num_proven_breeders_all_adultSenior = df_year[
                        "num proven breeders adultSenior all sex"
                    ].to_list()[0]
                    proven_breeders_female_adultSenior = (
                        num_breeders_female_adultSenior / number_adultSenior_female
                        if number_adultSenior_female > 0
                        else None
                    )
                    proven_breeders_male_adultSenior = (
                        num_breeders_male_adultSenior / number_adultSenior_male
                        if number_adultSenior_male > 0
                        else None
                    )
                    proven_breeders_all_adultSenior = (
                        num_proven_breeders_all_adultSenior / number_adultSenior
                        if number_adultSenior > 0
                        else None
                    )

                    num_birth_givers_female = df_year[df_year["Sex"] == "female"][
                        "number of birth-givers"
                    ].to_list()[0]
                    num_birth_givers_male = df_year[df_year["Sex"] == "male"][
                        "number of birth-givers"
                    ].to_list()[0]
                    num_birth_givers_female_adult = df_year[df_year["Sex"] == "female"][
                        "number of birth-givers adult"
                    ].to_list()[0]
                    num_birth_givers_male_adult = df_year[df_year["Sex"] == "male"][
                        "number of birth-givers adult"
                    ].to_list()[0]
                    num_birth_givers_female_senior = df_year[
                        df_year["Sex"] == "female"
                    ]["number of birth-givers senior"].to_list()[0]
                    num_birth_givers_male_senior = df_year[df_year["Sex"] == "male"][
                        "number of birth-givers senior"
                    ].to_list()[0]

                    prop_birth_givers_female = (
                        num_birth_givers_female / number_adultSenior_female
                        if number_adultSenior_female > 0
                        else None
                    )
                    prop_birth_givers_male = (
                        num_birth_givers_male / number_adultSenior_male
                        if number_adultSenior_male > 0
                        else None
                    )
                    prop_birth_givers_female_adult = (
                        num_birth_givers_female_adult / number_adult_female
                        if number_adult_female > 0
                        else None
                    )
                    prop_birth_givers_male_adult = (
                        num_birth_givers_male_adult / number_adult_male
                        if number_adult_male > 0
                        else None
                    )
                    prop_birth_givers_female_senior = (
                        num_birth_givers_female_senior / number_senior_female
                        if number_senior_female > 0
                        else None
                    )
                    prop_birth_givers_male_senior = (
                        num_birth_givers_male_senior / number_senior_male
                        if number_senior_male > 0
                        else None
                    )

                    num_birth_givers_adult = (
                        num_birth_givers_female_adult + num_birth_givers_male_adult
                    )
                    prop_birth_givers_adult = (
                        num_birth_givers_adult / number_adult
                        if number_adult > 0
                        else None
                    )
                    num_birth_givers_senior = (
                        num_birth_givers_female_senior + num_birth_givers_male_senior
                    )
                    prop_birth_givers_senior = (
                        num_birth_givers_senior / number_senior
                        if number_senior > 0
                        else None
                    )
                    num_birth_givers_adultSenior = (
                        num_birth_givers_adult + num_birth_givers_senior
                    )
                    prop_birth_givers_adultSenior = (
                        num_birth_givers_adultSenior / number_adultSenior
                        if number_adultSenior > 0
                        else None
                    )
                    num_birth_givers = num_birth_givers_adult + num_birth_givers_senior
                    prop_birth_givers = (
                        num_birth_givers / (number_adultSenior + number_juvenile)
                        if number_adultSenior + number_juvenile > 0
                        else None
                    )
                    num_birth_givers_female_adultSenior = (
                        num_birth_givers_female_adult + num_birth_givers_female_senior
                    )
                    num_birth_givers_male_adultSenior = (
                        num_birth_givers_male_adult + num_birth_givers_male_senior
                    )
                    prop_birth_givers_female_adultSenior = (
                        num_birth_givers_female_adultSenior / number_adultSenior_female
                        if number_adultSenior_female > 0
                        else None
                    )
                    prop_birth_givers_male_adultSenior = (
                        num_birth_givers_male_adultSenior / number_adultSenior_male
                        if number_adultSenior_male > 0
                        else None
                    )

                    prop_birth_givers_of_breeders = (
                        num_birth_givers / num_breeders_all
                        if num_breeders_all > 0
                        else None
                    )
                    prop_birth_givers_of_breeders_female = (
                        num_birth_givers_female / num_breeders_female
                        if num_breeders_female > 0
                        else None
                    )
                    prop_birth_givers_of_breeders_male = (
                        num_birth_givers_male / num_breeders_male
                        if num_breeders_male > 0
                        else None
                    )

                    mean_age = (
                        (n_male * mean_age_male + n_female * mean_age_female)
                        / (n_male + n_female)
                        if n_male + n_female > 0
                        else None
                    )
                    mean_age_perc_longevity = (
                        mean_age / config.MAXIMUM_LONGEVITY_BY_SPECIES[species]["all"]
                        if mean_age
                        else None
                    )
                    mean_age_female_perc_longevity = (
                        mean_age_female
                        / config.MAXIMUM_LONGEVITY_BY_SPECIES[species]["female"]
                        if mean_age_female
                        else None
                    )
                    mean_age_male_perc_longevity = (
                        mean_age_male
                        / config.MAXIMUM_LONGEVITY_BY_SPECIES[species]["male"]
                        if mean_age_male
                        else None
                    )
                    median_age_perc_longevity = (
                        median_age / config.MAXIMUM_LONGEVITY_BY_SPECIES[species]["all"]
                        if median_age
                        else None
                    )
                    median_age_female_perc_longevity = (
                        median_age_female
                        / config.MAXIMUM_LONGEVITY_BY_SPECIES[species]["female"]
                        if median_age_female
                        else None
                    )
                    median_age_male_perc_longevity = (
                        median_age_male
                        / config.MAXIMUM_LONGEVITY_BY_SPECIES[species]["male"]
                        if median_age_male
                        else None
                    )

                    overview_excel_data["mean_age_female"].append(
                        mean_age_female if year in graph_year_range else None
                    )
                    overview_excel_data["mean_age_male"].append(
                        mean_age_male if year in graph_year_range else None
                    )
                    overview_excel_data["mean_age"].append(
                        mean_age if year in graph_year_range else None
                    )
                    overview_excel_data["median_age"].append(
                        median_age if year in graph_year_range else None
                    )
                    overview_excel_data["median_age_female"].append(
                        median_age_female if year in graph_year_range else None
                    )
                    overview_excel_data["median_age_male"].append(
                        median_age_male if year in graph_year_range else None
                    )
                    overview_excel_data["mean_age_perc_longevity"].append(
                        mean_age_perc_longevity if year in graph_year_range else None
                    )
                    overview_excel_data["median_age_perc_longevity"].append(
                        median_age_perc_longevity if year in graph_year_range else None
                    )
                    overview_excel_data["mean_age_male_perc_longevity"].append(
                        mean_age_male_perc_longevity
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["median_age_male_perc_longevity"].append(
                        median_age_male_perc_longevity
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["mean_age_female_perc_longevity"].append(
                        mean_age_female_perc_longevity
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["median_age_female_perc_longevity"].append(
                        median_age_female_perc_longevity
                        if year in graph_year_range
                        else None
                    )

                    overview_excel_data["num_proven_breeders_male"].append(
                        num_breeders_male if year in graph_year_range else None
                    )
                    overview_excel_data["proven_breeders_male"].append(
                        proven_breeders_male if year in graph_year_range else None
                    )
                    overview_excel_data["num_proven_breeders_male_adult"].append(
                        num_breeders_male_adult if year in graph_year_range else None
                    )
                    overview_excel_data["proven_breeders_male_adult"].append(
                        proven_breeders_male_adult if year in graph_year_range else None
                    )
                    overview_excel_data["num_proven_breeders_male_senior"].append(
                        num_breeders_male_senior if year in graph_year_range else None
                    )
                    overview_excel_data["proven_breeders_male_senior"].append(
                        proven_breeders_male_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_proven_breeders_male_adultSenior"].append(
                        num_breeders_male_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["proven_breeders_male_adultSenior"].append(
                        proven_breeders_male_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_proven_breeders_female"].append(
                        num_breeders_female if year in graph_year_range else None
                    )
                    overview_excel_data["proven_breeders_female"].append(
                        proven_breeders_female if year in graph_year_range else None
                    )
                    overview_excel_data["num_proven_breeders_female_adult"].append(
                        num_breeders_female_adult if year in graph_year_range else None
                    )
                    overview_excel_data["proven_breeders_female_adult"].append(
                        proven_breeders_female_adult
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_proven_breeders_female_senior"].append(
                        num_breeders_female_senior if year in graph_year_range else None
                    )
                    overview_excel_data["proven_breeders_female_senior"].append(
                        proven_breeders_female_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data[
                        "num_proven_breeders_female_adultSenior"
                    ].append(
                        num_breeders_female_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["proven_breeders_female_adultSenior"].append(
                        proven_breeders_female_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_proven_breeders"].append(
                        num_breeders_all if year in graph_year_range else None
                    )
                    overview_excel_data["proven_breeders"].append(
                        proven_breeders_all if year in graph_year_range else None
                    )
                    overview_excel_data["num_proven_breeders_adult"].append(
                        num_proven_breeders_all_adult
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["proven_breeders_adult"].append(
                        proven_breeders_all_adult if year in graph_year_range else None
                    )
                    overview_excel_data["num_proven_breeders_senior"].append(
                        num_proven_breeders_all_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["proven_breeders_senior"].append(
                        proven_breeders_all_senior if year in graph_year_range else None
                    )
                    overview_excel_data["num_proven_breeders_adultSenior"].append(
                        num_proven_breeders_all_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["proven_breeders_adultSenior"].append(
                        proven_breeders_all_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_proven_breeders"].append(
                        prop_birth_givers_of_breeders
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_proven_breeders_male"].append(
                        prop_birth_givers_of_breeders_male
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_proven_breeders_female"].append(
                        prop_birth_givers_of_breeders_female
                        if year in graph_year_range
                        else None
                    )

                    number_RAP = num_proven_breeders_all_adult + num_birth_givers_senior
                    number_RAP_male = (
                        num_breeders_male_adult + num_birth_givers_male_senior
                    )
                    number_RAP_female = (
                        num_breeders_female_adult + num_birth_givers_female_senior
                    )
                    prop_RAP_adultSenior = (
                        number_RAP / number_adultSenior
                        if number_adultSenior > 0
                        else None
                    )
                    prop_RAP_male_adultSenior = (
                        number_RAP_male / number_adultSenior_male
                        if number_adultSenior_male > 0
                        else None
                    )
                    prop_RAP_female_adultSenior = (
                        number_RAP_female / number_adultSenior_female
                        if number_adultSenior_female > 0
                        else None
                    )

                    overview_excel_data["number_RAP"].append(
                        number_RAP if year in graph_year_range else None
                    )
                    overview_excel_data["number_RAP_male"].append(
                        number_RAP_male if year in graph_year_range else None
                    )
                    overview_excel_data["number_RAP_female"].append(
                        number_RAP_female if year in graph_year_range else None
                    )
                    overview_excel_data["RAP_adultSenior"].append(
                        prop_RAP_adultSenior if year in graph_year_range else None
                    )
                    overview_excel_data["RAP_male_adultSenior"].append(
                        prop_RAP_male_adultSenior if year in graph_year_range else None
                    )
                    overview_excel_data["RAP_female_adultSenior"].append(
                        prop_RAP_female_adultSenior
                        if year in graph_year_range
                        else None
                    )

                    overview_excel_data["number_birth_female"].append(
                        num_birth_female if year in graph_year_range else None
                    )
                    overview_excel_data["number_birth_male"].append(
                        num_birth_male if year in graph_year_range else None
                    )
                    overview_excel_data["number_birth"].append(
                        num_birth_male + num_birth_female
                        if year in graph_year_range
                        else None
                    )

                    overview_excel_data["num_birth_givers_female"].append(
                        num_birth_givers_female if year in graph_year_range else None
                    )
                    overview_excel_data["num_birth_givers_male"].append(
                        num_birth_givers_male if year in graph_year_range else None
                    )
                    overview_excel_data["num_birth_givers"].append(
                        num_birth_givers_female + num_birth_givers_male
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_female"].append(
                        prop_birth_givers_female if year in graph_year_range else None
                    )
                    overview_excel_data["birth_givers_male"].append(
                        prop_birth_givers_male if year in graph_year_range else None
                    )
                    overview_excel_data["birth_givers"].append(
                        prop_birth_givers if year in graph_year_range else None
                    )
                    overview_excel_data["num_birth_givers_female_adult"].append(
                        num_birth_givers_female_adult
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_birth_givers_male_adult"].append(
                        num_birth_givers_male_adult
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_birth_givers_adult"].append(
                        num_birth_givers_adult if year in graph_year_range else None
                    )
                    overview_excel_data["birth_givers_female_adult"].append(
                        prop_birth_givers_female_adult
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_male_adult"].append(
                        prop_birth_givers_male_adult
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_adult"].append(
                        prop_birth_givers_adult if year in graph_year_range else None
                    )
                    overview_excel_data["num_birth_givers_female_senior"].append(
                        num_birth_givers_female_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_birth_givers_male_senior"].append(
                        num_birth_givers_male_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_birth_givers_senior"].append(
                        num_birth_givers_female_senior + num_birth_givers_male_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_female_senior"].append(
                        prop_birth_givers_female_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_male_senior"].append(
                        prop_birth_givers_male_senior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_senior"].append(
                        prop_birth_givers_senior if year in graph_year_range else None
                    )

                    overview_excel_data["num_birth_givers_female_adultSenior"].append(
                        num_birth_givers_female_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_birth_givers_male_adultSenior"].append(
                        num_birth_givers_male_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["num_birth_givers_adultSenior"].append(
                        num_birth_givers_female_adultSenior
                        + num_birth_givers_male_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_female_adultSenior"].append(
                        prop_birth_givers_female_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_male_adultSenior"].append(
                        prop_birth_givers_male_adultSenior
                        if year in graph_year_range
                        else None
                    )
                    overview_excel_data["birth_givers_adultSenior"].append(
                        prop_birth_givers_adultSenior
                        if year in graph_year_range
                        else None
                    )

                    df_male = df_year[df_year["Sex"] == "male"]
                    (
                        classification_seq,
                        classification_m,
                        age_borders_m,
                        bucket_stats_m,
                    ) = self.get_population_evaluation(
                        df=df_male,
                        max_age=int(max_age_absolute),
                        juvenile_thresh=fr_m,
                        senior_thresh=lr_m,
                    )
                    classification_male_write = (
                        classification_m
                        if year in graph_year_range
                        and n_male >= 10
                        and number_senior_male < n_male
                        else None
                    )
                    classification_sequence_male_write = (
                        str(classification_seq)
                        if year in graph_year_range
                        and n_male >= 10
                        and number_senior_male < n_male
                        else None
                    )
                    overview_excel_data["pyramid_type_male"].append(
                        classification_male_write
                    )
                    overview_excel_data["pyramid_seq_male"].append(
                        classification_sequence_male_write
                    )

                    df_female = df_year[df_year["Sex"] == "female"]
                    (
                        classification_seq,
                        classification_f,
                        age_borders_f,
                        bucket_stats_f,
                    ) = self.get_population_evaluation(
                        df=df_female,
                        max_age=int(max_age_absolute),
                        juvenile_thresh=fr_f,
                        senior_thresh=lr_f,
                    )
                    classification_female_write = (
                        classification_f
                        if year in graph_year_range
                        and n_female >= 10
                        and number_senior_female < n_female
                        else None
                    )
                    classification_sequence_female_write = (
                        str(classification_seq)
                        if year in graph_year_range
                        and n_female >= 10
                        and number_senior_female < n_female
                        else None
                    )
                    overview_excel_data["pyramid_type_female"].append(
                        classification_female_write
                    )
                    overview_excel_data["pyramid_seq_female"].append(
                        classification_sequence_female_write
                    )

                else:
                    df_male = df_year[df_year["Sex"] == "male"]
                    (
                        classification_seq,
                        classification_m,
                        age_borders_m,
                        bucket_stats_m,
                    ) = self.get_population_evaluation(
                        df=df_male,
                        max_age=int(max_age_absolute),
                        juvenile_thresh=fr_m,
                        senior_thresh=lr_m,
                    )

                    df_female = df_year[df_year["Sex"] == "female"]
                    (
                        classification_seq,
                        classification_f,
                        age_borders_f,
                        bucket_stats_f,
                    ) = self.get_population_evaluation(
                        df=df_female,
                        max_age=int(max_age_absolute),
                        juvenile_thresh=fr_f,
                        senior_thresh=lr_f,
                    )

            if year in graph_year_range:
                fig = go.Figure()

                for age_cat in reversed(age_categories):
                    fig.add_trace(
                        go.Bar(
                            y=age_range,
                            x=male_bin[age_cat],
                            name=f"{age_cat}",
                            orientation="h",
                            marker=dict(
                                color=config.SEX_AGE_COLOR_MAP[f"male_{age_cat}"]
                            ),
                            text=-1 * male_bin[age_cat],
                            legendgroup="male",
                            legendgrouptitle_text="male",
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            y=age_range,
                            x=female_bin[age_cat],
                            name=f"{age_cat}",
                            orientation="h",
                            marker=dict(
                                color=config.SEX_AGE_COLOR_MAP[f"female_{age_cat}"]
                            ),
                            legendgroup="female",
                            legendgrouptitle_text="female",
                        )
                    )

                fig.update_layout(
                    title=f"<i>{species_name}</i> {int(year)}",
                    barmode="relative",
                    bargap=0.0,
                    bargroupgap=0,
                    xaxis=dict(
                        range=[-1 * max_x_value, max_x_value],
                        tickvals=x_ticks,
                        ticktext=x_labels,
                        title=(
                            "proportion"
                            if mode == "relative"
                            else "number of individuals"
                        ),
                    ),
                    yaxis=dict(
                        tickvals=y_ticks,
                        ticktext=y_ticks,
                        title="age",
                    ),
                )

                fig = fig.update_layout(
                    {
                        "plot_bgcolor": "rgb(255, 255, 255)",
                        "paper_bgcolor": "rgb(255, 255, 255)",
                        "title_x": 0.5,
                    }
                )
                fig.update_layout(
                    legend=dict(
                        orientation="v",
                        yanchor="bottom",
                        y=0.48,
                        xanchor="right",
                        x=1.075,
                    ),
                )
                fig.update_yaxes(
                    showgrid=False,
                    gridcolor=config.GRIDCOLOR,
                    showline=True,
                    linecolor=config.GRIDCOLOR,
                )
                fig.update_xaxes(
                    showgrid=True,
                    gridcolor=config.GRIDCOLOR,
                    showline=True,
                    linecolor=config.GRIDCOLOR,
                    zeroline=False,
                )
                fig.add_vline(x=0, line_width=1, line_color=config.GRIDCOLOR)

                if mode == "absolute":
                    male_mean_values = []
                    male_mean_error = []
                    for age in age_range:
                        curr_bucket_stat = (0, 0)
                        for j in range(1, len(age_borders_m)):
                            if age >= age_borders_m[j - 1] and age <= age_borders_m[j]:
                                curr_bucket_stat = bucket_stats_m[j - 1]
                        male_mean_values.append(curr_bucket_stat[0])
                        male_mean_error.append(curr_bucket_stat[1])

                    female_mean_values = []
                    female_mean_error = []
                    for age in age_range:
                        curr_bucket_stat = (0, 0)
                        for j in range(1, len(age_borders_f)):
                            if age >= age_borders_f[j - 1] and age <= age_borders_f[j]:
                                curr_bucket_stat = bucket_stats_f[j - 1]
                        female_mean_values.append(curr_bucket_stat[0])
                        female_mean_error.append(curr_bucket_stat[1])

                    fig.add_trace(
                        go.Scatter(
                            y=age_range,
                            x=-1 * np.array(male_mean_values),
                            error_x=dict(
                                array=config.PYRAMID_SEM_DISTINGUISH_FACTOR
                                * np.array(male_mean_error)
                            ),
                            name=None,
                            orientation="h",
                            marker=dict(color="black"),
                            text=-1 * np.array(male_mean_values),
                            legendgroup=None,
                            legendgrouptitle_text=None,
                            showlegend=False,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=age_range,
                            x=female_mean_values,
                            error_x=dict(
                                array=config.PYRAMID_SEM_DISTINGUISH_FACTOR
                                * np.array(female_mean_error)
                            ),
                            name=None,
                            orientation="h",
                            marker=dict(color="black"),
                            text=1 * np.array(female_mean_values),
                            legendgroup=None,
                            legendgrouptitle_text=None,
                            showlegend=False,
                        )
                    )

                    # if len(overview_excel_data["median_age"]) and overview_excel_data["median_age"][-1]:
                    #    fig.add_hline(y=overview_excel_data["median_age"][-1], line_width=1.5, line_dash="dot",
                    #                  line_color=config.SEX_COLOR_MAP["all"])
                    if (
                        len(overview_excel_data["median_age_female"])
                        and overview_excel_data["median_age_female"][-1]
                        and not np.isnan(overview_excel_data["median_age_female"][-1])
                    ):
                        fig.add_hline(
                            y=overview_excel_data["median_age_female"][-1],
                            line_width=1.5,
                            line_dash="dot",
                            line_color=config.SEX_COLOR_MAP["female"],
                        )
                    if (
                        len(overview_excel_data["median_age_male"])
                        and overview_excel_data["median_age_male"][-1]
                        and not np.isnan(overview_excel_data["median_age_male"][-1])
                    ):
                        fig.add_hline(
                            y=overview_excel_data["median_age_male"][-1],
                            line_width=1.5,
                            line_dash="dot",
                            line_color=config.SEX_COLOR_MAP["male"],
                        )
                    fig.add_annotation(
                        x=0.66 * max_x_value,
                        y=0.99 * max_age,
                        text=f"<b>{n_male + n_female}</b><br>{n_male}.{n_female}<br>f: {classification_female_write if classification_female_write is not None else '---'}<br>m: {classification_male_write if classification_male_write is not None else '---'}",
                        showarrow=False,
                    )

                save_path: str = (
                    f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/pyramids/{species}_Population-Pyramid_{mode}_{int(year)}_{config.REGION_STRING_PYRAMID_NAME}.png"
                )
                save_path_svg: str = (
                    f"{config.OUTPUT_FOLDER_VISUALISATION}{species}/pyramids/{species}_Population-Pyramid_{mode}_{int(year)}_{config.REGION_STRING_PYRAMID_NAME}.svg"
                )
                ensure_diretory(save_path)
                scale = (config.WIDTH_PYRAMID_IMAGE_MM / 25.4) / (
                    700 / config.FIGURE_DPI
                )

                fig.write_image(save_path, scale=scale)
                if config.SAVE_ADDITIONALLY_AS_VECTORGRAPHIC:
                    fig.write_image(save_path_svg, scale=scale)

        self.concatenate_pyramids(species, mode)
        if mode == "absolute":
            ensure_diretory(config.OUTPUT_FOLDER_PYRAMID_TMP_STAT)
            pd.DataFrame(overview_excel_data).to_excel(
                f"{config.OUTPUT_FOLDER_PYRAMID_TMP_STAT}{species}_Population-Pyramid_statistics_{config.REGION_STRING_PYRAMID_NAME}.xlsx",
                index=False,
            )

            fig1 = go.Figure()
            fig1.add_trace(
                go.Scatter(
                    y=overview_excel_data["number_individuals"],
                    x=overview_excel_data["year"],
                    name=f"total",
                    line_color=config.SEX_COLOR_MAP["all"],
                    legendgroup="sex",
                )
            )
            fig1.add_trace(
                go.Scatter(
                    y=overview_excel_data["number_male"],
                    x=overview_excel_data["year"],
                    name=f"male",
                    line_color=config.SEX_COLOR_MAP["male"],
                    legendgroup="sex",
                )
            )
            fig1.add_trace(
                go.Scatter(
                    y=overview_excel_data["number_female"],
                    x=overview_excel_data["year"],
                    name=f"female",
                    line_color=config.SEX_COLOR_MAP["female"],
                    legendgroup="sex",
                )
            )
            fig1.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
                ),
                xaxis=dict(
                    title="year",
                ),
                yaxis=dict(
                    title="number of individuals",
                ),
            )

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    y=overview_excel_data["percentage_juvenile"],
                    x=overview_excel_data["year"],
                    name=f"juvenile",
                    line_color=config.SEX_COLOR_MAP["all"],
                    legendgroup="age",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    y=overview_excel_data["percentage_adult"],
                    x=overview_excel_data["year"],
                    name=f"adult",
                    line_color=config.SEX_COLOR_MAP["male"],
                    legendgroup="age",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    y=overview_excel_data["percentage_senior"],
                    x=overview_excel_data["year"],
                    name=f"senior",
                    line_color=config.SEX_COLOR_MAP["female"],
                    legendgroup="age",
                )
            )
            fig2.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
                ),
                xaxis=dict(
                    title="year",
                ),
                yaxis=dict(
                    title="percentage of individuals",
                ),
            )

            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    y=overview_excel_data["median_age_female"],
                    x=overview_excel_data["year"],
                    name=None,
                    line_color=config.SEX_COLOR_MAP["female"],
                    legendgroup=None,
                    showlegend=False,
                )
            )
            fig3.add_trace(
                go.Scatter(
                    y=overview_excel_data["median_age_male"],
                    x=overview_excel_data["year"],
                    name=None,
                    line_color=config.SEX_COLOR_MAP["male"],
                    legendgroup=None,
                    showlegend=False,
                )
            )
            fig3.add_trace(
                go.Scatter(
                    y=overview_excel_data["median_age"],
                    x=overview_excel_data["year"],
                    name=None,
                    line_color=config.SEX_COLOR_MAP["all"],
                    legendgroup=None,
                    showlegend=False,
                )
            )

            fig3.update_layout(
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5
                ),
                xaxis=dict(
                    title="year",
                ),
                yaxis=dict(
                    title="median age",
                ),
            )

            fig = make_subplots(
                rows=1,
                cols=3,
                shared_xaxes=True,
                subplot_titles=(
                    "number of individuals",
                    "percentage of age groups",
                    "median population age",
                ),
            )

            # add each trace (or traces) to its specific subplot
            for i in fig1.data:
                fig.add_trace(i, row=1, col=1)

            for i in fig2.data:
                fig.add_trace(i, row=1, col=2)

            for i in fig3.data:
                fig.add_trace(i, row=1, col=3)

            fig.update_yaxes(
                showgrid=True, gridcolor=config.GRIDCOLOR, linecolor=config.GRIDCOLOR
            )
            fig.update_xaxes(
                showgrid=False,
                showline=True,
                linecolor=config.GRIDCOLOR,
                titlefont=dict(size=10),
                tickfont=dict(size=10),
            )

            fig.update_layout(
                legend=dict(font=dict(size=10)), legend_title=dict(font=dict(size=10))
            )

            fig = fig.update_layout(
                {
                    "plot_bgcolor": "rgb(255, 255, 255)",
                    "paper_bgcolor": "rgb(255, 255, 255)",
                    "title_x": 0.5,
                }
            )
            fig.update_layout(
                title=f"<i>{species_name}</i> {config.REGION_STRING_PYRAMID_NAME}",
            )
            fig.update_annotations(font_size=10)

            taxonomic_group_name = config.TAXONOMIC_GROUP_BY_SPECIES[species]
            output_folder = f"{config.OUTPUT_FOLDER_PYRAMID_OVERVIEW}graphs_stats/"

            save_path: str = (
                f"{output_folder}{taxonomic_group_name}_{config.REGION_STRING_PYRAMID_NAME}_{species}.png"
            )
            save_path_svg: str = (
                f"{output_folder}{taxonomic_group_name}_{config.REGION_STRING_PYRAMID_NAME}_{species}.svg"
            )
            ensure_diretory(save_path)
            scale = (config.WIDTH_PYRAMID_IMAGE_MM / 25.4) / (700 / config.FIGURE_DPI)
            fig.write_image(save_path, scale=scale)
            if config.SAVE_ADDITIONALLY_AS_VECTORGRAPHIC:
                fig.write_image(save_path_svg, scale=scale)

        if len(years_with_too_small_population) == 0:
            return False
        curr_year = config.SKIP_PYRAMIDS_BEFORE_YEAR - 1
        for year in years_with_too_small_population:
            if year - curr_year >= 2:
                return True
            curr_year = year
        return False

    def visualize_pyramid_classification(self, species: str):

        def sublist_count(test_list: List[str], sublist: List[str]) -> int:
            return len(
                [
                    sublist
                    for idx in range(len(test_list))
                    if test_list[idx : idx + len(sublist)] == sublist
                ]
            )

        def sublist_indices(test_list: List[str], sublist: List[str]) -> List[int]:
            return [
                idx
                for idx in range(len(test_list))
                if test_list[idx : idx + len(sublist)] == sublist
            ]

        def get_run_lengths(
            input_list: List[str],
        ) -> Dict[str, Dict[int, Tuple[int, int]]]:
            length_encoding = {}
            n = len(input_list)
            i = 0
            while i < n - 1:
                count = 1
                while i < n - 1 and input_list[i] == input_list[i + 1]:
                    count += 1
                    i += 1
                i += 1

                curr_shape = input_list[i - 1]
                start_index = i - count
                end_index = i - 1

                if not pd.isna(curr_shape):
                    if not curr_shape in length_encoding:
                        length_encoding[curr_shape] = {}
                    if not count in length_encoding[curr_shape]:
                        length_encoding[curr_shape][count] = []
                    length_encoding[curr_shape][count].append((start_index, end_index))

            return length_encoding

        # pre_calculate population size
        df_location: str = (
            f"{config.OUTPUT_FOLDER_PYRAMID_TMP_STAT}{species}_Population-Pyramid_statistics_{config.REGION_STRING_PYRAMID_NAME}.xlsx"
        )
        if not os.path.exists(df_location):
            self.log.error(
                f"Cannot create population pyramid for {species} as {df_location} is missing."
            )
            return

        df: pd.DataFrame = pd.read_excel(
            f"{config.OUTPUT_FOLDER_PYRAMID_TMP_STAT}{species}_Population-Pyramid_statistics_{config.REGION_STRING_PYRAMID_NAME}.xlsx"
        )
        df = df[df["year"] >= config.SKIP_PYRAMIDS_BEFORE_YEAR]

        population_size_maximum = max(df["number_individuals"].to_list())
        male_population_size_maximum = max(df["number_male"].to_list())
        female_population_size_maximum = max(df["number_female"].to_list())

        population_overview = {
            "species": [],
            "sex": [],
            "year of maximum population": [],
            "pyramid shape at maximum population": [],
            "current population": [],
        }

        pyramid_type_overview = {
            "species": [],
            "sex": [],
            "pyramid_type": [],
            "pyramid_count": [],
            "mean number of individuals": [],
            "sd number of individuals": [],
            "min number of individuals": [],
            "max number of individuals": [],
            "relative mean number of individuals": [],
            "relative sd number of individuals": [],
            "relative min number of individuals": [],
            "relative max number of individuals": [],
        }

        pyramid_type_run = {
            "species": [],
            "sex": [],
            "pyramid_type": [],
            "run_length": [],
            "run_length_count": [],
            "mean difference population size": [],
            "sd difference population size": [],
            "relative mean difference population size": [],
            "relative sd difference population size": [],
        }

        # pyramid_shape_matrices
        pyramid_types = [
            PyramidTypes.PYRAMID,
            PyramidTypes.I_PYRAMID,
            PyramidTypes.HOURGLASS,
            PyramidTypes.PLUNGER,
            PyramidTypes.I_PLUNGER,
            PyramidTypes.COL,
            PyramidTypes.BELL,
            PyramidTypes.I_BELL,
            PyramidTypes.L_DIAMOND,
            PyramidTypes.M_DIAMOND,
            PyramidTypes.U_DIAMOND,
        ]

        # values into population_overview
        idx_combined = df["number_individuals"].idxmax()
        idx_male = df["number_male"].idxmax()
        idx_female = df["number_female"].idxmax()
        idx_current_year = df["year"].idxmax()

        population_overview["species"].append(species)
        population_overview["sex"].append("combined")
        population_overview["year of maximum population"].append(
            df.loc[idx_combined, "year"]
        )
        population_overview["pyramid shape at maximum population"].append(
            f'{df.loc[idx_combined, "pyramid_type_male"]}-{df.loc[idx_combined, "pyramid_type_female"]}'
        )
        population_overview["current population"].append(
            df.loc[idx_current_year, "number_individuals"]
            / df.loc[idx_combined, "number_individuals"]
        )

        population_overview["species"].append(species)
        population_overview["sex"].append("male")
        population_overview["year of maximum population"].append(
            df.loc[idx_male, "year"]
        )
        population_overview["pyramid shape at maximum population"].append(
            f'{df.loc[idx_male, "pyramid_type_male"]}'
        )
        population_overview["current population"].append(
            df.loc[idx_current_year, "number_male"] / df.loc[idx_male, "number_male"]
        )

        population_overview["species"].append(species)
        population_overview["sex"].append("female")
        population_overview["year of maximum population"].append(
            df.loc[idx_female, "year"]
        )
        population_overview["pyramid shape at maximum population"].append(
            f'{df.loc[idx_female, "pyramid_type_female"]}'
        )
        population_overview["current population"].append(
            df.loc[idx_current_year, "number_female"]
            / df.loc[idx_male, "number_female"]
        )

        # values into pyramid_type_overview
        for sex in ["male", "female"]:
            male_types = pyramid_types if sex != "female" else ["XXXX"]
            for i in range(len(male_types)):
                female_types = pyramid_types if sex != "male" else ["XXXX"]
                for j in range(len(female_types)):
                    t_male = str(male_types[i])
                    t_female = str(female_types[j])

                    if sex == "combined":
                        individual_num = df[
                            (df["pyramid_type_male"] == t_male)
                            & (df["pyramid_type_female"] == t_female)
                        ]["number_individuals"].to_list()
                        pyram_shape = f"{t_male}-{t_female}"
                        pop_size_norm = population_size_maximum
                    elif sex == "female":
                        individual_num = df[df["pyramid_type_female"] == t_female][
                            "number_female"
                        ].to_list()
                        pyram_shape = f"{t_female}"
                        pop_size_norm = female_population_size_maximum
                    else:
                        individual_num = df[df["pyramid_type_male"] == t_male][
                            "number_male"
                        ].to_list()
                        pyram_shape = f"{t_male}"
                        pop_size_norm = male_population_size_maximum

                    pyramid_type_overview["species"].append(species)
                    pyramid_type_overview["sex"].append(sex)
                    pyramid_type_overview["pyramid_type"].append(pyram_shape)
                    pyramid_type_overview["pyramid_count"].append(len(individual_num))
                    pyramid_type_overview["mean number of individuals"].append(
                        np.mean(individual_num) if len(individual_num) > 0 else None
                    )
                    pyramid_type_overview["sd number of individuals"].append(
                        np.std(individual_num) if len(individual_num) > 0 else None
                    )
                    pyramid_type_overview["min number of individuals"].append(
                        np.min(individual_num) if len(individual_num) > 0 else None
                    )
                    pyramid_type_overview["max number of individuals"].append(
                        np.max(individual_num) if len(individual_num) > 0 else None
                    )

                    pyramid_type_overview["relative mean number of individuals"].append(
                        np.mean(individual_num) / pop_size_norm
                        if len(individual_num) > 0 and pop_size_norm > 0
                        else None
                    )
                    pyramid_type_overview["relative sd number of individuals"].append(
                        np.std(individual_num) / pop_size_norm
                        if len(individual_num) > 0 and pop_size_norm > 0
                        else None
                    )
                    pyramid_type_overview["relative min number of individuals"].append(
                        np.min(individual_num) / pop_size_norm
                        if len(individual_num) > 0 and pop_size_norm > 0
                        else None
                    )
                    pyramid_type_overview["relative max number of individuals"].append(
                        np.max(individual_num) / pop_size_norm
                        if len(individual_num) > 0 and pop_size_norm > 0
                        else None
                    )

        # values into pyramid_type_run
        df = df.sort_values("year")
        male_sequence = df["pyramid_type_male"].to_list()
        male_population = df["number_male"].to_list()
        female_sequence = df["pyramid_type_female"].to_list()
        female_population = df["number_female"].to_list()

        run_lengths_male = get_run_lengths(male_sequence)
        run_lengths_female = get_run_lengths(
            female_sequence
        )  # {type: runlen : (start, end)}

        max_run_length = 2024 - config.SKIP_PYRAMIDS_BEFORE_YEAR + 1

        for i in range(len(pyramid_types)):
            t_male = str(pyramid_types[i])
            curr_type = f"{t_male}"
            if not curr_type in run_lengths_male:
                run_lengths_male[curr_type] = {}
            for run_len in range(max_run_length):
                if run_len in run_lengths_male[curr_type]:
                    start_end_tuple = list(run_lengths_male[curr_type][run_len])
                else:
                    start_end_tuple = []
                pyramid_type_run["species"].append(species)
                pyramid_type_run["sex"].append("male")
                pyramid_type_run["pyramid_type"].append(curr_type)
                pyramid_type_run["run_length"].append(run_len)
                pyramid_type_run["run_length_count"].append(len(start_end_tuple))
                pop_change = []
                pop_change_relative = []
                for start, end in start_end_tuple:
                    pop_change.append(male_population[end] - male_population[start])
                    if male_population[start] > 0:
                        pop_change_relative.append(
                            (male_population[end] - male_population[start])
                            / male_population[start]
                        )
                pyramid_type_run["mean difference population size"].append(
                    np.mean(pop_change) if len(pop_change) > 0 else None
                )
                pyramid_type_run["sd difference population size"].append(
                    np.std(pop_change) if len(pop_change) > 0 else None
                )
                pyramid_type_run["relative mean difference population size"].append(
                    np.mean(pop_change_relative)
                    if len(pop_change_relative) > 0
                    else None
                )
                pyramid_type_run["relative sd difference population size"].append(
                    np.std(pop_change_relative)
                    if len(pop_change_relative) > 0
                    else None
                )

        for i in range(len(pyramid_types)):
            t_female = str(pyramid_types[i])
            curr_type = f"{t_female}"
            if not curr_type in run_lengths_female:
                run_lengths_female[curr_type] = {}
            for run_len in range(max_run_length):
                if run_len in run_lengths_female[curr_type]:
                    start_end_tuple = list(run_lengths_female[curr_type][run_len])
                else:
                    start_end_tuple = []
                pyramid_type_run["species"].append(species)
                pyramid_type_run["sex"].append("female")
                pyramid_type_run["pyramid_type"].append(curr_type)
                pyramid_type_run["run_length"].append(run_len)
                pyramid_type_run["run_length_count"].append(len(start_end_tuple))
                pop_change = []
                pop_change_relative = []
                for start, end in start_end_tuple:
                    pop_change.append(female_population[end] - female_population[start])
                    if female_population[start] > 0:
                        pop_change_relative.append(
                            (female_population[end] - female_population[start])
                            / female_population[start]
                        )
                pyramid_type_run["mean difference population size"].append(
                    np.mean(pop_change) if len(pop_change) > 0 else None
                )
                pyramid_type_run["sd difference population size"].append(
                    np.std(pop_change) if len(pop_change) > 0 else None
                )
                pyramid_type_run["relative mean difference population size"].append(
                    np.mean(pop_change_relative)
                    if len(pop_change_relative) > 0
                    else None
                )
                pyramid_type_run["relative sd difference population size"].append(
                    np.std(pop_change_relative)
                    if len(pop_change_relative) > 0
                    else None
                )

        # make transition matrices
        sequence_overall = male_sequence + ["XXXX"] + female_sequence
        population_overall = male_population + [0.0] + female_population
        transitions = np.zeros((len(pyramid_types), len(pyramid_types)))
        transitions_male = np.zeros((len(pyramid_types), len(pyramid_types)))
        transitions_female = np.zeros((len(pyramid_types), len(pyramid_types)))
        population_shift_mean = np.zeros((len(pyramid_types), len(pyramid_types)))
        population_shift_sd = np.zeros((len(pyramid_types), len(pyramid_types)))
        population_shift_mean_male = np.zeros((len(pyramid_types), len(pyramid_types)))
        population_shift_sd_male = np.zeros((len(pyramid_types), len(pyramid_types)))
        population_shift_mean_female = np.zeros(
            (len(pyramid_types), len(pyramid_types))
        )
        population_shift_sd_female = np.zeros((len(pyramid_types), len(pyramid_types)))

        for i in range(len(pyramid_types)):
            for j in range(len(pyramid_types)):
                t1 = str(pyramid_types[i])
                t2 = str(pyramid_types[j])
                transition = [t1, t2]

                num_transitions_male = sublist_count(male_sequence, transition)
                num_transitions_female = sublist_count(female_sequence, transition)
                transitions_male[i, j] += num_transitions_male
                transitions_female[i, j] += num_transitions_female
                transitions[i, j] += num_transitions_female + num_transitions_male

                position_transition = sublist_indices(sequence_overall, transition)
                difference_population = [
                    (population_overall[pos + 1] - population_overall[pos])
                    / population_overall[pos]
                    for pos in position_transition
                    if population_overall[pos] > 0
                ]
                mean_change = (
                    np.mean(difference_population)
                    if len(difference_population) > 0
                    else None
                )
                sd_change = (
                    np.std(difference_population)
                    if len(difference_population) > 0
                    else None
                )
                population_shift_mean[i, j] = mean_change
                population_shift_sd[i, j] = sd_change

                position_transition = sublist_indices(male_sequence, transition)
                difference_population = [
                    (male_population[pos + 1] - male_population[pos])
                    / male_population[pos]
                    for pos in position_transition
                    if male_population[pos] > 0
                ]
                mean_change = (
                    np.mean(difference_population)
                    if len(difference_population) > 0
                    else None
                )
                sd_change = (
                    np.std(difference_population)
                    if len(difference_population) > 0
                    else None
                )
                population_shift_mean_male[i, j] = mean_change
                population_shift_sd_male[i, j] = sd_change

                position_transition = sublist_indices(female_sequence, transition)
                difference_population = [
                    (female_population[pos + 1] - female_population[pos])
                    / female_population[pos]
                    for pos in position_transition
                    if female_population[pos] > 0
                ]
                mean_change = (
                    np.mean(difference_population)
                    if len(difference_population) > 0
                    else None
                )
                sd_change = (
                    np.std(difference_population)
                    if len(difference_population) > 0
                    else None
                )
                population_shift_mean_female[i, j] = mean_change
                population_shift_sd_female[i, j] = sd_change

        df_transitions = pd.DataFrame(
            transitions, columns=pyramid_types, index=pyramid_types
        )
        df_pop_shift_mean = pd.DataFrame(
            population_shift_mean, columns=pyramid_types, index=pyramid_types
        )
        df_pop_shift_sd = pd.DataFrame(
            population_shift_sd, columns=pyramid_types, index=pyramid_types
        )

        df_transitions_male = pd.DataFrame(
            transitions_male, columns=pyramid_types, index=pyramid_types
        )
        df_pop_shift_mean_male = pd.DataFrame(
            population_shift_mean_male, columns=pyramid_types, index=pyramid_types
        )
        df_pop_shift_sd_male = pd.DataFrame(
            population_shift_sd_male, columns=pyramid_types, index=pyramid_types
        )

        df_transitions_female = pd.DataFrame(
            transitions_female, columns=pyramid_types, index=pyramid_types
        )
        df_pop_shift_mean_female = pd.DataFrame(
            population_shift_mean_female, columns=pyramid_types, index=pyramid_types
        )
        df_pop_shift_sd_female = pd.DataFrame(
            population_shift_sd_female, columns=pyramid_types, index=pyramid_types
        )

        df_run_len = pd.DataFrame(pyramid_type_run).sort_values(
            ["sex", "pyramid_type", "run_length"], ascending=[True, True, True]
        )

        # output excel-file
        taxonomic_group_name = config.TAXONOMIC_GROUP_BY_SPECIES[species]
        output_folder_matrices = f"{config.OUTPUT_FOLDER_PYRAMID_OVERVIEW}matrices/"
        output_folder_basedata = f"{config.OUTPUT_FOLDER_PYRAMID_OVERVIEW}stats/"

        if not os.path.exists(output_folder_matrices):
            os.makedirs(output_folder_matrices)
        if not os.path.exists(output_folder_basedata):
            os.makedirs(output_folder_basedata)

        writer = pd.ExcelWriter(
            f"{output_folder_matrices}{taxonomic_group_name}_{config.REGION_STRING_PYRAMID_NAME}_{species}.xlsx"
        )
        df_transitions.to_excel(writer, sheet_name="TransitionCount")
        df_pop_shift_mean.to_excel(writer, sheet_name="PopulationChangeMean")
        df_pop_shift_sd.to_excel(writer, sheet_name="PopulationChangeSd")
        df_transitions_male.to_excel(writer, sheet_name="TransitionCountMale")
        df_pop_shift_mean_male.to_excel(writer, sheet_name="PopulationChangeMeanMale")
        df_pop_shift_sd_male.to_excel(writer, sheet_name="PopulationChangeSdMale")
        df_transitions_female.to_excel(writer, sheet_name="TransitionCountFemale")
        df_pop_shift_mean_female.to_excel(
            writer, sheet_name="PopulationChangeMeanFemale"
        )
        df_pop_shift_sd_female.to_excel(writer, sheet_name="PopulationChangeSdFemale")
        pd.DataFrame(population_overview).to_excel(
            writer, sheet_name="PopulationOverview", index=False
        )
        pd.DataFrame(pyramid_type_overview).to_excel(
            writer, sheet_name="PyramidShapeOverview", index=False
        )
        df_run_len.to_excel(writer, sheet_name="PyramidShapeRunLength", index=False)
        writer.close()

        df.to_excel(
            f"{output_folder_basedata}{taxonomic_group_name}_{config.REGION_STRING_PYRAMID_NAME}_{species}.xlsx",
            index=False,
        )
