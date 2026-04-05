"""
---------
Clase DataLoader: carga y limpieza básica de cualquier CSV.
Usa src.utils.path para resolver rutas relativas desde la raíz del proyecto.
"""

import os
import warnings
import pandas as pd
import numpy as np
from pandas import DataFrame

from src.utils import path as path_utils

warnings.filterwarnings("ignore")


class DataLoader:
    """
    Carga un CSV y realiza limpieza básica.

    Parámetros
    ----------
    filepath : str
        Ruta relativa al CSV desde la raíz del proyecto
        (p.ej. 'data/raw/tourism.csv').
        Si se proporciona ruta absoluta, se usa directamente.
        La raíz se resuelve automáticamente con src.utils.path.
    delimiter : str
        Separador de columnas (por defecto ',').
    decimal : str
        Carácter decimal (por defecto '.').
    encoding : str
        Codificación del archivo (por defecto 'utf-8').
    project_root_name : str
        Nombre de la carpeta raíz del proyecto buscada por path_utils
        (por defecto 'ReglasAsociacion').
    """

    def __init__(self, df: pd.DataFrame,
                 project_root_name: str = "ReglasAsociacion"):

        self.df = df
        self.project_root_name = project_root_name

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    def load(self) -> "DataLoader":
        """Lee el CSV y guarda el DataFrame crudo."""
        self.df_raw = self.df.copy()
        print(f"             Dimensiones: {self.df_raw.shape[0]} filas x {self.df_raw.shape[1]} columnas")
        return self

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------
    def clean(self,
              drop_duplicates: bool = True,
              drop_na_threshold: float = 0.5,
              columns_to_keep: list[str] | None = None) -> "DataLoader":
        """
        Aplica limpieza básica.

        Parámetros
        ----------
        drop_duplicates : bool
            Elimina filas duplicadas.
        drop_na_threshold : float
            Elimina columnas cuyo % de nulos supera este umbral (0-1).
        columns_to_keep : list[str] | None
            Si se proporciona, mantiene sólo esas columnas.
        """
        if self.df_raw is None:
            raise RuntimeError("Llame a .load() antes de .clean()")

        df = self.df_raw.copy()

        # --- Selección de columnas ---
        if columns_to_keep:
            existing = [c for c in columns_to_keep if c in df.columns]
            df = df[existing]
            print(f"[DataLoader] Columnas seleccionadas: {existing}")

        # --- Duplicados ---
        before = len(df)
        if drop_duplicates:
            df = df.drop_duplicates()
            removed = before - len(df)
            print(f"[DataLoader] Duplicados eliminados: {removed}")

        # --- Columnas con demasiados nulos ---
        null_ratio = df.isnull().mean()
        cols_to_drop = null_ratio[null_ratio > drop_na_threshold].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"[DataLoader] Columnas eliminadas por nulos > {drop_na_threshold*100:.0f}%: {cols_to_drop}")

        # --- Limpiar espacios en strings ---
        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

        self.df_clean = df
        print(f"[DataLoader] Dimensiones limpias: {df.shape[0]} filas x {df.shape[1]} columnas")
        return self

    # ------------------------------------------------------------------
    # Acceso
    # ------------------------------------------------------------------
    def get_dataframe(self) -> pd.DataFrame:
        """Devuelve el DataFrame limpio (o crudo si no se limpió)."""
        if self.df_clean is not None:
            return self.df_clean
        if self.df_raw is not None:
            return self.df_raw
        raise RuntimeError("No hay datos cargados. Llame a .load() primero.")
