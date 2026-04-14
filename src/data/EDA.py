"""
------
Clase EDA: análisis exploratorio de datos con estadísticas y gráficas.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class EDA:
    """
    Realiza análisis exploratorio de datos sobre un DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame ya limpio.
    output_dir : str
        Carpeta donde se guardan las gráficas generadas.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "outputs/eda"):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resumen general
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Imprime y devuelve un diccionario con el resumen del dataset."""
        print("\n" + "=" * 60)
        print("RESUMEN DEL DATASET")
        print("=" * 60)
        print(f"  Filas    : {self.df.shape[0]}")
        print(f"  Columnas : {self.df.shape[1]}")
        print(f"  Tipos de datos:\n{self.df.dtypes.to_string()}")

        duplicates = self.df.duplicated().sum()
        null_total = self.df.isnull().sum().sum()
        print(f"\n  Duplicados: {duplicates} ({duplicates/len(self.df)*100:.1f}%)")
        print(f"  Nulos totales: {null_total}")

        return {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.to_dict(),
            "duplicates": int(duplicates),
            "nulls_total": int(null_total),
        }

    # ------------------------------------------------------------------
    # Estadísticas descriptivas
    # ------------------------------------------------------------------
    def descriptive_stats(self) -> dict:
        """Muestra estadísticas numéricas y categóricas."""
        result = {}

        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            print("\n--- ESTADÍSTICAS NUMÉRICAS ---")
            stats = self.df[numeric_cols].describe().T
            print(stats.to_string())
            result["numeric"] = stats

        cat_cols = self.df.select_dtypes(include="object").columns
        if len(cat_cols) > 0:
            print("\n--- ESTADÍSTICAS CATEGÓRICAS ---")
            cat_stats = {}
            for col in cat_cols:
                vc = self.df[col].value_counts()
                print(f"\n  {col} (top 5):")
                print(vc.head(5).to_string())
                cat_stats[col] = vc
            result["categorical"] = cat_stats

        return result

    # ------------------------------------------------------------------
    # Nulos
    # ------------------------------------------------------------------
    def null_analysis(self) -> pd.DataFrame:
        """Analiza valores nulos por columna."""
        null_df = pd.DataFrame({
            "nulos": self.df.isnull().sum(),
            "porcentaje": (self.df.isnull().mean() * 100).round(2),
        }).sort_values("porcentaje", ascending=False)
        print("\n--- ANÁLISIS DE NULOS ---")
        print(null_df[null_df["nulos"] > 0].to_string() if null_df["nulos"].sum() > 0
              else "  Sin valores nulos.")
        return null_df

    # ------------------------------------------------------------------
    # Gráficas
    # ------------------------------------------------------------------
    def plot_distributions(self, save: bool = True) -> None:
        """Histogramas de columnas numéricas."""
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) == 0:
            print("[EDA] No hay columnas numéricas para graficar.")
            return

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].hist(self.df[col].dropna(), bins=30, edgecolor="black", color="steelblue")
            axes[i].set_title(col)
            axes[i].set_xlabel("Valor")
            axes[i].set_ylabel("Frecuencia")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Distribuciones de Variables Numéricas", fontsize=14, y=1.02)
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "distributions.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[EDA] Gráfica guardada: {path}")
        plt.show()
        plt.close()

    def plot_boxplots(self, save: bool = True) -> None:
        """Boxplots para detección de outliers."""
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) == 0:
            return

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].boxplot(self.df[col].dropna())
            axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Boxplots - Detección de Outliers", fontsize=14, y=1.02)
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "boxplots.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[EDA] Gráfica guardada: {path}")
        plt.show()
        plt.close()

    def plot_correlation(self, save: bool = True) -> None:
        """Matriz de correlación."""
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) < 2:
            print("[EDA] Se necesitan al menos 2 columnas numéricas para correlación.")
            return

        plt.figure(figsize=(10, 8))
        corr = self.df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Matriz de Correlación")
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "correlation.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[EDA] Gráfica guardada: {path}")
        plt.show()
        plt.close()

    def plot_top_categories(self, columns: list[str], top_n: int = 10,
                             save: bool = True) -> None:
        """
        Gráficas de barras para columnas categóricas.

        Parámetros
        ----------
        columns : list[str]
            Columnas categóricas a graficar.
        top_n : int
            Cantidad de categorías más frecuentes a mostrar.
        """
        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            return

        fig, axes = plt.subplots(1, len(columns), figsize=(8 * len(columns), 5))
        if len(columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, columns):
            counts = self.df[col].value_counts().head(top_n)
            counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title(f"Top {top_n}: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "top_categories.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[EDA] Gráfica guardada: {path}")
        plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # Ejecución completa
    # ------------------------------------------------------------------
    def run_full(self, cat_columns_to_plot: list[str] | None = None) -> dict:
        """
        Ejecuta todo el EDA de una sola vez.

        Parámetros
        ----------
        cat_columns_to_plot : list[str] | None
            Columnas categóricas para graficar barras.
        """
        info = self.summary()
        info["stats"] = self.descriptive_stats()
        info["nulls"] = self.null_analysis()
        self.plot_distributions()
        self.plot_boxplots()
        self.plot_correlation()
        if cat_columns_to_plot:
            self.plot_top_categories(cat_columns_to_plot)
        return info
