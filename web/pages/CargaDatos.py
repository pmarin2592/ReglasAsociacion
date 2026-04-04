# Carga Librerías
import streamlit as st
import pandas as pd
import time
import hashlib
import io
from abc import ABC, abstractmethod
from typing import Optional

# Carga Clases
from src.utils.path import obtener_ruta_app


# ──────────────────────────────────────────────
# Clases de lectura de archivos
# ──────────────────────────────────────────────

class FileReader(ABC):
    @abstractmethod
    def read(self, file, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        pass


class CSVReader(FileReader):
    def read(self, file, **kwargs) -> pd.DataFrame:
        return pd.read_csv(file, **kwargs)

    def supported_extensions(self) -> list[str]:
        return [".csv"]


class ExcelReader(FileReader):
    def read(self, file, **kwargs) -> pd.DataFrame:
        xls = pd.ExcelFile(file)
        sheets = xls.sheet_names
        if len(sheets) > 1:
            selected = st.selectbox("📋 Hoja de Excel", options=sheets, key="sheet_selector")
            return pd.read_excel(file, sheet_name=selected, **kwargs)
        return pd.read_excel(file, sheet_name=0, **kwargs)

    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".xls"]


class FileReaderFactory:
    _readers: list[FileReader] = [CSVReader(), ExcelReader()]

    @classmethod
    def get_reader(cls, filename: str) -> Optional[FileReader]:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        for reader in cls._readers:
            if ext in reader.supported_extensions():
                return reader
        return None

    @classmethod
    def accepted_types(cls) -> list[str]:
        exts = []
        for r in cls._readers:
            exts.extend(r.supported_extensions())
        return exts


# ──────────────────────────────────────────────
# Clase de selección de columnas
# ──────────────────────────────────────────────

class ColumnSelector:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.all_columns = list(df.columns)

    def _on_col_change(self):
        val = st.session_state["col_selectbox"]
        if val != "— Elige una columna —" and val not in st.session_state["selected_cols"]:
            st.session_state["selected_cols"].append(val)

    def render(self) -> list[str]:
        st.markdown("#### Seleccionar columnas para visualizar")

        if "selected_cols" not in st.session_state:
            st.session_state["selected_cols"] = []

        col1, col2 = st.columns([4, 1])
        with col1:
            available = [c for c in self.all_columns if c not in st.session_state["selected_cols"]]
            st.selectbox(
                "Columnas disponibles",
                options=["— Elige una columna —"] + available,
                key="col_selectbox",
                label_visibility="collapsed",
                on_change=self._on_col_change,
            )
        with col2:
            if st.button("🗑 Limpiar", use_container_width=True):
                st.session_state["selected_cols"] = []
                st.rerun()

        if st.session_state["selected_cols"]:
            tag_cols = st.columns(min(len(st.session_state["selected_cols"]), 4))
            for i, col in enumerate(list(st.session_state["selected_cols"])):
                with tag_cols[i % 4]:
                    if st.button(f"✕ {col}", key=f"del_{col}", use_container_width=True):
                        st.session_state["selected_cols"].remove(col)
                        st.rerun()
        else:
            st.caption("Sin columnas seleccionadas.")

        return st.session_state["selected_cols"]


# ──────────────────────────────────────────────
# Clase de visualización del DataFrame
# ──────────────────────────────────────────────

class DataFrameViewer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render(self, selected_columns: list[str]):
        if not selected_columns:
            st.warning("⚠️ Sin columnas seleccionadas.")
            return

        filtered = self.df[selected_columns]
        st.markdown(f"**{len(filtered):,} filas · {len(selected_columns)} columnas seleccionadas**")
        st.dataframe(filtered, use_container_width=True, height=350)

        col_stats, col_dl = st.columns([2, 1])
        with col_stats:
            with st.expander("📈 Estadísticas descriptivas"):
                st.dataframe(filtered.describe(include="all"), use_container_width=True)
        with col_dl:
            self._download_button(filtered)

    def _download_button(self, df: pd.DataFrame):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv_buffer.getvalue(),
            file_name="datos_seleccionados.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# Clase principal de carga de datos
# ──────────────────────────────────────────────

class CargaDatos:
    def __init__(self):
        self.archivo_cargado = None
        self.analisis_realizado = False

    def _generar_hash_archivo(self, archivo_bytes):
        return hashlib.md5(archivo_bytes).hexdigest()

    def _obtener_tipo_dato_legible(self, tipo_pandas):
        tipo_str = str(tipo_pandas)
        if "int" in tipo_str:        return "🔢 Entero"
        elif "float" in tipo_str:    return "🔢 Decimal"
        elif "bool" in tipo_str:     return "☑️ Bool"
        elif "datetime" in tipo_str: return "📅 Fecha"
        elif "object" in tipo_str:   return "📝 Texto"
        elif "category" in tipo_str: return "🏷️ Categoría"
        else:                         return f"❓ {tipo_str}"

    def _crear_tabla_con_tooltips(self, df):
        tooltips = {}
        for col in df.columns:
            tipo_dato = self._obtener_tipo_dato_legible(df[col].dtype)
            valores_nulos = df[col].isnull().sum()
            total_valores = len(df)
            porcentaje_nulos = (valores_nulos / total_valores) * 100
            tooltips[col] = (
                f"{tipo_dato}\n"
                f"📊 {total_valores - valores_nulos} válidos\n"
                f"❌ {valores_nulos} nulos ({porcentaje_nulos:.1f}%)"
            )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                col: st.column_config.Column(col, help=tooltips[col], width="medium")
                for col in df.columns
            }
        )

    def _vista_head_tail(self, df: pd.DataFrame, n: int = 5):
        """Muestra head y tail del dataframe lado a lado."""
        col_h, col_t = st.columns(2)
        with col_h:
            st.caption(f"⬆️ Primeras {n} filas")
            self._crear_tabla_con_tooltips(df.head(n))
        with col_t:
            st.caption(f"⬇️ Últimas {n} filas")
            self._crear_tabla_con_tooltips(df.tail(n))

    def _leer_archivo(self, nombre, archivo_bytes, delimitador, decimal) -> pd.DataFrame:
        reader = FileReaderFactory.get_reader(nombre)
        if reader is None:
            raise ValueError(f"Formato no soportado: {nombre}")
        archivo_io = io.BytesIO(archivo_bytes)
        if nombre.lower().endswith(".csv"):
            return reader.read(archivo_io, delimiter=delimitador, decimal=decimal)
        return reader.read(archivo_io, decimal=decimal)

    def render(self):
        try:
            st.title("🗃️ Carga de Datos")

            accepted = FileReaderFactory.accepted_types()

            # ── Controles de carga ──────────────────────────────
            col_opts, col_upload = st.columns([1, 2])
            with col_opts:
                delimitador = st.selectbox("Delimitador", options=[",", ";", "\t"])
                decimal     = st.selectbox("Decimal",     options=[".", ","])
            with col_upload:
                archivo = st.file_uploader(
                    "Sube tu archivo (.csv o .xlsx)",
                    type=[ext.lstrip(".") for ext in accepted],
                    key="uploader",
                )

            # ── Detección de cambios ────────────────────────────
            prev_hash         = st.session_state.get("file_hash")
            prev_delimitador  = st.session_state.get("delimitador")
            prev_decimal      = st.session_state.get("decimal")

            archivo_nuevo        = False
            parametros_cambiados = False

            if archivo is not None:
                archivo_bytes = archivo.read()
                archivo.seek(0)
                archivo_hash = self._generar_hash_archivo(archivo_bytes)

                if prev_hash != archivo_hash:
                    archivo_nuevo = True
                    st.session_state.archivo_bytes = archivo_bytes
                    st.session_state.file_name     = archivo.name
                    st.session_state.file_hash     = archivo_hash

            if prev_delimitador != delimitador or prev_decimal != decimal:
                parametros_cambiados = True

            if "file_name" in st.session_state and (
                archivo_nuevo or parametros_cambiados or "df_cargado" not in st.session_state
            ):
                try:
                    df = self._leer_archivo(
                        st.session_state.file_name,
                        st.session_state.archivo_bytes,
                        delimitador, decimal
                    )
                    st.session_state.df_cargado         = df
                    st.session_state.delimitador        = delimitador
                    st.session_state.decimal            = decimal
                    st.session_state.analisis_generado  = False
                    st.session_state["selected_cols"]   = []
                    st.session_state.pop("eda", None)

                    label = st.session_state.file_name if archivo_nuevo else "parámetros actualizados"
                    st.success(f"✅ {label}")

                except Exception as e:
                    st.error(f"Error al procesar el archivo: {e}")
                    for key in ["df_cargado", "eda", "analisis_generado"]:
                        st.session_state.pop(key, None)

            # ── Vista del dataset ───────────────────────────────
            if "df_cargado" in st.session_state:
                self.archivo_cargado = st.session_state.df_cargado
                df = st.session_state.df_cargado

                # Métricas compactas en una fila
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Filas",    f"{len(df):,}")
                m2.metric("Columnas", f"{len(df.columns)}")
                m3.metric("Nulos",    f"{df.isnull().sum().sum():,}")
                m4.metric("Tamaño",   f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

                st.divider()

                # Head + Tail lado a lado
                st.markdown("#### Vista previa")
                n_rows = st.slider("Filas por sección", min_value=3, max_value=20, value=5, step=1)
                self._vista_head_tail(df, n=n_rows)

                st.divider()

                # Selector de columnas
                selector = ColumnSelector(df)
                selected_columns = selector.render()

                if selected_columns:
                    st.divider()
                    viewer = DataFrameViewer(df)
                    viewer.render(selected_columns)

                st.divider()

                # Botón de análisis centrado
                _, col_btn, _ = st.columns([3, 2, 3])
                with col_btn:
                    if st.button("📊 Generar análisis", use_container_width=True):
                        progreso = st.progress(0, text="Iniciando análisis...")
                        for i in range(1, 101):
                            time.sleep(0.02)
                            progreso.progress(i, text=f"Analizando... {i}%")
                        st.success("✅ Análisis completado.")
                        self.analisis_realizado            = True
                        st.session_state.analisis_generado = True
                        st.rerun()
            else:
                st.info("📂 Sube un archivo CSV o Excel (.xlsx) para comenzar.")

        except Exception as e:
            st.error(f"Error en la pantalla de datos: {e}")