# Carga Librerias
import os
import sys
import streamlit as st
# Agrega el path del proyecto raíz para que se pueda importar src.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Carga Clases
from web.pages.MenuPrincipal import MenuPrincipal

def crear_config_streamlit(max_mb=2000):
    ruta_config = os.path.expanduser("~/.streamlit/config.toml")

    if os.name == "nt":
        ruta_config = os.path.expandvars(r"%USERPROFILE%\.streamlit\config.toml")

    os.makedirs(os.path.dirname(ruta_config), exist_ok=True)

    contenido = f"""
                [server]
                maxUploadSize = {max_mb}
                """
    with open(ruta_config, "w") as f:
        f.write(contenido.strip())

    print(f"Archivo config.toml creado o actualizado en: {ruta_config}")

def main():
    if "config_creado" not in st.session_state:
        crear_config_streamlit()
        st.session_state["config_creado"] = True
    menu = MenuPrincipal(os.path.dirname(__file__))
    menu.menu_principal()

if __name__ == '__main__':
    main()