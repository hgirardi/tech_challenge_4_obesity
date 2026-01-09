from __future__ import annotations

from pathlib import Path
import streamlit as st

import tomllib


def _load_css(path: str = "app/style/style.css") -> None:
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _load_nav(path: str = "app/components/nav.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def layout(page_title: str, nav_path: str = "app/components/nav.toml") -> None:
    st.set_page_config(page_title=page_title, layout="wide")

    # Carregar CSS personalizado
    _load_css()

    # Carregar barra de navegação
    cfg = _load_nav(nav_path)
    # st.sidebar.title(cfg.get("title", "Menu"))

    # Adicionar itens de navegação de forma dinâmica do arquivo components/nav.toml
    for item in cfg.get("items", []):
        t = item.get("type")
        if t == "section":
            st.sidebar.markdown(f"### {item['label']}")
            continue

        icon = item.get("icon", "")
        label = item.get("label", "")
        full_label = f"{icon} {label}".strip()

        if t == "page":
            st.sidebar.page_link(item["path"], label=full_label)
        elif t == "link":
            st.sidebar.page_link(item["url"], label=full_label)

def first_subheader(texto: str, id_elem: str | None = None):
    id_attr = f' id="{id_elem}"' if id_elem else ""
    st.markdown(f'<h3 class="first-subheader"{id_attr}>{texto}</h3>', unsafe_allow_html=True)