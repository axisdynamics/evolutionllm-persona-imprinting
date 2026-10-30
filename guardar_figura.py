#!/usr/bin/env python3
"""
Guardado único de figuras: PNG para revisar y para los PDF de trabajo (hacer_pdfs.py),
más **PDF vectorial** para el envío a revista, que no acepta rasterizado.

Los scripts graficar_*.py llaman `guardar(fig, "artifacts/x.png")` en vez de
`fig.savefig(...)`; el .pdf sale al lado con el mismo nombre.

Importar este módulo ANTES de crear la figura: fija fonttype 42 (TrueType embebido)
en vez del default 3, porque el Type 3 de matplotlib rompe la extracción de texto y
varias editoriales lo rechazan.
"""
from pathlib import Path
import matplotlib

# Fuentes embebidas y editables en el vectorial (requisito habitual de las editoriales).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def guardar(fig, ruta_png, dpi=150):
    """Escribe <ruta>.png y <ruta>.pdf. Devuelve (png, pdf) como Path."""
    png = Path(ruta_png)
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")          # vectorial, sin dpi
    return png, pdf
