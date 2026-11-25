
import json
import os
import textwrap
from datetime import date, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

#Configuración
URL = "https://api.nasa.gov/neo/rest/v1/feed"
ARCHIVO = "respuesta.json"
INICIO = date(2026, 10, 1)   #primer día a consultar
SEMANAS = 8                  #NeoWs solo permite rangos de 7 días por petición

GRIS = "#B9C3D1"             #contexto
AZUL = "#1C4C9A"             #lo que se destaca

#1. Token: vive fuera del código y se valida
def leer_token():
    token = os.environ.get("NASA_API_KEY", "").strip()
    origen = "variable de entorno NASA_API_KEY"
    if not token and os.path.exists("token.txt"):
        with open("token.txt", encoding="utf-8") as f:
            token = f.read().strip()
        origen = "token.txt"

    if not token:
        print("ERROR: no encontré el token. Guárdalo en token.txt "
              "o en la variable de entorno NASA_API_KEY.")
        raise SystemExit(1)
    if len(token) < 30 or any(c.isspace() for c in token):
        print("ERROR: el token parece incompleto o tiene espacios. "
              "Revisa que copiaste la llave completa, en una sola línea.")
        raise SystemExit(1)

    print(f"Token leído desde {origen} ({len(token)} caracteres).")
    return token

#2 y 3. Consulta con revisión de estado y de tipo de contenido
def revisar_estado(r, token):
    c = r.status_code
    if c == 200:
        return
    detalle = r.text[:300].replace(token, "~~TOKEN~~")
    if c in (401, 403):
        print(f"ERROR {c}: el token no sirve (NASA responde 403 cuando la "
              "llave es inválida). Revisa token.txt o genera otra llave.")
    elif c == 400:
        print(f"ERROR 400: la petición está mal hecha (fechas o parámetros). {detalle}")
    elif c == 429:
        print("ERROR 429: te pasaste del límite de peticiones. "
              "Espera una hora y vuelve a intentar.")
    else:
        print(f"ERROR {c}: respuesta inesperada del servidor. {detalle}")
    raise SystemExit(1)


def consultar_api(token):
    semanas = []
    for i in range(SEMANAS):
        ini = INICIO + timedelta(days=7 * i)
        fin = ini + timedelta(days=6)
        params = {"start_date": ini.isoformat(),
                  "end_date": fin.isoformat(),
                  "api_key": token}
        print(f"Consultando la API: {ini} a {fin} ...")
        try:
            r = requests.get(URL, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"ERROR de red: {e}")
            raise SystemExit(1)

        revisar_estado(r, token)

        if "json" not in (r.headers.get("Content-Type") or ""):
            print(r.text[:300].replace(token, "~~TOKEN~~"))
            raise SystemExit(1)
        try:
            semanas.append(r.json())
        except ValueError:
            print("ERROR: el servidor dijo que era JSON pero no se pudo leer.")
            raise SystemExit(1)
    return semanas

#4. Guardar la respuesta y reutilizarla
def guardar_respuesta(semanas, token):
    for sem in semanas:
        sem.pop("links", None)
    texto = json.dumps(semanas, ensure_ascii=False, indent=1)
    texto = texto.replace(token, "~~TOKEN~~")
    if token in texto:
        print("ERROR: el token aparece en la respuesta; no se guardó el archivo.")
        raise SystemExit(1)
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"Respuesta guardada en {ARCHIVO} (sin token).")


def obtener_datos():
    if os.path.exists(ARCHIVO):
        print(f"Leyendo {ARCHIVO} (no se consulta la API).")
        try:
            with open(ARCHIVO, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"ERROR: {ARCHIVO} está dañado. Bórralo y vuelve a ejecutar.")
            raise SystemExit(1)
    token = leer_token()
    semanas = consultar_api(token)
    guardar_respuesta(semanas, token)
    return semanas

#5. Tabla y limpieza documentada
def a_tabla(semanas):
    filas = []
    for sem in semanas:
        if "near_earth_objects" not in sem:
            print("ERROR: no encontré la llave 'near_earth_objects'. "
                  f"Llaves recibidas: {list(sem.keys())}")
            raise SystemExit(1)
        for fecha, lista in sem["near_earth_objects"].items():
            for a in lista:
                lista_ap = a.get("close_approach_data") or [{}]
                ap = lista_ap[0]
                filas.append({
                    "nombre": a.get("name"),
                    "fecha": fecha,
                    "diametro_m": (a.get("estimated_diameter", {})
                                   .get("meters", {})
                                   .get("estimated_diameter_max")),
                    "peligroso": a.get("is_potentially_hazardous_asteroid"),
                    "velocidad_kms": (ap.get("relative_velocity", {})
                                      .get("kilometers_per_second")),
                    "distancia_lunar": ap.get("miss_distance", {}).get("lunar"),
                })
    return pd.DataFrame(filas)


def limpiar(df):
    print("\n" + "=" * 60)
    print("LIMPIEZA DE DATOS")
    print("=" * 60)
    total = len(df)
    print(f"\nRegistros recibidos: {total}")
    print("\nTipos de dato al cargar:")
    print(df.dtypes.to_string())
    print("\nFaltantes por columna:")
    print(df.isna().sum().to_string())

    #Texto -> número
    print("\nConversión de texto a número:")
    for col in ["diametro_m", "velocidad_kms", "distancia_lunar"]:
        antes = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        perdidos = antes - df[col].notna().sum()
        print(f"  {col}: convertida a número, se perdieron {perdidos} valores")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    print(f"  fecha: convertida a fecha, "
          f"{df['fecha'].isna().sum()} valores no válidos")

    #Quitar incompletos,no positivos y duplicados
    n0 = len(df)
    df = df.dropna(subset=["diametro_m", "velocidad_kms", "distancia_lunar",
                           "fecha", "peligroso"])
    print(f"\nEliminados por datos faltantes: {n0 - len(df)}")
    n1 = len(df)
    df = df[(df["diametro_m"] > 0) & (df["distancia_lunar"] > 0)]
    print(f"Eliminados por valores no positivos: {n1 - len(df)}")
    n2 = len(df)
    df = df.drop_duplicates(subset=["nombre", "fecha"])
    print(f"Eliminados por duplicados: {n2 - len(df)}")

    df["peligroso"] = df["peligroso"].astype(bool)
    df = df.reset_index(drop=True)

    print("\nTipos de dato después de limpiar:")
    print(df.dtypes.to_string())
    print(f"\nRegistros utilizables: {len(df)} de {total}")
    if len(df) < 200:
        print("AVISO: hay menos de 200 registros. Sube SEMANAS en la "
              "configuración y borra respuesta.json.")
    return df

#Gráficas
def estilo(ax, titulo, xlabel, ylabel):
    ax.set_title(textwrap.fill(titulo, 62), loc="left",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def terminar(fig, nombre):
    fig.tight_layout()
    fig.savefig(nombre, dpi=150, bbox_inches="tight")   # primero guardar...
    plt.show()                     # ...y después mostrar
    plt.close(fig)
    print(f"Gráfica guardada: {nombre}")


def grafica_distribucion(df):
    d = df["diametro_m"]
    mediana, maximo = d.median(), d.max()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(d.min()), np.log10(d.max()), 25)
    ax.hist(d, bins=bins, color=GRIS, edgecolor="white")
    ax.axvline(mediana, color=AZUL, linewidth=2)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.12)
    ax.text(mediana * 1.08, 0.93, f"Mediana: {mediana:.0f} m", color=AZUL,
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    estilo(ax,
           f"La mitad de los asteroides mide menos de {mediana:.0f} m, "
           f"pero el mayor llega a {maximo:,.0f} m",
           "Diámetro máximo estimado (m, escala logarítmica)",
           "Número de asteroides")
    terminar(fig, "g1_distribucion.png")


def grafica_comparacion(df):
    peli = df.loc[df["peligroso"], "diametro_m"]
    nope = df.loc[~df["peligroso"], "diametro_m"]
    if peli.empty or nope.empty:
        print("AVISO: no hay asteroides de ambos grupos; se omite la "
              "gráfica de comparación.")
        return
    mp, mn = peli.median(), nope.median()
    if mp > mn:
        titulo = (f"Los asteroides potencialmente peligrosos son más grandes: "
                  f"mediana de {mp:.0f} m contra {mn:.0f} m")
    else:
        titulo = (f"Los potencialmente peligrosos no son más grandes: "
                  f"mediana de {mp:.0f} m contra {mn:.0f} m")

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([nope, peli], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=GRIS, markeredgecolor=GRIS))
    for caja, color in zip(bp["boxes"], [GRIS, AZUL]):
        caja.set_facecolor(color)
    ax.set_xticklabels([f"No peligrosos\n(n = {len(nope)})",
                        f"Potencialmente peligrosos\n(n = {len(peli)})"])
    ax.set_yscale("log")
    estilo(ax, titulo, "Clasificación de NASA",
           "Diámetro máximo estimado (m, escala logarítmica)")
    terminar(fig, "g2_comparacion.png")


def grafica_relacion(df):
    r = df["velocidad_kms"].corr(df["distancia_lunar"], method="spearman")
    if abs(r) < 0.2:
        titulo = (f"La velocidad casi no dice nada de qué tan cerca pasa "
                  f"un asteroide (correlación de Spearman = {r:.2f})")
    elif r > 0:
        titulo = (f"Los asteroides más rápidos tienden a pasar más lejos "
                  f"(correlación de Spearman = {r:.2f})")
    else:
        titulo = (f"Los asteroides más rápidos tienden a pasar más cerca "
                  f"(correlación de Spearman = {r:.2f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    nope = df[~df["peligroso"]]
    peli = df[df["peligroso"]]
    ax.scatter(nope["velocidad_kms"], nope["distancia_lunar"], s=18,
               color=GRIS, alpha=0.8, label="No peligrosos")
    ax.scatter(peli["velocidad_kms"], peli["distancia_lunar"], s=26,
               color=AZUL, alpha=0.9, label="Potencialmente peligrosos")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2)   # el color sí carga información aquí
    estilo(ax, titulo,
           "Velocidad relativa (km/s)",
           "Distancia de paso (veces la distancia Tierra-Luna, escala log)")
    terminar(fig, "g3_relacion.png")


def grafica_evolucion(df):
    por_dia = df.groupby("fecha").size()
    rango = pd.date_range(por_dia.index.min(), por_dia.index.max())
    por_dia = por_dia.reindex(rango, fill_value=0)
    pico_fecha = por_dia.idxmax()
    pico = por_dia.max()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(por_dia.index, por_dia.values, color=GRIS, linewidth=2)
    ax.scatter([pico_fecha], [pico], color=AZUL, s=70, zorder=3)
    ax.annotate(f"Pico: {pico} asteroides",
                (pico_fecha, pico), textcoords="offset points",
                xytext=(10, 6), color=AZUL, fontweight="bold")
    ax.set_ylim(0, pico * 1.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    fig.autofmt_xdate()
    estilo(ax,
           f"El día con más acercamientos fue el "
           f"{pico_fecha.strftime('%d/%m/%Y')}, con {pico} asteroides "
           f"(promedio: {por_dia.mean():.0f} por día)",
           "Fecha de máxima aproximación",
           "Asteroides que pasan cerca (por día)")
    terminar(fig, "g4_evolucion.png")

#Programa principal
def main():
    semanas = obtener_datos()
    df = limpiar(a_tabla(semanas))
    grafica_distribucion(df)
    grafica_comparacion(df)
    grafica_relacion(df)
    grafica_evolucion(df)


if __name__ == "__main__":
    main()
