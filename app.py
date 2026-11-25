import json
import pandas as pd
import streamlit as st
from practica12 import obtener_datos

st.set_page_config(page_title="Dashboard NEO - API", layout="wide")
st.title("Análisis de Objetos Cercanos a la Tierra (API)")


def _parse_obj(val):
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


def _extraer_de_neo_dict(neo_dict: dict) -> list[dict]:
    filas = []
    for fecha_str, objetos in neo_dict.items():
        if not isinstance(objetos, list):
            continue
        for obj in objetos:
            obj = _parse_obj(obj)
            if not isinstance(obj, dict):
                continue

            cad_list = _parse_obj(obj.get("close_approach_data") or [{}])
            cad = cad_list[0] if isinstance(cad_list, list) and cad_list else {}
            cad = _parse_obj(cad)
            if not isinstance(cad, dict):
                cad = {}

            diam_root = _parse_obj(obj.get("estimated_diameter") or {})
            diam = _parse_obj((diam_root.get("meters") if isinstance(diam_root, dict) else {}) or {})
            d_min = diam.get("estimated_diameter_min") if isinstance(diam, dict) else None
            d_max = diam.get("estimated_diameter_max") if isinstance(diam, dict) else None
            diam_prom = ((float(d_min) + float(d_max)) / 2) if (d_min is not None and d_max is not None) else None

            vel_obj = _parse_obj(cad.get("relative_velocity") or {})
            dist_obj = _parse_obj(cad.get("miss_distance") or {})

            vel_val = vel_obj.get("kilometers_per_second") if isinstance(vel_obj, dict) else None
            dist_val = dist_obj.get("lunar") if isinstance(dist_obj, dict) else None
            fecha_aprox = cad.get("close_approach_date") or fecha_str

            filas.append({
                "nombre": obj.get("name", "Sin nombre"),
                "fecha": fecha_aprox,
                "diametro_m": float(diam_prom) if diam_prom is not None else None,
                "peligroso": bool(obj.get("is_potentially_hazardous_asteroid", False)),
                "velocidad_kms": float(vel_val) if vel_val is not None else None,
                "distancia_lunar": float(dist_val) if dist_val is not None else None,
            })
    return filas


def normalizar_datos(raw) -> pd.DataFrame:
    raw = _parse_obj(raw)

    # Caso 1: ya es DataFrame limpio
    if isinstance(raw, pd.DataFrame):
        if {"nombre", "fecha", "diametro_m", "velocidad_kms", "distancia_lunar"}.issubset(raw.columns):
            return raw.copy()
        if "near_earth_objects" in raw.columns:
            filas = []
            for _, row in raw.iterrows():
                neo_d = _parse_obj(row["near_earth_objects"])
                if isinstance(neo_d, dict):
                    filas.extend(_extraer_de_neo_dict(neo_d))
            if filas:
                return pd.DataFrame(filas)
        # Si es DataFrame de registros planos con otros nombres
        return raw.copy()

    # Caso 2: dict con near_earth_objects
    if isinstance(raw, dict):
        neo_d = _parse_obj(raw.get("near_earth_objects", raw))
        if isinstance(neo_d, dict):
            filas = _extraer_de_neo_dict(neo_d)
            if filas:
                return pd.DataFrame(filas)
        return pd.DataFrame([raw])

    # Caso 3: lista (de dicts planos, de respuestas API, o de objetos NEO)
    if isinstance(raw, list) and raw:
        filas = []
        for item in raw:
            item = _parse_obj(item)
            if not isinstance(item, dict):
                continue

            # Subcaso 3a: el elemento es una respuesta completa con 'near_earth_objects'
            if "near_earth_objects" in item:
                neo_d = _parse_obj(item["near_earth_objects"])
                if isinstance(neo_d, dict):
                    filas.extend(_extraer_de_neo_dict(neo_d))
                continue

            # Subcaso 3b: el elemento ya es un dict plano con las columnas esperadas
            if any(k in item for k in ("nombre", "fecha", "diametro_m", "velocidad_kms", "distancia_lunar")):
                filas.append(item)
                continue

            # Subcaso 3c: el elemento es un objeto NEO suelto
            if "name" in item or "close_approach_data" in item:
                cad_list = _parse_obj(item.get("close_approach_data") or [{}])
                cad = _parse_obj(cad_list[0] if isinstance(cad_list, list) and cad_list else {})
                diam_root = _parse_obj(item.get("estimated_diameter") or {})
                diam = _parse_obj((diam_root.get("meters") if isinstance(diam_root, dict) else {}) or {})
                d_min = diam.get("estimated_diameter_min") if isinstance(diam, dict) else None
                d_max = diam.get("estimated_diameter_max") if isinstance(diam, dict) else None
                diam_prom = ((float(d_min) + float(d_max)) / 2) if (d_min is not None and d_max is not None) else None
                vel_obj = _parse_obj(cad.get("relative_velocity") if isinstance(cad, dict) else {})
                dist_obj = _parse_obj(cad.get("miss_distance") if isinstance(cad, dict) else {})

                filas.append({
                    "nombre": item.get("name", "Sin nombre"),
                    "fecha": cad.get("close_approach_date") if isinstance(cad, dict) else None,
                    "diametro_m": float(diam_prom) if diam_prom is not None else None,
                    "peligroso": bool(item.get("is_potentially_hazardous_asteroid", False)),
                    "velocidad_kms": float(vel_obj.get("kilometers_per_second")) if isinstance(vel_obj, dict) and vel_obj.get("kilometers_per_second") is not None else None,
                    "distancia_lunar": float(dist_obj.get("lunar")) if isinstance(dist_obj, dict) and dist_obj.get("lunar") is not None else None,
                })

        if filas:
            return pd.DataFrame(filas)
        return pd.DataFrame(raw)

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def cargar_datos() -> pd.DataFrame:
    return normalizar_datos(obtener_datos())


df = cargar_datos()

if df.empty:
    st.warning("No se encontraron registros para mostrar.")
    st.stop()

# Renombrar columnas comunes si vienen en otro idioma/nombre
renames = {}
for col in df.columns:
    cl = col.lower()
    if cl in ("fecha", "date", "close_approach_date") and "fecha" not in df.columns:
        renames[col] = "fecha"
    elif cl in ("nombre", "name") and "nombre" not in df.columns:
        renames[col] = "nombre"
    elif cl in ("diametro_m", "diametro", "diameter_m") and "diametro_m" not in df.columns:
        renames[col] = "diametro_m"
    elif cl in ("velocidad_kms", "velocidad", "velocity_kms") and "velocidad_kms" not in df.columns:
        renames[col] = "velocidad_kms"
    elif cl in ("distancia_lunar", "distancia", "lunar_distance") and "distancia_lunar" not in df.columns:
        renames[col] = "distancia_lunar"
    elif cl in ("peligroso", "hazardous", "is_potentially_hazardous_asteroid") and "peligroso" not in df.columns:
        renames[col] = "peligroso"

if renames:
    df = df.rename(columns=renames)

if "fecha" in df.columns:
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

filtrado = df.copy()
st.sidebar.header("Filtros")

if "fecha" in filtrado.columns and filtrado["fecha"].notna().any():
    f_min = filtrado["fecha"].dropna().min().date()
    f_max = filtrado["fecha"].dropna().max().date()
    rango_fechas = st.sidebar.date_input("Rango de fechas", value=(f_min, f_max), min_value=f_min, max_value=f_max)
    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        filtrado = filtrado[
            (filtrado["fecha"] >= pd.to_datetime(rango_fechas[0]))
            & (filtrado["fecha"] <= pd.to_datetime(rango_fechas[1]))
        ]

if "peligroso" in filtrado.columns:
    opcion_pel = st.sidebar.selectbox("Potencialmente peligroso", ["Todos", True, False])
    if opcion_pel != "Todos":
        filtrado = filtrado[filtrado["peligroso"] == opcion_pel]

if "velocidad_kms" in filtrado.columns and filtrado["velocidad_kms"].notna().any():
    v_min = float(filtrado["velocidad_kms"].min())
    v_max = float(filtrado["velocidad_kms"].max())
    if v_min < v_max:
        rango_vel = st.sidebar.slider("Velocidad (km/s)", min_value=v_min, max_value=v_max, value=(v_min, v_max))
        filtrado = filtrado[(filtrado["velocidad_kms"] >= rango_vel[0]) & (filtrado["velocidad_kms"] <= rango_vel[1])]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", len(filtrado))
c2.metric("Peligrosos", int(filtrado["peligroso"].sum()) if ("peligroso" in filtrado.columns and len(filtrado)) else 0)
c3.metric("Diámetro prom. (m)", f"{filtrado['diametro_m'].mean():.2f}" if ("diametro_m" in filtrado.columns and filtrado["diametro_m"].notna().any()) else "N/A")
c4.metric("Distancia lunar prom.", f"{filtrado['distancia_lunar'].mean():.2f}" if ("distancia_lunar" in filtrado.columns and filtrado["distancia_lunar"].notna().any()) else "N/A")

tab1, tab2, tab3 = st.tabs(["Relación", "Distribución", "Datos y descarga"])

with tab1:
    if not filtrado.empty and {"distancia_lunar", "velocidad_kms"}.issubset(filtrado.columns):
        st.subheader("Velocidad (km/s) vs Distancia lunar")
        color_col = "peligroso" if "peligroso" in filtrado.columns else None
        st.scatter_chart(filtrado, x="distancia_lunar", y="velocidad_kms", color=color_col)
    else:
        st.info("No hay columnas suficientes para la gráfica de relación.")

with tab2:
    if not filtrado.empty and {"nombre", "diametro_m"}.issubset(filtrado.columns):
        st.subheader("Diámetro estimado (m) por registro")
        st.bar_chart(filtrado.set_index("nombre")["diametro_m"].dropna().head(25))
    else:
        st.info("No hay columnas suficientes para la gráfica de distribución.")

with tab3:
    st.dataframe(filtrado, use_container_width=True)
    csv = filtrado.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV filtrado", data=csv, file_name="datos_filtrados.csv", mime="text/csv")
