"""
Análisis extendido del algoritmo K-means (basado en kmeans.py de
https://github.com/MichelleRergis/IA-TC3009C.-K-means) aplicado al dataset real
"Mall Customer Segmentation Data" (Kaggle).

Autor: Michelle Rergis Novelo
Materia: TC3006C - IA Modulo 2.
"""
import json                        # para guardar todas las métricas númericas en un solo archivo
from pathlib import Path           
import numpy as np                 
import pandas as pd                # carga y manejo del CSV como DataFrame
import matplotlib
matplotlib.use("Agg")              # backend sin ventana (necesario para correr sin GUI / en servidor)
import matplotlib.pyplot as plt
import seaborn as sns              # solo se usa para el estilo de las graficas

from sklearn.model_selection import train_test_split   # particion train/val/test
from sklearn.preprocessing import StandardScaler       
from sklearn.cluster import KMeans, kmeans_plusplus    # modelo principal + inicializador
from sklearn.decomposition import PCA                  # reduccion de dimensionalidad / visualizacion 2D
from sklearn.metrics import silhouette_score, adjusted_rand_score  # metricas de evaluacion

RNG = 42                    # semilla fija para que todo el analisis sea reproducible
OUT = Path("salida_real")   
OUT.mkdir(exist_ok=True)
sns.set_style("whitegrid")  # estilo visual de todas las gráficas
resultados = {}            

# 1. Carga y preparacion del dataset real

df = pd.read_csv("Mall_Customers.csv")
df = df.rename(columns={"Annual Income (k$)": "Annual_Income",
                         "Spending Score (1-100)": "Spending_Score"})  
df["Gender_num"] = (df["Gender"] == "Male").astype(int)  # codificacion binaria: Male=1, Female=0

# Variables candidatas (modelo base) vs. variables tras la seleccion/regularizacion
FEATURES_FULL = ["Age", "Annual_Income", "Spending_Score", "Gender_num"]
FEATURES_REG = ["Annual_Income", "Spending_Score"]  # tras seleccion de variables (se quita Gender_num)

X = df[FEATURES_FULL].copy()   # CustomerID se excluye: es un identificador, no aporta señal

resultados["n_muestras"] = len(df)
resultados["features_originales"] = FEATURES_FULL


# 2. Particion Train / Validation / Test  (60 / 20 / 20)
# Primero se separa un 40% (validacion + prueba), luego ese 40% se parte a la mitad.
X_train, X_temp = train_test_split(X, test_size=0.4, random_state=RNG)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=RNG)

resultados["split"] = {
    "train": len(X_train), "val": len(X_val), "test": len(X_test)
}
print("Split:", resultados["split"])

# 3. Escalado (ajustado SOLO con train, para evitar fuga de informacion)
scaler_full = StandardScaler().fit(X_train)        # aprende media/desv. SOLO de entrenamiento
Xtr_full = scaler_full.transform(X_train)           # aplica esa transformacion a los 3 conjuntos
Xval_full = scaler_full.transform(X_val)
Xtest_full = scaler_full.transform(X_test)

# 4. Barrido de K: inercia y silueta en TRAIN, silueta transferida a VAL/TEST
#    (diagnostico de bias / varianza / ajuste)

k_values = list(range(2, 11))   # probamos K de 2 a 10 clusters
inertia_train = []              # WCSS (suma de distancias^2 a su centroide) por cada K
sil_train = []                  
sil_val = []                    
sil_test = []                   # silhouette score en prueba (idem)

for k in k_values:
    # k-means++: inicializacion inteligente de centroides; n_init=10 corridas internas,
    # se conserva la de menor inercia (reduce el riesgo de un mal optimo local).
    mod = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RNG)
    lab_tr = mod.fit_predict(Xtr_full)          # entrena y asigna cluster a cada punto de train
    inertia_train.append(mod.inertia_)         
    sil_train.append(silhouette_score(Xtr_full, lab_tr))  # que tan bien separados quedaron los clusters

    lab_val = mod.predict(Xval_full)            # NO se reentrena: solo se asignan val/test
    lab_test = mod.predict(Xtest_full)          # al cluster mas cercano de los centroides ya aprendidos
    sil_val.append(silhouette_score(Xval_full, lab_val))
    sil_test.append(silhouette_score(Xtest_full, lab_test))

resultados["barrido_k"] = {
    "k": k_values, "inertia_train": inertia_train,
    "sil_train": sil_train, "sil_val": sil_val, "sil_test": sil_test
}

# Gráfica 1: Metodo del codo (train) 
# Sirve para ver donde la inercia deja de bajar de forma pronunciada ("codo").
plt.figure(figsize=(7, 5))
plt.plot(k_values, inertia_train, marker="o", color="b")
plt.title("Metodo del Codo - Inercia (WCSS) en Entrenamiento")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("WCSS (Inercia)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "01_metodo_codo.png", dpi=150)
plt.close()

# Gráfica 2 (CLAVE): Silueta Train vs Val vs Test por K 
# si las 3 curvas son bajas y parecidas -> underfitting (bias alto).
# si difieren mucho entre si o son inestables -> variance alto.
plt.figure(figsize=(8, 5.5))
plt.plot(k_values, sil_train, marker="o", label="Entrenamiento", color="#1f77b4")
plt.plot(k_values, sil_val, marker="s", label="Validacion", color="#ff7f0e")
plt.plot(k_values, sil_test, marker="^", label="Prueba", color="#2ca02c")
plt.title("Silhouette Score vs K: Entrenamiento / Validacion / Prueba")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "02_silueta_train_val_test.png", dpi=150)
plt.close()

# El K final se elige con el conjunto de VALIDACION (nunca con prueba, para no sesgar la evaluacion final)
k_mejor = k_values[int(np.argmax(sil_val))]
resultados["k_elegido_por_validacion"] = k_mejor
print("K elegido segun validacion:", k_mejor)

# 5. Diagnostico de VARIANZA: estabilidad de clusters ante distintas semillas
#    (ARI promedio entre pares de corridas con distinta inicializacion)

def estabilidad_ari(X_data, k, n_semillas=15, n_init=1):
    """
    Entrena el mismo K-means varias veces (cambiando solo la semilla de
    inicializacion) y mide, con el Indice de Rand Ajustado (ARI), que tan
    parecidas son las particiones resultantes entre si.
    ARI cercano a 1  -> el resultado es estable / poca varianza.
    ARI cercano a 0  -> el resultado depende mucho del azar -> varianza alta.
    Devuelve (ari_promedio, ari_desviacion_estandar) entre todas las corridas.
    """
    etiquetas = []
    for s in range(n_semillas):
        
        mod = KMeans(n_clusters=k, init="random", n_init=n_init, random_state=s)
        etiquetas.append(mod.fit_predict(X_data))
    aris = []
    for i in range(len(etiquetas)):
        for j in range(i + 1, len(etiquetas)):
            aris.append(adjusted_rand_score(etiquetas[i], etiquetas[j]))  # compara cada par de corridas
    return float(np.mean(aris)), float(np.std(aris))

estab_k = []
estab_media = []
estab_std = []
for k in k_values:
    m, s = estabilidad_ari(Xtr_full, k)   # se calcula la estabilidad para cada K del barrido
    estab_k.append(k)
    estab_media.append(m)
    estab_std.append(s)

resultados["estabilidad_baseline"] = {
    "k": estab_k, "ari_medio": estab_media, "ari_std": estab_std
}

# Gráfica 3: Estabilidad (varianza) vs K 
plt.figure(figsize=(7, 5))
plt.errorbar(estab_k, estab_media, yerr=estab_std, marker="o", color="purple",
             capsize=4)
plt.title("Estabilidad del clustering ante distintas inicializaciones\n"
          "(ARI promedio entre corridas, 4 variables sin regularizar)")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("ARI promedio entre corridas (1 = identico)")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "03_estabilidad_varianza_baseline.png", dpi=150)
plt.close()

# 6. Modelo BASELINE final (sin regularizar) con K elegido por validacion
modelo_base = KMeans(n_clusters=k_mejor, init="k-means++", n_init=10, random_state=RNG)
modelo_base.fit(Xtr_full)   # entrenamiento final (una sola vez) con el K elegido

lab_tr_base = modelo_base.labels_          # etiquetas de cluster asignadas a cada punto de train
lab_val_base = modelo_base.predict(Xval_full)   # se reutilizan los mismos centroides (no se reentrena)
lab_test_base = modelo_base.predict(Xtest_full)

sil_tr_base = silhouette_score(Xtr_full, lab_tr_base)
sil_val_base = silhouette_score(Xval_full, lab_val_base)
sil_test_base = silhouette_score(Xtest_full, lab_test_base)
ari_base_mean, ari_base_std = estabilidad_ari(Xtr_full, k_mejor)   # estabilidad del modelo final elegido

resultados["baseline"] = {
    "k": k_mejor,
    "inertia_train": float(modelo_base.inertia_),
    "sil_train": float(sil_tr_base),
    "sil_val": float(sil_val_base),
    "sil_test": float(sil_test_base),
    "ari_estabilidad_media": ari_base_mean,
    "ari_estabilidad_std": ari_base_std,
}
print("Baseline:", resultados["baseline"])

# Función auxiliar: visualizacion de clusters proyectados a 2D con PCA 
def graficar_pca(X_list, labels_list, titulos, archivo, pca_fit_data):
    """
    Proyecta cada conjunto de datos (train/val/test) a 2 componentes principales
    (ajustando el PCA SOLO con pca_fit_data, normalmente train) y dibuja un
    scatter por cada conjunto, coloreado segun el cluster asignado.
    Sirve unicamente para visualizar; el modelo K-means opera en el espacio
    original (no en el espacio PCA de esta funcion, salvo que se le pase asi).
    """
    pca = PCA(n_components=2, random_state=RNG).fit(pca_fit_data)
    fig, axes = plt.subplots(1, len(X_list), figsize=(6 * len(X_list), 5))
    if len(X_list) == 1:
        axes = [axes]
    for ax, Xd, lab, tit in zip(axes, X_list, labels_list, titulos):
        X2 = pca.transform(Xd)
        sca = ax.scatter(X2[:, 0], X2[:, 1], c=lab, cmap="tab10", s=25, alpha=0.7)
        ax.set_title(tit)
        ax.set_xlabel("CP1")
        ax.set_ylabel("CP2")
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()

graficar_pca(
    [Xtr_full, Xval_full, Xtest_full],
    [lab_tr_base, lab_val_base, lab_test_base],
    [f"Entrenamiento (Sil={sil_tr_base:.3f})",
     f"Validacion (Sil={sil_val_base:.3f})",
     f"Prueba (Sil={sil_test_base:.3f})"],
    OUT / "04_clusters_baseline_train_val_test.png",
    Xtr_full,
)


# 7. Curva de convergencia del modelo baseline (equivalente a curva de entrenamiento)
def curva_convergencia(X_e, k, semilla, archivo, max_iter=25, tol=1e-6):
    """
    K-means no tiene una funcion de perdida por "epocas" como una red neuronal,
    pero SI minimiza la inercia (WCSS) de forma monotona en cada iteracion.
    Esta funcion fuerza al algoritmo a avanzar de a UNA iteracion (max_iter=1),
    reutilizando los centroides de la iteracion anterior como punto de partida
    de la siguiente, para poder graficar la inercia iteracion por iteracion
    (una curva "de entrenamiento" analoga a la de un modelo supervisado).
    Se detiene cuando el cambio en los centroides es menor a `tol`.
    """
    centros, _ = kmeans_plusplus(X_e, n_clusters=k, random_state=semilla)  # centroides iniciales
    inercias = []
    for it in range(1, max_iter + 1):
        mod = KMeans(n_clusters=k, init=centros, n_init=1, max_iter=1, random_state=semilla)
        mod.fit(X_e)
        inercias.append(mod.inertia_)
        centros_nuevos = mod.cluster_centers_
        cambio = np.linalg.norm(centros_nuevos - centros)  # que tanto se movieron los centroides
        centros = centros_nuevos
        if cambio < tol:      # convergencia: los centroides ya casi no se mueven
            break
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(inercias) + 1), inercias, marker="o", color="green")
    plt.title(f"Curva de Convergencia de K-means (K={k})")
    plt.xlabel("Iteracion")
    plt.ylabel("Inercia (WCSS)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()
    return inercias

conv_base = curva_convergencia(Xtr_full, k_mejor, RNG, OUT / "05_convergencia_baseline.png")
resultados["convergencia_baseline"] = conv_base

# 8. REGULARIZACION / AJUSTE DE HIPERPARAMETROS

scaler_reg = StandardScaler().fit(X_train[FEATURES_REG])   # se re-ajusta el escalador solo con las 2 var. finales
Xtr_reg_raw = scaler_reg.transform(X_train[FEATURES_REG])
Xval_reg_raw = scaler_reg.transform(X_val[FEATURES_REG])
Xtest_reg_raw = scaler_reg.transform(X_test[FEATURES_REG])

pca_reg = PCA(n_components=2, random_state=RNG).fit(Xtr_reg_raw)   # PCA ajustado SOLO con train
Xtr_reg = pca_reg.transform(Xtr_reg_raw)     # datos ya en el espacio reducido (2 componentes)
Xval_reg = pca_reg.transform(Xval_reg_raw)
Xtest_reg = pca_reg.transform(Xtest_reg_raw)

resultados["pca_varianza_explicada"] = pca_reg.explained_variance_ratio_.tolist()  

# Barrido de K en el espacio regularizado (para elegir K de forma consistente con el nuevo espacio)
sil_train_reg = []
sil_val_reg = []
for k in k_values:
    mod = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RNG)  # n_init mas alto -> menos varianza
    lab = mod.fit_predict(Xtr_reg)
    sil_train_reg.append(silhouette_score(Xtr_reg, lab))
    lab_v = mod.predict(Xval_reg)
    sil_val_reg.append(silhouette_score(Xval_reg, lab_v))

k_mejor_reg = k_values[int(np.argmax(sil_val_reg))]   # nuevo K optimo, elegido otra vez solo con validacion
resultados["k_elegido_regularizado"] = k_mejor_reg

# Gráfica 5: comparacion silhouette antes (lineas punteadas) vs despues (lineas solidas) 
plt.figure(figsize=(8, 5.5))
plt.plot(k_values, sil_train, marker="o", linestyle="--", color="#1f77b4", alpha=0.5,
         label="Train (antes, 4 var.)")
plt.plot(k_values, sil_val, marker="s", linestyle="--", color="#ff7f0e", alpha=0.5,
         label="Val (antes, 4 var.)")
plt.plot(k_values, sil_train_reg, marker="o", color="#1f77b4",
         label="Train (despues, PCA 2D)")
plt.plot(k_values, sil_val_reg, marker="s", color="#ff7f0e",
         label="Val (despues, PCA 2D)")
plt.title("Silhouette vs K: antes vs despues de regularizar")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "06_silueta_antes_despues.png", dpi=150)
plt.close()

# Modelo regularizado final: mismo procedimiento que el baseline, pero en el espacio PCA de 2D
modelo_reg = KMeans(n_clusters=k_mejor_reg, init="k-means++", n_init=20, random_state=RNG)
modelo_reg.fit(Xtr_reg)

lab_tr_reg = modelo_reg.labels_
lab_val_reg = modelo_reg.predict(Xval_reg)
lab_test_reg = modelo_reg.predict(Xtest_reg)

sil_tr_reg = silhouette_score(Xtr_reg, lab_tr_reg)
sil_val_reg_f = silhouette_score(Xval_reg, lab_val_reg)
sil_test_reg = silhouette_score(Xtest_reg, lab_test_reg)
ari_reg_mean, ari_reg_std = estabilidad_ari(Xtr_reg, k_mejor_reg, n_init=1)   # estabilidad tras regularizar

resultados["regularizado"] = {
    "k": k_mejor_reg,
    "inertia_train": float(modelo_reg.inertia_),
    "sil_train": float(sil_tr_reg),
    "sil_val": float(sil_val_reg_f),
    "sil_test": float(sil_test_reg),
    "ari_estabilidad_media": ari_reg_mean,
    "ari_estabilidad_std": ari_reg_std,
}
print("Regularizado:", resultados["regularizado"])

graficar_pca(
    [Xtr_reg, Xval_reg, Xtest_reg],
    [lab_tr_reg, lab_val_reg, lab_test_reg],
    [f"Entrenamiento (Sil={sil_tr_reg:.3f})",
     f"Validacion (Sil={sil_val_reg_f:.3f})",
     f"Prueba (Sil={sil_test_reg:.3f})"],
    OUT / "07_clusters_regularizado_train_val_test.png",
    Xtr_reg,
)

# Estabilidad (varianza) despues de regularizar, para comparar contra el baseline
estab_k_reg = []
estab_media_reg = []
estab_std_reg = []
for k in k_values:
    m, s = estabilidad_ari(Xtr_reg, k, n_init=1)
    estab_k_reg.append(k)
    estab_media_reg.append(m)
    estab_std_reg.append(s)

resultados["estabilidad_regularizado"] = {
    "k": estab_k_reg, "ari_medio": estab_media_reg, "ari_std": estab_std_reg
}

# --- Grafica 7: estabilidad antes vs despues, para dejar visible la reduccion de varianza ---
plt.figure(figsize=(7, 5))
plt.errorbar(estab_k, estab_media, yerr=estab_std, marker="o", color="purple",
             capsize=4, label="Antes (4 variables)", alpha=0.6, linestyle="--")
plt.errorbar(estab_k_reg, estab_media_reg, yerr=estab_std_reg, marker="s",
             color="darkgreen", capsize=4, label="Despues (PCA 2D)")
plt.title("Estabilidad ante inicializacion: antes vs despues de regularizar")
plt.xlabel("Numero de clusters (K)")
plt.ylabel("ARI promedio entre corridas")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "08_estabilidad_antes_despues.png", dpi=150)
plt.close()


# 9. Comparativa final (barras) antes vs despues
categorias = ["Sil. Train", "Sil. Val", "Sil. Test", "Estabilidad (ARI)"]
antes = [sil_tr_base, sil_val_base, sil_test_base, ari_base_mean]
despues = [sil_tr_reg, sil_val_reg_f, sil_test_reg, ari_reg_mean]

x = np.arange(len(categorias))
w = 0.35
plt.figure(figsize=(8, 5.5))
plt.bar(x - w / 2, antes, width=w, label=f"Antes (K={k_mejor}, 4 var.)", color="#F4A6A6")
plt.bar(x + w / 2, despues, width=w, label=f"Despues (K={k_mejor_reg}, PCA 2D)", color="#8FD9A8")
plt.xticks(x, categorias)
plt.ylim(0, 1.05)
plt.ylabel("Valor de la metrica")
plt.title("Comparacion de desempeno: antes vs despues de regularizar")
plt.legend()
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT / "09_comparacion_antes_despues.png", dpi=150)
plt.close()

# 10. Guardar todos los resultados numericos (para el reporte / trazabilidad)

with open(OUT / "resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print("\nProceso terminado. Resultados en:", OUT.resolve())