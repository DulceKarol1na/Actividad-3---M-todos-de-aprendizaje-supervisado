import pandas as pd  # Manipulación y análisis de datos
from sklearn.model_selection import train_test_split  # División de datos en entrenamiento y prueba
from sklearn.ensemble import RandomForestRegressor  # Modelo de regresión basado en bosque aleatorio
from sklearn.preprocessing import LabelEncoder  # Codificación de etiquetas categóricas
from collections import defaultdict  # Diccionario con valores predeterminados
import heapq  # Operaciones eficientes con montículos

# Cargar los datos desde el archivo Excel
datos = pd.read_excel("DatosEjercicioModeloSupervisado.xlsx")

# Crear un diccionario de distancias entre ciudades
distancias = defaultdict(dict)
for _, row in datos.iterrows():
    origen = row['Origen']
    destino = row['Destino']
    kilometros = row['Kilometros']
    distancias[origen][destino] = kilometros
    distancias[destino][origen] = kilometros  # Asumiendo que las distancias son bidireccionales

# Codificar las variables categóricas
encoder = LabelEncoder()
ciudades = pd.concat([datos['Origen'], datos['Destino']])  # Concatenar ambas columnas para codificarlas juntas
encoder.fit(ciudades)
datos['Origen'] = encoder.transform(datos['Origen'])
datos['Destino'] = encoder.transform(datos['Destino'])

# Mapeo inverso para convertir ID de ciudad a nombre de ciudad
ciudades_unicas = ciudades.unique()
id_ciudad_mapa = {encoder.transform([ciudad])[0]: ciudad for ciudad in ciudades_unicas}

print("Etiquetas únicas de 'Origen':", datos['Origen'].unique())
print("Etiquetas únicas de 'Destino':", datos['Destino'].unique())

# Separar las características y la variable objetivo
X = datos[['Origen', 'Destino']]
y_kilometros = datos['Kilometros']
y_tiempo_baja = datos['Temporada_Baja']
y_tiempo_alta = datos['Temporada_Alta']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_kilometros_train, y_kilometros_test, y_tiempo_baja_train, y_tiempo_baja_test, y_tiempo_alta_train, y_tiempo_alta_test = train_test_split(
    X, y_kilometros, y_tiempo_baja, y_tiempo_alta, test_size=0.2, random_state=42)

# Entrenar modelos para predecir kilómetros y tiempo en temporada baja y alta
modelo_kilometros = RandomForestRegressor()
modelo_kilometros.fit(X_train, y_kilometros_train)

modelo_tiempo_baja = RandomForestRegressor()
modelo_tiempo_baja.fit(X_train, y_tiempo_baja_train)

modelo_tiempo_alta = RandomForestRegressor()
modelo_tiempo_alta.fit(X_train, y_tiempo_alta_train)

# Función para encontrar la ruta más corta
def encontrar_ruta_mas_corta(origen, destino):
    queue = [(0, origen, [])]
    visitados = set()
    while queue:
        (costo, ciudad_actual, camino) = heapq.heappop(queue)
        if ciudad_actual in visitados:
            continue
        visitados.add(ciudad_actual)
        camino = camino + [ciudad_actual]
        if ciudad_actual == destino:
            return camino, costo
        for vecino, distancia in distancias[ciudad_actual].items():
            if vecino not in visitados:
                heapq.heappush(queue, (costo + distancia, vecino, camino))
    return None, float('inf')

# Función para predecir los resultados
def predecir_resultados(origen, destino, mes):
    print("Iniciando predicción de resultados...")
    
    # Quita los espacios en blanco al principio y al final de las cadenas origen y destino, y convierte ambos a minúsculas
    origen = origen.strip().lower()
    destino = destino.strip().lower()
    
    print(f"Origen normalizado: {origen}")
    print(f"Destino normalizado: {destino}")

    # Verifica si origen y destino están en la lista de clases del encoder. Si no están, imprime un mensaje de error y retorna cuatro valores None
    if origen not in encoder.classes_:
        print(f"La ciudad de origen '{origen}' no está en la lista. Por favor, inténtelo de nuevo.")
        return None, None, None, None
    if destino not in encoder.classes_:
        print(f"La ciudad de destino '{destino}' no está en la lista. Por favor, inténtelo de nuevo.")
        return None, None, None, None

    # Codifica las cadenas origen y destino en valores numéricos utilizando el encoder
    origen_codificado = encoder.transform([origen])[0]
    destino_codificado = encoder.transform([destino])[0]
    entrada = [[origen_codificado, destino_codificado]]
    
    print(f"Origen codificado: {origen_codificado}")
    print(f"Destino codificado: {destino_codificado}")
    print(f"Entrada para predicción: {entrada}")

    # Verifica si el mes está en la lista de meses de "temporada baja". Si es así, utiliza el modelo modelo_tiempo_baja para predecir el tiempo y establece la temporada como "temporada baja". Si no, utiliza el modelo modelo_tiempo_alta para predecir el tiempo y establece la temporada como "temporada alta".
    if mes in ['febrero', 'marzo', 'abril', 'mayo', 'agosto', 'septiembre']:
        tiempo = modelo_tiempo_baja.predict(entrada)[0]
        temporada = "temporada baja"
    else:
        tiempo = modelo_tiempo_alta.predict(entrada)[0]
        temporada = "temporada alta"
    
    print(f"Tiempo estimado: {tiempo} horas en {temporada}")

    # Utiliza el modelo modelo_kilometros para predecir la distancia en kilómetros entre origen y destino
    kilometros = modelo_kilometros.predict(entrada)[0]
    
    print(f"Distancia estimada: {kilometros} km")

    # Encontrar la ruta más corta usando el diccionario de distancias
    ruta, distancia_total = encontrar_ruta_mas_corta(origen, destino)
    if ruta:
        pueblos = [ciudad.capitalize() for ciudad in ruta]
    else:
        pueblos = []
    
    print(f"Ruta más corta encontrada: {pueblos}")
    
    return kilometros, tiempo, temporada, pueblos

# Solicitar al usuario el origen, destino y mes de salida
while True:
    origen = input("Ingrese el origen: ").strip().lower()
    destino = input("Ingrese el destino: ").strip().lower()
    mes = input("Ingrese el mes de salida: ").strip().lower()

    # Predecir los resultados si no hay error en las ciudades ingresadas
    if origen and destino and mes:
        kilometros, tiempo, temporada, pueblos = predecir_resultados(origen, destino, mes)
        if kilometros is not None and tiempo is not None:
            print("Pueblos por los que pasa:")
            for pueblo in pueblos:
                print("-", pueblo)
            print(f"La distancia estimada entre {origen.capitalize()} y {destino.capitalize()} es de {kilometros} km y el tiempo estimado es {tiempo} horas durante la {temporada}.")
            break
