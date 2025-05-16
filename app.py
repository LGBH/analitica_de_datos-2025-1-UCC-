from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


app = Flask(__name__)

# ============================
# MODELO REGRESIÓN LOGÍSTICA
# ============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/modelo_logistico', methods=['GET', 'POST'])
def modelo_logistico():
    resultado = None
    if request.method == 'POST':
        try:
            valor = float(request.form['valor'])
            modelo = joblib.load("archivos_.pkl/modelo_FGR.pkl")
            prediccion = modelo.predict([[valor]])
            resultado = "Normal" if prediccion[0] == 0 else "FGR"
        except Exception as e:
            resultado = f"❌ Error: {str(e)}"
    return render_template('modelo_logistico.html', resultado=resultado)

# ============================
# MODELO ANN
# ============================
@app.route('/modelo_ann', methods=['GET', 'POST'])
def modelo_ann():
    resultado = None
    if request.method == 'POST':
        try:
            datos = [float(request.form[f"C{i}"]) for i in range(1, 31)]
            modelo = load_model("archivos_.pkl/modelo_ann.keras")
            scaler = joblib.load("archivos_.pkl/escalador_ann.pkl")
            datos_escalados = scaler.transform([datos])
            prediccion = modelo.predict(datos_escalados)
            prob = prediccion[0][0]
            resultado = f"Su peso fetal es: {'FGR' if prob > 0.5 else 'Normal'} (probabilidad: {prob:.2f})"
        except Exception as e:
            resultado = f"❌ Error: {str(e)}"
    return render_template('modelo_ann.html', resultado=resultado)

# ============================
# MODELO SVC - 30 columnas
# ============================
@app.route('/modelo_svc', methods=['GET', 'POST'])
def modelo_svc():
    resultado = None
    if request.method == 'POST':
        try:
            archivo = request.files['archivo']
            df = pd.read_excel(archivo)

            if df.shape[1] != 30:
                raise ValueError("El archivo debe tener exactamente 30 columnas.")

            modelo = joblib.load("archivos_.pkl/modelo_svc.pkl")
            predicciones = modelo.predict(df)

            resultado = [("FGR" if p == 1 else "Normal") for p in predicciones]
        except Exception as e:
            resultado = f"❌ Error: {str(e)}"
    return render_template(
    "modelo_svm.html",
    resultado=metricas_svm,
    grafico_confusion="confusion_matrix_svm.png",
    grafico_comparacion="comparacion_svm.png",
    grafico_residuos="residuos_svm.png"
)


# ============================
# MODELO FCM - Matriz + Exactitud + Gráfico
# ============================
@app.route('/modelo_fcm', methods=['GET', 'POST'])
def modelo_fcm():
    import networkx as nx
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

    resultado = []
    grafico_confusion = None
    grafico_grafo = None
    grafico_comparacion = None

    if request.method == 'POST':
        archivo = request.files['archivo']
        if archivo and archivo.filename.endswith('.xlsx'):
            df = pd.read_excel(archivo)

            # -----------------------
            # Preparación de datos
            # -----------------------
            input_columns = [col for col in df.columns if col != 'C31']
            output_column = 'C31'
            concepts = input_columns + [output_column]

            # Normalización
            df[input_columns] = (df[input_columns] - df[input_columns].min()) / (df[input_columns].max() - df[input_columns].min())

            # -----------------------
            # Matriz de pesos
            # -----------------------
            W = np.zeros((len(concepts), len(concepts)))
            G = nx.DiGraph()

            for i, col_in in enumerate(input_columns):
                correlation = df[col_in].corr(df[output_column])
                j = concepts.index(output_column)
                W[i][j] = correlation

                if not np.isnan(correlation):
                    G.add_edge(col_in, output_column, weight=round(correlation, 2))

            # -----------------------
            # Predicción FCM
            # -----------------------
            def activation(x):
                return 1 / (1 + np.exp(-x))

            def predict_fcm_row(row, W, steps=5):
                state = np.array([row[col] if col != 'C31' else 0 for col in concepts])
                for _ in range(steps):
                    state = activation(np.dot(state, W))
                return state[-1]

            predicciones = []
            for _, row in df.iterrows():
                pred = predict_fcm_row(row, W)
                decision = 1 if pred >= 0.5 else 0
                predicciones.append((pred, decision))

            df['C31_pred'] = [v for v, _ in predicciones]
            df['C31_decision'] = [d for _, d in predicciones]

            # -----------------------
            # Métricas y gráfica
            # -----------------------
            y_true = df['C31'].astype(int)
            y_pred = df['C31_decision']
            accuracy = accuracy_score(y_true, y_pred)
            resultado.append(f'🔍 Exactitud del modelo FCM: {accuracy * 100:.2f}%')

            cm = confusion_matrix(y_true, y_pred)

            # Guardar Matriz de Confusión como imagen
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No FGR", "FGR"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Matriz de Confusión FCM - C31")
            plt.savefig("static/confusion_matrix_fcm.png")
            plt.close()
            grafico_confusion = "confusion_matrix_fcm.png"

            # Guardar el Grafo FCM como imagen
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, arrows=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in G.edges(data=True)})
            plt.title("Grafo FCM generado automáticamente")
            plt.savefig("static/fcm_grafo.png")
            plt.close()
            grafico_grafo = "fcm_grafo.png"

            # Guardar comparación real vs predicho
            plt.figure(figsize=(12, 5))
            plt.plot(df['C31'].values, label='Real (C31)', marker='o')
            plt.plot(df['C31_pred'].values, label='Predicho (C31_pred)', marker='x')
            plt.title("Comparación entre valores reales y predichos (C31)")
            plt.xlabel("Índice de muestra")
            plt.ylabel("Valor normalizado")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("static/comparacion_c31.png")
            plt.close()
            grafico_comparacion = "comparacion_c31.png"

    return render_template("modelo_fcm.html",
                           resultado=resultado,
                           grafico_confusion=grafico_confusion,
                           grafico_grafo=grafico_grafo,
                           grafico_comparacion=grafico_comparacion)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)