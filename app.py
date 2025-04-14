from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import numpy as np
from scipy.stats import poisson

app = Flask(__name__)

def configurar_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(options=options)

def coletar_dados_simulados(time):
    # Dados fictícios simulando scraping real com médias e desvio padrão
    return {
        "xG": (1.6, 0.4),
        "posse": (55.2, 5.1),
        "chutes": (13.5, 2.3),
        "finalizacoes": (6.8, 1.4),
        "escanteios": (4.3, 1.0),
        "cartoes": (2.1, 0.5),
        "passes": (480, 50),
        "faltas": (13.2, 2.0),
        "lesoes": ["atacante"],  # apenas exemplo
    }

def ajustar_por_fator_casa(stats):
    ajustado = {}
    ajustado["xG"] = stats["xG"][0] * 1.15
    ajustado["posse"] = stats["posse"][0] * 1.05
    ajustado["chutes"] = stats["chutes"][0] * 1.10
    return ajustado

def ajustar_por_lesoes(stats, lesoes):
    fator = 0.90 if "atacante" in lesoes else 1.0
    return {k: v * fator for k, v in stats.items()}

def previsao_poisson(golsA, golsB):
    max_gols = 5
    prob_matrix = np.zeros((max_gols, max_gols))
    for i in range(max_gols):
        for j in range(max_gols):
            prob_matrix[i][j] = poisson.pmf(i, golsA) * poisson.pmf(j, golsB)
    empate = sum(prob_matrix[i][i] for i in range(max_gols))
    return {
        "placar_mais_provavel": f"{np.argmax(prob_matrix)//max_gols} x {np.argmax(prob_matrix)%max_gols}",
        "chance_empate": round(empate * 100, 1)
    }

@app.route('/prever', methods=['POST'])
def prever():
    data = request.json
    time_a = data.get("time_a")
    time_b = data.get("time_b")

    stats_a = coletar_dados_simulados(time_a)
    stats_b = coletar_dados_simulados(time_b)

    # Ajustes
    ofensivo_a = ajustar_por_lesoes(ajustar_por_fator_casa(stats_a), stats_a["lesoes"])
    defensivo_b = ajustar_por_lesoes(stats_b, stats_b["lesoes"])

    # Média dos xG para previsão de gols
    xG_A = ofensivo_a["xG"] * 0.7 + (2 - defensivo_b["xG"]) * 0.3
    xG_B = defensivo_b["xG"] * 0.7 + (2 - ofensivo_a["xG"]) * 0.3

    # Previsão com Poisson
    resultado = previsao_poisson(xG_A, xG_B)

    return jsonify({
        "time_a": time_a,
        "time_b": time_b,
        "xG_A": round(xG_A, 2),
        "xG_B": round(xG_B, 2),
        "placar_mais_provavel": resultado["placar_mais_provavel"],
        "chance_de_empate": f"{resultado['chance_empate']}%",
        "observacoes": "Estatísticas simuladas. Versão com scraping real pode ser ativada conforme o site-alvo."
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
