from flask import Flask, request, jsonify
import numpy as np
from scipy.stats import poisson

app = Flask(__name__)

def coletar_dados_simulados(time):
    # Retorna apenas os valores médios (evita tuplas)
    return {
        "xG": 1.6,
        "posse": 55.2,
        "chutes": 13.5,
        "finalizacoes": 6.8,
        "escanteios": 4.3,
        "cartoes": 2.1,
        "passes": 480,
        "faltas": 13.2,
        "lesoes": ["atacante"]  # apenas exemplo
    }

def ajustar_por_fator_casa(stats):
    return {
        "xG": stats["xG"] * 1.15,
        "posse": stats["posse"] * 1.05,
        "chutes": stats["chutes"] * 1.10
    }

def ajustar_por_lesoes(stats, lesoes):
    fator = 0.90 if "atacante" in lesoes else 1.0
    return {k: (v * fator if isinstance(v, (int, float)) else v) for k, v in stats.items()}


def previsao_poisson(golsA, golsB):
    max_gols = 5
    prob_matrix = np.zeros((max_gols, max_gols))
    for i in range(max_gols):
        for j in range(max_gols):
            prob_matrix[i][j] = poisson.pmf(i, golsA) * poisson.pmf(j, golsB)
    empate = sum(prob_matrix[i][i] for i in range(max_gols))
    index = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    return {
        "placar_mais_provavel": f"{index[0]} x {index[1]}",
        "chance_empate": round(empate * 100, 1)
    }

@app.route('/prever', methods=['POST'])
def prever():
    data = request.json
    time_a = data.get("time_a")
    time_b = data.get("time_b")

    stats_a = coletar_dados_simulados(time_a)
    stats_b = coletar_dados_simulados(time_b)

    ofensivo_a = ajustar_por_lesoes(ajustar_por_fator_casa(stats_a), stats_a["lesoes"])
    defensivo_b = ajustar_por_lesoes(stats_b, stats_b["lesoes"])

    xG_A = ofensivo_a["xG"] * 0.7 + (2 - defensivo_b["xG"]) * 0.3
    xG_B = defensivo_b["xG"] * 0.7 + (2 - ofensivo_a["xG"]) * 0.3

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
