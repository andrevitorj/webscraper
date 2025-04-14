from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup
import numpy as np
from scipy.stats import poisson
import re

app = Flask(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0"}

def encontrar_url_time_fbref(time_nome):
    pesquisa_url = f"https://fbref.com/en/search/search.fcgi?search={time_nome.replace(' ', '+')}"
    resp = requests.get(pesquisa_url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    link = soup.find("div", class_="search-item-url")
    if link:
        href = link.text.strip()
        return f"https://fbref.com{href}"
    return None

def extrair_stats_dos_jogos(url_time):
    resp = requests.get(url_time, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")
    tabela = soup.find("table", id="matchlogs_for")

    stats = {
        "xG": [],
        "chutes": [],
        "posse": [],
        "finalizacoes": [],
        "escanteios": [],
        "cartoes": [],
        "passes": [],
        "faltas": []
    }

    if not tabela:
        return None

    linhas = tabela.find_all("tr", class_=lambda x: x != "thead")[:10]
    for linha in linhas:
        col = linha.find_all("td")
        if not col:
            continue
        try:
            stats["xG"].append(float(col[-2].text))  # xG
            stats["chutes"].append(int(col[9].text))
            stats["posse"].append(float(col[7].text.replace("%", "")))
            stats["finalizacoes"].append(int(col[11].text))
            stats["escanteios"].append(int(col[17].text))
            stats["cartoes"].append(int(col[19].text))
            stats["passes"].append(int(col[13].text))
            stats["faltas"].append(int(col[18].text))
        except:
            continue

    return stats

def media_ponderada(valores):
    if not valores:
        return 0
    pesos = [0.1]*5 + [0.3]*5
    if len(valores) < 10:
        valores = [valores[-1]] * (10 - len(valores)) + valores
    return round(np.average(valores[-10:], weights=pesos), 2)

def extrair_stats_fbref(time_nome):
    url_time = encontrar_url_time_fbref(time_nome)
    if not url_time:
        return None
    stats_jogos = extrair_stats_dos_jogos(url_time)
    if not stats_jogos:
        return None

    return {
        k: media_ponderada(v) for k, v in stats_jogos.items()
    } | {"lesoes": []}

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

    stats_a = extrair_stats_fbref(time_a)
    stats_b = extrair_stats_fbref(time_b)

    if not stats_a or not stats_b:
        return jsonify({"erro": "Não foi possível obter estatísticas reais dos times."}), 400

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
        "fonte": "FBref.com (scraping real dos últimos 10 jogos)"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
