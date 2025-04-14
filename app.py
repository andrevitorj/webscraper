from flask import Flask, request, jsonify
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.json.get('url')
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    titulo = soup.title.string if soup.title else "Sem título"
    return jsonify({'titulo': titulo})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
