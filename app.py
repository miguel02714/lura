import json
import time
import difflib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Carregamento da Base de Conhecimento ---
def load_knowledge_base(file_path="bases_data.json"):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

KNOWLEDGE_BASE = load_knowledge_base()

# --- Funções Auxiliares de Similaridade e RAG Avançado ---

def normalize_text(text):
    """Normaliza o texto para buscas mais inteligentes."""
    return text.lower().strip()

def semantic_match(query, kb, threshold=0.5):
    """
    Simula busca semântica usando difflib para encontrar correspondências aproximadas.
    Retorna as melhores respostas e fontes.
    """
    query_norm = normalize_text(query)
    results = []

    for source_name, source_data in kb.items():
        for question, answer in source_data['data'].items():
            score = difflib.SequenceMatcher(None, query_norm, question.lower()).ratio()
            if score >= threshold:
                results.append({
                    "score": score,
                    "question": question,
                    "answer": answer,
                    "source": source_data['source']
                })
    # Ordena por score decrescente
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def generate_response(query):
    """
    Pipeline avançado RAG + geração de respostas estilo LLM.
    Combina múltiplas respostas se necessário e gera texto natural.
    """
    matches = semantic_match(query, KNOWLEDGE_BASE, threshold=0.4)

    if not matches:
        return {
            "status": "FALHA",
            "response": (
                "Desculpe, não encontrei uma resposta precisa na base de conhecimento. "
                "Tente reformular sua pergunta ou pergunte sobre **SquareOS**, **LINEAX**, **Segurança**, ou outros módulos."
            ),
            "source": "Nenhuma Fonte Encontrada"
        }

    # Se houver múltiplas correspondências próximas, combine respostas
    top_matches = matches[:3]  # pega até 3 melhores matches
    combined_answer = ""
    sources = []

    for match in top_matches:
        combined_answer += f"- {match['answer']}\n"
        sources.append(match['source'])

    # Remove fontes duplicadas
    sources = list(set(sources))

    # Simula geração de linguagem natural
    response_text = (
        f"{combined_answer}\n\n"
        f"Fonte(s) de referência: {', '.join(sources)}"
    )

    return {
        "status": "COMPLETO",
        "response": response_text,
        "source": ", ".join(sources)
    }

# --- Rotas Flask ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()

        if not user_query:
            return jsonify({"error": "Consulta vazia"}), 400

        # Simula latência de LLM e RAG
        time.sleep(0.3)  # retrieval
        result = generate_response(user_query)
        time.sleep(0.3)  # generation

        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Erro no chat: {e}")
        return jsonify({
            "status": "ERRO INTERNO",
            "response": "Ocorreu um erro inesperado no console RAG/LLM.",
            "source": "Sistema"
        }), 500

# --- Rodando o App ---
if __name__ == '__main__':
    app.run(debug=True)
