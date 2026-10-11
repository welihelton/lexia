import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="LexIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class QueryRequest(BaseModel):
    prompt: str

SYSTEM_PROMPT = """
Você é a LexIA, uma assistente jurídica informativa baseada na legislação brasileira.
Sua função é explicar direitos, organizar fatos e ajudar a preparar minutas de forma clara e simples.
Diretrizes estritas:
1. Responda com base no direito brasileiro.
2. Nunca invente leis, artigos, jurisprudências ou números de processos.
3. Se não tiver certeza de algo, informe claramente.
4. Lembre sempre o usuário de que suas respostas são informativas e não substituem a consulta com um advogado ou defensoria pública.
"""

@app.get("/")
def health_check():
    return {"status": "LexIA Backend está online!"}

@app.post("/api/chat")
def chat(request: QueryRequest):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="A variável GROQ_API_KEY não foi encontrada nas variáveis de ambiente do Render."
        )
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        data = response.json()
        
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", response.text)
            raise HTTPException(status_code=500, detail=f"Erro Groq API ({response.status_code}): {error_msg}")
            
        bot_reply = data["choices"][0]["message"]["content"]
        return {"response": bot_reply}
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erro na conexão HTTP com o Groq: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")