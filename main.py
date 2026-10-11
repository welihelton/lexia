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
Responda de forma objetiva, clara e direta ao ponto.
Lembre o usuário de que a resposta não substitui um advogado.
"""

@app.get("/")
def health_check():
    return {"status": "LexIA Backend está online!"}

@app.post("/api/chat")
def chat(request: QueryRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada.")
    
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
        "max_tokens": 1024,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        data = response.json()
        
        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", "Erro na API do Groq.")
            raise HTTPException(status_code=500, detail=f"Erro Groq: {error_msg}")
            
        return {"response": data["choices"][0]["message"]["content"]}
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Tempo limite esgotado ao consultar a IA.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
