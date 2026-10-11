import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI(title="LexIA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Chave GEMINI_API_KEY não configurada no Render.")
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nConsulta do usuário: {request.prompt}"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no Gemini: {str(e)}")
