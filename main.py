import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="LexIA API")

# Permite chamadas de qualquer origem (inclusive GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
    
    # Ordem de tentativa dos identificadores oficiais da API
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "gemini-pro"
    ]
    
    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT
            )
            response = model.generate_content(request.prompt)
            return {"response": response.text}
        except Exception as e:
            last_error = str(e)
            continue
            
    # Tentativa de fallback simplificado sem system_instruction
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"{SYSTEM_PROMPT}\n\nConsulta: {request.prompt}")
        return {"response": response.text}
    except Exception as e:
        last_error = str(e)

    raise HTTPException(status_code=500, detail=f"Erro ao conectar ao Gemini: {last_error}")
