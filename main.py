import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="LexIA API")

# Libera o acesso para o seu site no GitHub Pages
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

def get_active_model():
    """Busca dinamicamente o primeiro modelo disponível para a chave fornecida."""
    available_models = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
    except Exception as e:
        print(f"Erro ao listar modelos: {e}")

    # Lista de preferência de nomes caso a busca estática seja necessária
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "gemini-1.5-flash"
    ]

    for model in available_models:
        if any(p in model for p in ["flash", "pro"]):
            return model
            
    return available_models[0] if available_models else "models/gemini-1.5-flash"

@app.get("/")
def health_check():
    return {"status": "LexIA Backend está online!"}

@app.post("/api/chat")
def chat(request: QueryRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Chave GEMINI_API_KEY não configurada no Render.")
    
    selected_model = get_active_model()
    
    try:
        model = genai.GenerativeModel(
            model_name=selected_model,
            system_instruction=SYSTEM_PROMPT
        )
        response = model.generate_content(request.prompt)
        return {"response": response.text}
    except Exception as e:
        # Tentativa de fallback sem system_instruction
        try:
            model = genai.GenerativeModel(selected_model)
            response = model.generate_content(f"{SYSTEM_PROMPT}\n\nPergunta do usuário: {request.prompt}")
            return {"response": response.text}
        except Exception as err:
            raise HTTPException(
                status_code=500, 
                detail=f"Erro no modelo [{selected_model}]: {str(err)}"
            )
