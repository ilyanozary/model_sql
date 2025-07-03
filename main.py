from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_schema, load_sqlcoder

app = FastAPI()
print("create app")
print("load schema")
schema_clean = load_schema("schema/pagila_schema.sql")
print("load sqlcoder")
sql_generator = load_sqlcoder()
print("app ready")

class QueryRequest(BaseModel):
    question: str

@app.post("/generate_sql")
async def generate_sql(request: QueryRequest):
    prompt = f"""### Postgres SQL tables, with their properties:
{schema_clean}

### Task
{request.question}

### SQL query
"""
    output = sql_generator(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    generated_sql = output.split("### SQL query")[-1].strip()
    return {"sql": generated_sql}