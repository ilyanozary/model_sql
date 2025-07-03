import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def load_schema(schema_path="schema/pagila_schema.sql"):
    with open(schema_path, "r") as f:
        schema_full = f.read()
    table_defs = re.findall(r'CREATE TABLE .*?;\n', schema_full, re.DOTALL)
    schema_clean = "\n".join(table_defs)
    return schema_clean

def load_sqlcoder(model_id="defog/sqlcoder-7b-2"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    sql_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return sql_generator