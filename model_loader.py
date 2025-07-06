import re
import os
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

def load_schema(schema_path="schema/pagila_schema.sql"):
    """Load and clean database schema from SQL file"""
    try:
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_full = f.read()

        if not schema_full.strip():
            raise ValueError("Schema file is empty")

        # Extract CREATE TABLE statements
        table_defs = re.findall(r'CREATE TABLE .*?;\n', schema_full, re.DOTALL)

        if not table_defs:
            logger.warning("No CREATE TABLE statements found in schema")

        schema_clean = "\n".join(table_defs)
        logger.info(f"Loaded schema with {len(table_defs)} tables")
        return schema_clean

    except Exception as e:
        logger.error(f"Failed to load schema from {schema_path}: {str(e)}")
        raise

def load_sqlcoder(model_id="defog/sqlcoder-7b-2"):
    """Load SQLCoder model and tokenizer"""
    try:
        logger.info(f"Loading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading model {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )

        logger.info("Creating text generation pipeline...")
        sql_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False
        )

        logger.info("SQLCoder model loaded successfully")
        return sql_generator

    except Exception as e:
        logger.error(f"Failed to load SQLCoder model {model_id}: {str(e)}")
        raise