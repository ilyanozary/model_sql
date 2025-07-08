from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_schema, load_sqlcoder
import logging
import traceback
from contextlib import asynccontextmanager


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and schema
schema_clean = None
sql_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global schema_clean, sql_generator
    try:
        # ---------- Startup ----------
        logger.info("Starting application…")
        logger.info("Loading schema…")
        schema_clean = load_schema("schema/pagila_schema.sql")

        logger.info("Loading SQLCoder model…")
        sql_generator = load_sqlcoder()

        logger.info("Application ready!")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
    finally:
        # ---------- Shutdown ----------
        logger.info("Shutting down application…")
        schema_clean = None
        sql_generator = None

app = FastAPI(
    title="SQL Generator API",
    description="Generate SQL queries from natural language",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=False,      
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Natural language question to convert to SQL")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": sql_generator is not None,
        "schema_loaded": schema_clean is not None
    }

@app.post("/generate_sql")
async def generate_sql(request: QueryRequest):
    """Generate SQL query from natural language question"""
    try:
        # Check if models are loaded
        if schema_clean is None or sql_generator is None:
            logger.error("Models not properly initialized")
            raise HTTPException(status_code=503, detail="Service not ready. Models are still loading.")

        logger.info(f"Generating SQL for question: {request.question[:100]}...")

        # Create prompt
        prompt = f"""### Postgres SQL tables, with their properties:
{schema_clean}

### Task
{request.question}

### SQL query
"""

        # Generate SQL
        try:
            output = sql_generator(
                prompt,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=sql_generator.tokenizer.eos_token_id
            )
            generated_text = output[0]["generated_text"]
            generated_sql = generated_text.split("### SQL query")[-1].strip()

            # Basic validation
            if not generated_sql:
                raise HTTPException(status_code=422, detail="Failed to generate valid SQL query")

            logger.info("SQL generated successfully")
            return {"sql": generated_sql, "question": request.question}

        except Exception as model_error:
            logger.error(f"Model generation error: {str(model_error)}")
            raise HTTPException(status_code=500, detail="Failed to generate SQL query")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_sql: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")