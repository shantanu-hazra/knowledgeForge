from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid

from api.models import RetrieveRequest

from rag.ingestion import get_ingestion_pipeline

router = APIRouter(prefix="/documents")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    document_id = str(uuid.uuid4())

    try:
        # Save uploaded file
        extension = Path(file.filename).suffix
        saved_path = UPLOAD_DIR / f"{document_id}{extension}"

        with open(saved_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Feed document into RAG pipeline
        ingestion_pipeline = get_ingestion_pipeline()
        result=ingestion_pipeline.ingest(file_path=saved_path, document_id=document_id)

        if result.status == "failed":
            raise HTTPException(
                status_code=500,
                detail=result.error
            )

        return {
            "document_id": result.document_id,
            "status": result.status,
            "num_chunks": result.num_chunks
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/retrieve")
def retrieve_document(request: RetrieveRequest):

    from rag.retriever import run_retrieval

    print(f"Received retrieval request for query: {request.query}")

    results = run_retrieval(request.query)

    print(results)
    return {
        "query": request.query,
        "results": [
            {
                "score": chunk.score,
                "document": chunk.document,
                "metadata": chunk.metadata,
            }
            for chunk in results
        ]
    }