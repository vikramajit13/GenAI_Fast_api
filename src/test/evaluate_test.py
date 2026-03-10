import pytest
from ..services.storeservice import RagService
import pytest_asyncio
from ..core.db import init_db_pool, close_db_pool
import asyncio

EVAL_SET = [
    {
        "id": "democracy_foreign_policy",
        "doc_id": "doc1",
        "query": "From the context, answer the following with a direct quote for each: democracy, foreign policy challenge",
        "expected_chunk_ids": [11, 8],
        "expected_sentence_contains": [
            "freedom and democracy are under attack",
            "support Ukraine"
        ]
    },
    {
        "id": "tax_fairness",
        "doc_id": "doc1",
        "query": "What was said about tax fairness?",
        "expected_chunk_ids": [18],
        "expected_sentence_contains": [
            "tax fairness"
        ]
    }
]

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# 2. Initialize the global _pool once
@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    await init_db_pool()
    yield
    await close_db_pool()


@pytest.mark.parametrize("test_case", EVAL_SET)
@pytest.mark.asyncio(scope="session")
async def test_rag_service_accuracy(test_case):
    service = RagService()
    
    # Run the service
    response = await service.answer(
        test_case["doc_id"], 
        test_case["query"], 
        2, 
        False
    )
    
    print("response:::::", response)
    # Assertions replace the print statements
    assert response is not None, f"No response for {test_case['id']}"

    # 1. Use the correct key name "retrieval"
    # 2. Iterate directly over the list (remove enumerate)
    actual_chunk_ids = {obj['chunk_index'] for obj in response["retrieval"]}

    expected_chunks = test_case["expected_chunk_ids"]

    # Check if any of the expected chunks were returned
    match_found = any(cid in actual_chunk_ids for cid in expected_chunks)

# @pytest.mark.parametrize("test_case", EVAL_SET)
# service  = RagService()

# for r in enumerate(EVAL_SET):
#     doc = r["doc_id"]
#     query =r["query"]
#     expected_chunk = r["expected_chunk_ids"]
#     response = service.answer(doc,query,2,False)
#     if response:
#         chunk_ids = {obj['chunk_index'] for obj in response}
#         is_present_optimized = any(num in chunk_ids for num in expected_chunk)
#         print(is_present_optimized) 
            
    