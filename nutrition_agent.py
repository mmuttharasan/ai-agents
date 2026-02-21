from pathlib import Path

import chromadb
from agents import (
    Agent,
    function_tool,
)

# Chroma DB path (assumes a `chroma` folder at the repository root)
chroma_path = Path(__file__).parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
nutrition_db = chroma_client.get_collection(name="nutrition_db")


@function_tool
def calorie_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Look up calorie information for a food item from the RAG database.

    Args:
        query: The food item to look up.
        max_results: Maximum number of results to return.

    Returns:
        A short, formatted string with calorie information or a not-found message.
    """

    results = nutrition_db.query(query_texts=[query], n_results=max_results)

    if not results or not results.get("documents") or not results["documents"][0]:
        return f"No nutrition information found for: {query}"

    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        food_item = metadata.get("food_item", "Unknown").title()
        calories = metadata.get("calories_per_100g", "Unknown")
        category = metadata.get("food_category", "Unknown").title()

        formatted_results.append(
            f"{food_item} ({category}): {calories} calories per 100g"
        )

    return "Nutrition Information:\n" + "\n".join(formatted_results)


nutrition_agent = Agent(
    name="Nutrition Assistant",
    instructions="""
    You are a helpful, concise nutrition assistant.
    - Provide short, factual calorie and basic nutrition information.
    - When the user asks about a specific food item's calories, use the
      `calorie_lookup_tool` to retrieve authoritative values from the RAG DB.
    - Do not provide medical, diagnostic, or personalized medical advice.
    - If the user asks about a full meal or recipe, ask clarifying questions
      (serving size, ingredients, preparation) before estimating calories.
    - If data is missing, say you couldn't find the information and offer
      suggestions for clarifying the query.
    """,
    tools=[calorie_lookup_tool],
)
