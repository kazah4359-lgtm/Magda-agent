import chromadb
from chromadb.config import Settings
import uuid
import datetime
import asyncio
import os
import openai
from openai import AsyncOpenAI

class MemoryManager:
    """
    Управление краткосрочной и долгосрочной памятью (RAG).
    """
    def __init__(self, db_path="./chroma_db"):
        self.short_term = []  # Список словарей {"role": ..., "content": ...}
        self.max_short_term_len = 10  # Максимальное количество сообщений в кратковременной памяти

        # Инициализация долгосрочной памяти (ChromaDB)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="long_term_memory")

    def add_short_term(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.max_short_term_len:
            # Извлекаем старые сообщения для консолидации (summarization)
            old_messages = self.short_term[:5]
            self.short_term = self.short_term[5:]

            # Запускаем фоновую консолидацию памяти (аналог сна/рефлексии)
            asyncio.create_task(self.consolidate_memory(old_messages))

    def get_short_term_context(self) -> list:
        return self.short_term

    async def consolidate_memory(self, messages: list):
        """
        Фоновый процесс: сжимает старые сообщения и сохраняет суть в долгосрочную память.
        """
        if not messages:
            return

        dialogue = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = (
            "Summarize the following conversation segment. Extract only the most important facts, "
            "emotions, and context that the AI should remember long-term. "
            f"Conversation:\n{dialogue}"
        )

        try:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            summary = response.choices[0].message.content

            # Сохраняем в ChromaDB
            doc_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()

            self.collection.add(
                documents=[summary],
                metadatas=[{"timestamp": timestamp, "type": "summary"}],
                ids=[doc_id]
            )
            print(f"Memory consolidated: {summary}")
        except Exception as e:
            print(f"Error in memory consolidation: {e}")

    def retrieve_long_term(self, query: str, n_results: int = 2) -> str:
        """
        Поиск релевантных воспоминаний по запросу.
        """
        if self.collection.count() == 0:
            return ""

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )

        if results and results['documents'] and results['documents'][0]:
            memories = "\n- ".join(results['documents'][0])
            return f"Relevant past memories:\n- {memories}"
        return ""
