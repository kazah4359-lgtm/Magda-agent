import logging
from typing import List, Dict
import warnings

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            DDGS = None
except ImportError:
    DDGS = None

def search_internet(query: str, max_results: int = 5) -> str:
    """
    Выполняет поиск в интернете по заданному запросу.
    Возвращает отформатированную строку с результатами (заголовок, ссылка, фрагмент текста).
    """
    if DDGS is None:
        return "Ошибка: библиотека duckduckgo-search не установлена. Поиск невозможен."

    try:
        results: List[Dict[str, str]] = DDGS().text(query, max_results=max_results)

        if not results:
            return f"По запросу '{query}' ничего не найдено."

        formatted_results = []
        for i, res in enumerate(results, 1):
            title = res.get('title', 'Без заголовка')
            href = res.get('href', 'Нет ссылки')
            body = res.get('body', 'Нет описания')
            formatted_results.append(f"{i}. {title}\n   Ссылка: {href}\n   Описание: {body}")

        return "\n\n".join(formatted_results)
    except Exception as e:
        logging.error(f"Ошибка при поиске в интернете: {e}", exc_info=True)
        return f"Произошла ошибка при выполнении поиска: {e}"
