def execute(code: str) -> str:
    """
    Базовый системный навык.
    Выполняет безопасный Python-код в изолированном пространстве (eval).
    Внимание: в реальном production-коде здесь должен быть строгий sandbox (например, Docker/gVisor).
    """
    try:
        # Для простоты реализации используем exec с ограниченными глобальными переменными.
        # Вывод перехватываем через StringIO
        import io
        import sys

        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()

        # Опасная зона: выполнение кода, сгенерированного LLM
        exec(code, {"__builtins__": __builtins__}, {})

        return redirected_output.getvalue()
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        sys.stdout = old_stdout
