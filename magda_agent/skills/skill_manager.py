import os
import importlib
import sys

class SkillManager:
    """
    Управление навыками агента.
    Позволяет динамически загружать, выполнять и создавать новые навыки (Python-скрипты).
    """
    def __init__(self, skills_dir="magda_agent/skills"):
        self.skills_dir = skills_dir
        self.skills = {}
        self.load_all_skills()

    def load_all_skills(self):
        """Динамически загружает все .py файлы из папки skills."""
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)

        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                self._load_skill(module_name)

    def _load_skill(self, module_name: str):
        full_module_name = f"magda_agent.skills.{module_name}"
        try:
            if full_module_name in sys.modules:
                module = importlib.reload(sys.modules[full_module_name])
            else:
                module = importlib.import_module(full_module_name)

            # Ищем функцию execute в модуле
            if hasattr(module, 'execute'):
                self.skills[module_name] = module.execute
                print(f"Loaded skill: {module_name}")
        except Exception as e:
            print(f"Failed to load skill {module_name}: {e}")

    def create_new_skill(self, skill_name: str, code: str) -> str:
        """
        Самоулучшение: Агент может написать код нового навыка и сохранить его.
        """
        # Базовая защита: убедимся, что имя файла безопасно
        safe_name = "".join([c for c in skill_name if c.isalnum() or c == '_'])

        if not safe_name:
            return "Error: Invalid skill name provided."

        filepath = os.path.join(self.skills_dir, f"{safe_name}.py")

        # Предварительная проверка синтаксиса
        try:
            compile(code, filepath, 'exec')
        except SyntaxError as se:
            return f"Error creating skill: Syntax Error in code. {se}"

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)
            self._load_skill(safe_name)
            if safe_name in self.skills:
                return f"Skill '{safe_name}' successfully created and loaded."
            else:
                return f"Error: Skill '{safe_name}' created but 'execute' function was not found."
        except Exception as e:
            return f"Error creating skill: {e}"

    def execute_skill(self, skill_name: str, *args, **kwargs) -> str:
        if skill_name in self.skills:
            try:
                return str(self.skills[skill_name](*args, **kwargs))
            except Exception as e:
                return f"Error executing skill {skill_name}: {e}"
        return f"Skill '{skill_name}' not found."
