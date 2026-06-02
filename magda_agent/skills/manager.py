from typing import Dict, List, Callable

class Skill:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

class SkillManager:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self._register_default_skills()

    def _register_default_skills(self):
        self.register_skill(Skill(
            name="web_search",
            description="Search the web for information",
            func=lambda q: f"Searching web for: {q} (Mock result)"
        ))
        self.register_skill(Skill(
            name="code_executor",
            description="Run Python code in a sandbox",
            func=lambda c: f"Executing code: {c} (Mock result)"
        ))

    def register_skill(self, skill: Skill):
        self.skills[skill.name] = skill

    def get_skill(self, name: str) -> Skill:
        return self.skills.get(name)

    def list_skills(self) -> List[Dict[str, str]]:
        return [{"name": s.name, "description": s.description} for s in self.skills.values()]

    def execute_skill(self, name: str, *args, **kwargs):
        skill = self.get_skill(name)
        if skill:
            return skill.func(*args, **kwargs)
        return f"Skill '{name}' not found."
