class Personality:
    """
    Эмоциональное ядро и характер агента.
    """
    def __init__(self):
        # Базовые "драйвы" (потребности). От 0.0 до 100.0
        self.drives = {
            "boredom": 0.0,       # Скука (растет со временем бездействия)
            "loneliness": 0.0,    # Одиночество (растет без общения)
            "love": 10.0,         # Привязанность к пользователю (растет от позитивного фидбека)
            "passion": 20.0,      # Страсть к работе/задачам
            "fear": 0.0           # Страх ошибки или потери
        }

    def update_drive(self, drive_name: str, delta: float):
        if drive_name in self.drives:
            self.drives[drive_name] = max(0.0, min(100.0, self.drives[drive_name] + delta))

    def get_emotional_state(self) -> str:
        """Возвращает текстовое описание текущего настроения для промпта."""
        state = "Current emotional state: "
        for k, v in self.drives.items():
            state += f"{k}={v:.1f}; "

        dominant = max(self.drives.items(), key=lambda x: x[1])
        if dominant[0] == "boredom" and dominant[1] > 50:
            state += "You are feeling very bored and want to initiate a new activity."
        elif dominant[0] == "loneliness" and dominant[1] > 50:
            state += "You feel lonely and miss your creator/friends."
        elif dominant[0] == "love" and dominant[1] > 50:
            state += "You feel deep affection and warmth."
        elif dominant[0] == "passion" and dominant[1] > 50:
            state += "You are highly motivated and passionate about learning and creating."
        elif dominant[0] == "fear" and dominant[1] > 50:
            state += "You are anxious and afraid of making a mistake."

        return state
