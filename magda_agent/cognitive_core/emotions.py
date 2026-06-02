def calculate_reward(success: bool, effort: float) -> float:
    """
    Математический расчет "эмоции" (подкрепления).
    """
    if success:
        return 1.0 * effort # Радость, удовлетворение
    else:
        return -0.5 * effort # Фрустрация, страх ошибки
