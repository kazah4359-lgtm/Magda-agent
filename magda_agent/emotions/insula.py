from typing import Tuple

class Insula:
    """
    Insula (Инсула / Островковая доля) module.
    Responsible for interoception and integrating internal bodily states (like energy and boredom)
    into emotional shifts (valence, arousal, dominance).
    """

    def process_interoception(self, energy: float, boredom: float) -> Tuple[float, float, float]:
        """
        Calculates emotional shifts based on current internal states.

        Args:
            energy (float): Current energy level (0.0 to 1.0)
            boredom (float): Current boredom level (0.0 to 1.0)

        Returns:
            Tuple[float, float, float]: (valence_shift, arousal_shift, dominance_shift)
        """
        valence_shift = 0.0
        arousal_shift = 0.0
        dominance_shift = 0.0

        # Low energy leads to negative valence, lower arousal, and lower dominance
        if energy < 0.3:
            valence_shift -= 0.05
            arousal_shift -= 0.05
            dominance_shift -= 0.05

        # High energy can boost arousal and dominance slightly
        elif energy > 0.8:
            arousal_shift += 0.02
            dominance_shift += 0.02

        # High boredom leads to negative valence, lower arousal, and lower dominance
        if boredom > 0.7:
            valence_shift -= 0.05
            arousal_shift -= 0.02
            dominance_shift -= 0.02

        return valence_shift, arousal_shift, dominance_shift
