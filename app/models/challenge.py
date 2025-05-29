# app/models/challenge.py

class ChallengeData:
    """
    A data class to hold information about a single challenge instance.
    This can be expanded as needed.
    For now, it's a placeholder.
    """
    def __init__(self, topic: str = None):
        self.topic: str = topic
        self.image1_pil = None # To store PIL.Image object
        self.image2_pil = None # To store PIL.Image object
        self.caption1: str = None
        self.caption2: str = None
        self.result: str = None

    def __repr__(self):
        return f"<ChallengeData topic='{self.topic}'>"
