from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import RuM2M100ModelForSpellingCorrection
from abc import abstractmethod, ABC


class Corrector(ABC):
    @abstractmethod
    def __init__(self, gpu=False, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def correct(self, text: str) -> str:
        pass

    def batch_correct(self, texts: tuple[str], batch_size=1) -> tuple[str]:
        batch_result = (self.correct(text) for text in texts)     

        return batch_result

class SageCorrector(Corrector):
    def __init__(self, gpu=False, *args, **kwargs) -> None:
        self.corrector_m2m = RuM2M100ModelForSpellingCorrection.from_pretrained(AvailableCorrectors.m2m100_418M.value)
    
    def correct(self, text: str) -> str:
        result = self.corrector_m2m.correct(text)
        return result[0]
    
class YandexTranslateCorrector(Corrector):
    def __init__(self, gpu=False, *args, **kwargs) -> None:
        super().__init__(gpu, *args, **kwargs)

    def correct(self, text: str) -> str:
        return super().correct(text)
