from sage.spelling_correction import AvailableCorrectors
from sage.spelling_correction import RuM2M100ModelForSpellingCorrection
from abc import abstractmethod, ABC
import requests

class Corrector(ABC):
    @abstractmethod
    def __init__(self, gpu=False, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def correct(self, text: str) -> str:
        pass

    def batch_correct(self, texts: tuple[str], batch_size=1) -> tuple[str]:
        batch_result = tuple(self.correct(text) for text in texts)     

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

    def send_text(self, texts: tuple[str]):
        url = "https://speller.yandex.net/services/spellservice.json/checkTexts"
        params = {
            "sid": "5b74521.6676d4ba.8d57b981.74722d74657874",
            "srv": "tr-text",
            "yu": "1964372131718528356",
            "yum": "1718542772623236876"
        }

        data = {
            "text": texts,
            "lang": "ru",
            "options": "516"
        }

        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "ru,en;q=0.9,sr;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://translate.yandex.ru",
            "Referer": "https://translate.yandex.ru/?source_lang=uk&target_lang=be",
            "Sec-Ch-Ua": '"Chromium";v="124", "YaBrowser";v="24.6", "Not-A.Brand";v="99", "Yowser";v="2.5"',
            "Sec-Ch-Ua-Arch": '"x86"',
            "Sec-Ch-Ua-Bitness": '"64"',
            "Sec-Ch-Ua-Full-Version-List": '"Chromium";v="124.0.6367.71", "YaBrowser";v="24.6.0.1874", "Not-A.Brand";v="99.0.0.0", "Yowser";v="2.5"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Ch-Ua-Platform-Version": '"15.0.0"',
            "Sec-Ch-Ua-Wow64": "?0",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 YaBrowser/24.6.0.0 Safari/537.36"
        }

        response = requests.post(url, params=params, headers=headers, data=data)
        spell_response = response.json()
        if spell_response:
            return spell_response
        else:
            return texts

    def correct(self, text: str) -> str:
        corrected_texts = self.send_text((text,))
        corrected_line = text

        for correction in corrected_texts[0]:
            incorrect_word = correction['word']
            correct_word = correction['s'][0]
            
            corrected_line = corrected_line.replace(incorrect_word, correct_word)
        
        return corrected_line
    
    def batch_correct(self, texts: tuple[str], batch_size=1) -> tuple[str]:
        corrected_texts = self.send_text(texts)
        corrected_lines = []

        for original_line, corrections in zip(texts, corrected_texts):
            corrected_line = original_line

            for correction in corrections:
                incorrect_word = correction['word']
                correct_word = correction['s'][0]

                corrected_line = corrected_line.replace(incorrect_word, correct_word)
                
            corrected_lines.append(corrected_line)

        return tuple(corrected_lines)
