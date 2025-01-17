from funasr import AutoModel
import argparse

parser = argparse.ArgumentParser()

class ASR:
    def __init__(self):
        self.model = AutoModel(model="paraformer-zh")

    def transcribe_from_file(self, filename):
        raw_result = self.model.generate(filename)[0]

        result = {}
        result["characters"] = raw_result.get("text", "").split()
        result["timestamps"] = raw_result.get("timestamp", [])
        assert len(result["characters"]) == len(
            result["timestamps"]
        ), "The number of characters and timestamps do not match."
        return result

if __name__ == "__main__":
    asr_recognizer = ASR()
    if asr_recognizer is None:
        raise Exception("ASR model could not be loaded.")
    else:
        pass

    parser.add_argument('--filename', type=str, required=True)
    args = parser.parse_args()

    result = asr_recognizer.transcribe_from_file(args.filename)
    print(result)
