from ocelot import ocelot_generation, ocelot_load_models
import soundfile as sf
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.inference.interfaces import foreign_class
from speechbrain.utils.metric_stats import ErrorRateStats
import random
from tqdm import tqdm
from phonemizer import phonemize


def main():
    # audio_list = []
    # text_list = []

    # with open("filelists/ljs_audio_text_train_filelist.txt", "r") as f:
    #     lll = f.readlines()
    #     for lst in lll:
    #         audio_list.append(lst.split('|')[0])
    #         text_list.append(lst.split('|')[1])

    # all_texts = random.sample(text_list, 1000)
    # all_texts = [text.strip() for text in all_texts]
    # print(all_text)
    # phonemes = phonemize(t, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    # print(t)
    # print(phonemes)
    # exit(0)

    all_texts = []
    with open('test_sentences_english.txt', 'r') as f:
        all_texts.extend(f.readlines())

    all_texts = [text.strip() for text in all_texts]
    all_texts = [s.replace('[', '').replace(']', '') for s in all_texts]
    all_texts = [s.upper() for s in all_texts]
    sampling_rate = 16000


    chinese_model, english_model = ocelot_load_models()
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="/data1/jiyuyu/speechbrain/pretrained_models/asr-transformer-transformerlm-librispeech")

    # calculate rtf and wer
    total_inference_time = 0
    total_audio_duration = 0
    wer_stats = ErrorRateStats()
    predicted_list = []

    for text_input in tqdm(all_texts, desc="Processing"):
        _, audio_output, *rest, inference_time = ocelot_generation(english_model, 'English', text_input, 0, 'neutral')
        
        with sf.SoundFile(audio_output) as audio_file:
            frames = audio_file.frames
            sampling_rate = audio_file.samplerate
            audio_duration = frames / sampling_rate

        try:
            predicted_text = asr_model.transcribe_file(audio_output)
            predicted_list.append(predicted_text)
        except Exception as e:
            print(f"Error processing audio for WER calculation: {e}")

        total_inference_time += float(inference_time)
        total_audio_duration += audio_duration

    average_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0
    wer_stats.append(list(range(len(all_texts))), predicted_list, all_texts)
    with open('./cer_english.txt', "w", encoding="utf-8") as w:
        wer_stats.write_stats(w)

    with open("RTF_english.txt", "w") as file:
        file.write(f"Overall Average RTF: {average_rtf}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()