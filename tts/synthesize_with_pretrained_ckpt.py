from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

#tts.tts_to_file(text="IN THE LIKE MANNER FORTUNE DOES WITH FLORENCE THEREFORE SHOULD NOT APPEAR A MARVELLOUS THING WHAT I SHALL SAY OF THE GREAT FLORENTINES OF WHOM THE FAME IS HIDDEN IN THE PAST I SAW THE UGHI", file_path="./output.wav", speaker_wav="/NasData/home/rsh/SR/mlcl_handover/asr/dataset/LibriSpeech/LibriSpeech/train-clean-100/87/121553/87-121553-0097.flac", language="en")

tts.tts_to_file(text="Hello World!", file_path="./output.wav", speaker_wav="/home2/jw/workspace/asr/mlcl_handover/asr/dataset/LibriSpeech/LibriSpeech/train-clean-100/87/121553/87-121553-0097.flac", language="en")