# How to run


## Install coqui-ai/TTS
https://github.com/idiap/coqui-ai-TTS

install like below:

```
$ git clone https://github.com/idiap/coqui-ai-TTS
$ cd coqui-ai-TTS
$ pip install -e .
```


## Generate Speech
python3 synthesize.py --model xtts

In this work, the LibriSpeech train-clean-100 should be on `../asr/dataset/LibriSpeech/LibriSpeech`


The synthetic samples will be at `./synthetic/`

## Fine-tuning
For fine-tuning,
```
$ cd `coqui-ai-TTS/recipes/`
$ mkdir `mlcl`
$ cd ../../
$ cp `./train_gpt_xtts.py` `./coqui-ai-TTS/recipes/mlcl/`
$ cd `coqui-ai-TTS/`
$ python3 recipes/mlcl/train_gpt_xtts.py
```
For TTS generation (evaluation), refer to `synthesize_with_ours.py`
```
python synthesize_with_ours.py
```
You should change the `CONFIG_PATH`, `TOKENIZER_PATH`, `XTTS_CHECKPOINT`, `SPEAKER_REFERENCE`, `OUTPUT_WAV_PATH`, and words


## Miscellaneous
For generating LibriSpeech, you should have to change the XTTS's character limites as shown in  

https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/tokenizer.py#L598

So, open /TTS/tts/layers/xtts/tokenizer.py and change 250 of "en" as 400 or 500 
