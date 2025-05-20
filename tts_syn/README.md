# How to run


## Install coqui-ai/TTS

https://github.com/coqui-ai/TTS

please install like below:

```
$ git clone https://github.com/coqui-ai/TTS
$ pip install -e .[all,dev,notebooks]  # Select the relevant extras
```


## Generate Speech
python3 synthesize.py --model xtts

In this work, the LibriSpeech train-clean-100 should be on `../asr/dataset/LibriSpeech/LibriSpeech`


The synthetic samples will be at `./synthetic/`


