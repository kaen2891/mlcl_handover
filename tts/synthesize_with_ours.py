import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("/home2/jw/workspace/asr/mlcl_handover/tts/coqui-ai-TTS/recipes/mlcl/run/training/GPT_XTTS_v2.0_LibriSpeech_train-clean-100_FT-May-20-2025_01+17PM-44460033/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/home2/jw/workspace/asr/mlcl_handover/tts/coqui-ai-TTS/recipes/mlcl/run/training/XTTS_v2.0_original_model_files/", use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["/home2/jw/workspace/asr/mlcl_handover/asr/dataset/LibriSpeech/LibriSpeech/train-clean-100/87/121553/87-121553-0097.flac"])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("./xtts_origin.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)
