#/usr/bin/python3

'''
This file iterates through all of the speaker embeddings and develops a
descriptor for each vector. The resulting database is stored in a file in this
folder entitled voice_info.csv.
'''


lang = 'English'
fs = 24000 #@param {type:"integer"}
tag = 'kan-bayashi/libritts_gst+xvector_conformer_fastspeech2' #@param ["kan-bayashi/vctk_gst_tacotron2", "kan-bayashi/vctk_gst_transformer", "kan-bayashi/vctk_xvector_tacotron2", "kan-bayashi/vctk_xvector_transformer", "kan-bayashi/vctk_xvector_conformer_fastspeech2", "kan-bayashi/vctk_gst+xvector_tacotron2", "kan-bayashi/vctk_gst+xvector_transformer", "kan-bayashi/vctk_gst+xvector_conformer_fastspeech2", "kan-bayashi/libritts_xvector_transformer", "kan-bayashi/libritts_xvector_conformer_fastspeech2", "kan-bayashi/libritts_gst+xvector_transformer", "kan-bayashi/libritts_gst+xvector_conformer_fastspeech2"] {type:"string"}
vocoder_tag = "libritts_parallel_wavegan.v1.long" #@param ["vctk_parallel_wavegan.v1.long", "vctk_multi_band_melgan.v2", "libritts_parallel_wavegan.v1.long", "libritts_multi_band_melgan.v2"] {type:"string"}

import time
import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import download_pretrained_model
from parallel_wavegan.utils import load_model
import os
import numpy as np
import pandas as pd
import kaldiio
import librosa
from tqdm import tqdm

class VoiceSampler():
    def __init__(self):
        d = ModelDownloader()
        
        self.text2speech = Text2Speech(
            **d.download_and_unpack(tag),
            device="cuda",
            # Only for Tacotron 2
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3,
            # Only for FastSpeech & FastSpeech2
            speed_control_alpha=1.0,
        )
        self.text2speech.spc2wav = None  # Disable griffin-lim
        self.vocoder = load_model(download_pretrained_model(vocoder_tag)).to("cuda").eval()
        self.vocoder.remove_weight_norm()

        # load x-vector
        model_dir = os.path.dirname(d.download_and_unpack(tag)["train_config"])
        xvector_ark = f"{model_dir}/../../dump/xvector/tr_no_dev/spk_xvector.ark"  # training speakers
        # xvector_ark = f"{model_dir}/../../dump/xvector/dev/spk_xvector.ark"  # development speakers
        # xvector_ark = f"{model_dir}/../../dump/xvector/eval1/spk_xvector.ark"  # eval speakers
        self.xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
        self.speaker_vector_array = np.array(list(self.xvectors.values()))
        self.speaker_names = np.array(list(self.xvectors.keys()))

    def gen_audio(self, sentence, speaker_name):
        spembs = self.xvectors[speaker_name]
        with torch.no_grad():
            wav, c, *_ = self.text2speech(sentence, 
                                        speech=torch.randn(50000,), 
                                        spembs=spembs)
            wav = self.vocoder.inference(c)

        return wav.view(-1).cpu().numpy()

    def get_speaker_features(self, speaker_id):

        utterances = [
            "My mom drove me to school fifteen minutes late on Tuesday.",
            "The girl wore her hair in two braids, tied with two blue bows.",
            "The mouse was so hungry he ran across the kitchen floor without even looking for humans.",
            "The tape got stuck on my lips so I couldn't talk anymore.",
            "The door slammed down on my hand and I screamed like a little baby.",
            "My shoes are blue with yellow stripes and green stars on the front.",
            "The mailbox was bent and broken and looked like someone had knocked it over on purpose.",
            "I was so thirsty I couldn't wait to get a drink of water.",
            "I found a gold coin on the playground after school today."
        ]

        speech = torch.randn(50000,)
        features = []
        for sentence in utterances:
            with torch.no_grad():
                wav, c, *_ = self.text2speech(sentence, 
                                            speech=speech, 
                                            spembs=speaker_id)
                wav = self.vocoder.inference(c)

            wav = wav.view(-1).cpu().numpy()

            f0, voiced_flag, voiced_probs = librosa.pyin(wav, fmin=30, fmax=380)
            
            #begin sentence-wise features calculations
            avg_f0 = np.nanmean(f0[voiced_flag])
            std_f0 = np.nanstd(f0[voiced_flag])
            speed = len(wav) / fs

            features.append((speed, avg_f0, std_f0))
        
        return np.mean(np.array(features), axis=0)

if __name__ == '__main__':
    sampler = VoiceSampler()
    data = []
    with tqdm(total=len(sampler.speaker_names)) as pbar:
        for voice_name, voice_id in zip(sampler.speaker_names, 
                                        sampler.speaker_vector_array):
            speed, average_f0, std_f0 = sampler.get_speaker_features(voice_id)
            data.append({
                'name': voice_name,
                'speed': speed,
                'avg_f0': average_f0,
                'std_f0': std_f0,
            })
            pbar.update(1)

    voice_dataframe = pd.DataFrame(data)
    voice_dataframe.to_csv('./voice_info.csv')

