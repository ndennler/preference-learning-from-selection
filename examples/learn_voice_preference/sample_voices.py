#@title English multi-speaker pretrained model { run: "auto" }

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
import kaldiio
import wavio

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
        xvectors = {k: v for k, v in kaldiio.load_ark(xvector_ark)}
        spks = list(xvectors.keys())
        self.speaker_vector_array = np.array(list(xvectors.values()))

    def get_audio_sample_and_vector(self, sentence):
        '''
        returns a random speaker numpy array and its streaming frequency (in that order)
        '''
        # synthesis
        rand_speaker = np.random.randint(len(self.speaker_vector_array))
        spembs = self.speaker_vector_array[rand_speaker,:]
        speech = torch.randn(50000,) * 0.01

        with torch.no_grad():
            wav, c, *_ = self.text2speech(sentence, speech=speech, spembs=spembs)
            wav = self.vocoder.inference(c)

        return wav.view(-1).cpu().numpy(), fs, spembs

    def generate_good_utterances(self, omega, number):
        rewards = np.matmul(self.speaker_vector_array, omega)
        samples = np.argpartition(rewards, -number)[-number:]
        speakers = self.speaker_vector_array[samples]

        speech = torch.randn(50000,) * 0.01

        for i, spembs in enumerate(speakers):
            with torch.no_grad():
                wav, c, *_ = self.text2speech('this is a good voice! I hope you like it', speech=speech, spembs=spembs)
                wav = self.vocoder.inference(c)

            wav = wav.view(-1).cpu().numpy()
            wavio.write(f"data/good_choice_{i}.wav", wav, fs, sampwidth=2)