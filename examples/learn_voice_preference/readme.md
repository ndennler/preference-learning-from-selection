This example implements a voice search through neural representations of speakers.

The following libraries should be installed:

```
python3 -m pip install espnet==0.10.2 parallel_wavegan==0.5.3 pyopenjtalk
python3 -m pip install --pre espnet_model_zoo
```

Then run the program "voice_preferences.py" in this folder. The first time it
runs, it will take a VERY long time to download the model teehee.

Once the model does download, the program will ask for the input. Check the data
folder for sample1.wav and sample2.wav. select the one you like better with the
corresponding number. If you don't really have a preference, press any other
number.

At the end of 10 trials, the program will exit and leave you with what it thinks
are the 10 best options based on what you chose.