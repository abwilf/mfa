# Montreal-Forced-Aligner Utils


## Setup
Install MFA

https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

```bash
pip install librosa soundfile pandas numpy requests tqdm
```

Make sure you're using `python3`.  I use `python3.7`.

## Minimal Example

```bash
python mfa_utils.py
```

Below, I walk through what this code does.  It is also well documented in `mfa_utils.py`.
To transform .vtt files (partial transcripts) and long wavs to a forced alignment (we have timestamp boundaries for the start and end of each word), we do the following steps:

1. Split the wavs into partials.  The original wavs (`siq_wavs`) and original vtt files (`siq_vtts`) are used to create the segments and corresponding segment transcripts.  These segmented wavs and transcripts go in `corpus_dir`, e.g. `siq`. This happens in `create_corpus()` in `mfa_utils.py`.
Vtt files look like this:
```
    WEBVTT
    Kind: captions
    Language: en

    0:00:00.000 --> 0:00:01.974
    - [Voiceover] Oh, why would
    he want to wreck something?

    0:00:01.974 --> 0:00:03.494
    That wouldn't be very nice.
    ...
```

In this example, we would have two segments from the original.  One corresponding to 0-1.974, the other corresponding to 1.974-3.494.

2. We align using `mfa`.  As long as your corpus is in the same format as `corpus_dir`, this code will be able to process it.  The key lines are below. The first few lines just make sure the executable `mfa` is available for your command line; the magic happens in the last line.
```bash
cd {this_root}
export KALDI_ROOT='{kaldi_root}'
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH

mfa align {corpus_dir} english_dict.txt english {aligned_dir} --clean
```

3. Aligned outputs are in `aligned_dir`, e.g. `aligned_siq`. We then process and recombine the outputs into a single dictionary we save to `intervals.json`, of the form below.  This happens in `mfa_utils`::`align_corpus()`.

```json
{
    "video_id_1": {
        "wav_idx": 0, // in case you need to know where to find its outputs in siq and aligned_siq for debugging
        "intervals": [
            [
                [
                    0.333, // word start
                    0.393 // word end
                ],
                "so" // word
            ],
            ...
        ]
    },
    "video_id_2": ...
```

