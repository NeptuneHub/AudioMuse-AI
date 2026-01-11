# Student CLAP: Lightweight Audio Encoder

## Quick Start

```bash
# Setup and install
python3 -m venv venv && source venv/bin/actiqvate && pip install -r requirements.txt

# Run training
python3 train_real.py --config config.yaml
```

## License
This is a standalone project that try to do a distillation process of LAION CLAP:
- https://github.com/LAION-AI/CLAP

by using the pretrained model: 
- music_audioset_epoch_15_esc_90.14.pt

As dataset it use a subset of the MTG jamendo dataset:
- https://github.com/MTG/mtg-jamendo-dataset 

that comes from:
- https://www.jamendo.com/

And a dataset from:
- https://freemusicarchive.org/

All the songs are under Creative Commons license, more details can be found here:
- [student_clap/models/FMA_SONGS_2247_LICENSE.md](student_clap/models/FMA_SONGS_2247_LICENSE.md)
- [student_clap/models/JAMENDO_SONGS_LICENSE.md](student_clap/models/JAMENDO_SONGS_LICENSE.md)