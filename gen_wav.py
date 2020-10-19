import numpy as np
import torch
from pathlib import Path
import argparse

from hparams import create_hparams
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write


def infer(text, checkpoint_path, out_filename, speaker_id = None, griffin_iters=60):
    hparams = create_hparams()

    if speaker_id is not None:
        assert speaker_id in range(hparams.n_speakers) 
        speaker_id = torch.LongTensor([speaker_id]).cuda()

    hparams.sampling_rate = 22050

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    model_input = sequence if speaker_id is None else (sequence,speaker_id)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(model_input)

    taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

    mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio_path = Path('samples')
    audio_path.mkdir(exist_ok=True)
    audio_path = audio_path.joinpath(f'{speaker_id[0]}_{out_filename}')
    write(audio_path, hparams.sampling_rate, audio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates wav from text with Tacotron2 and Griffin-lim.')
    parser.add_argument('text', type=str, 
                        help='Text to generate wav from.')
    parser.add_argument('speaker_id', type=int, default=None,
                        help='Generated speech will be voiced by speaker with this id.')
    parser.add_argument('checkpoint_path', type=str, 
                        help='Checkpoint to trained Tacotron2.')

    parser.add_argument('-out_filename', type=str, default='sample.wav', metavar='FILE_PATH',
                        help='Path to output wav file.')
    
    args = vars(parser.parse_args())

    infer(args['text'], args['speaker_id'], args['checkpoint_path'],args['out_filename'])

#infer("outdir/checkpoint_3000", 60, "What a terrible tongue twister, what a terrible tongue twister, what a terrible tongue twister.", "sample.wav")
# infer("outdir/checkpoint_200", 60, "Geralt of Rivia was a legendary witcher of the School of the Wolf active throughout the 13th century. He loved the sorceress Yennefer, considered the love of his life despite their tumultuous relationship, and became Ciri's adoptive father.", "sample.wav")
#infer("outdir/checkpoint_1000", 60, "The humour is extremely subtle, and without a solid grasp of theoretical physics most of the jokes will go over a typical viewer's head.", "sample.wav")
