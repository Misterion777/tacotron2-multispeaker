# Multispeaker architecture ideas
1. Just concatenate speaker embeddings with encoder output, like in [this repository](https://github.com/ide8/tacotron2).

2. Three locations to input speaker embeddings: concatenating with encoder output before decoder, inputting to prenet, inputting to postnet. [Multi-speaker with neural speaker embeddings, part 3](https://arxiv.org/pdf/1910.10838.pdf)

3. Forward speaker embeddings through fully connected layers with softsign and then concatenate. [Deep Voice 3](https://arxiv.org/pdf/1710.07654.pdf)