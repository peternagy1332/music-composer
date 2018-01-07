# Algorithmic music generation using recurrent neural networks

The model is capable to learn the style of a given artist and continue a previously unseen music in that style.

Since it learns based on WAV files with high dynamic range, the output is quite noisy as well. Nonetheless, if the music were written for e.g. piano, better results could be accomplished.

A more detailed README.md is coming soon...

## The architecture
I used two time-distributed dense layers as input and output and LSTMs between them.

<img src="https://github.com/peternagy1332/music-composer/blob/master/assets/arch.png?raw=true" width="50%"/>

## Acknowledgement
Thanks for Matt Vitelli and Aran Nayebi for their remarkable work (https://github.com/MattVitelli/GRUV) that was the kickstarter of this project.
