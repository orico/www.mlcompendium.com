# Mix N Match

Single-modality models get more useful when they are chained, so that one model's output becomes another's input. This page shows one system that combines detection, segmentation, and generation across image, text, and speech, and then one that applies generative models to formal theorem proving.

The same notes are in [DETECTION](../deep-learning/deep-neural-machine-vision.md#detection), [Segmentation](../deep-learning/deep-neural-machine-vision.md#segmentation), and [Speech](speech.md).

[Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) is IDEA-Research's Grounded SAM: it marries Grounding DINO with Segment Anything, Stable Diffusion, and Recognize Anything to automatically detect, segment, and generate anything. The repository extends that chain to more inputs:

 > Grounding DINO with Segment Anything & Stable Diffusion & BLIP & Whisper - Automatically Detect , Segment and Generate Anything with Image, Text, and Speech Inputs

The same mix-and-match idea reaches beyond media. [LeadDojo theorem proving](https://leandojo.org/) is AI-driven formal theorem proving in the Lean ecosystem.
