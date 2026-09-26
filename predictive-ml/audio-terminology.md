# Terminology

The algorithm pages name their tasks as if the reader already knows them, so the terms come first. The page defines audio source separation, then sound event detection, then query-based separation, which narrows separation down to one sound you ask for.

The same notes are in [Audio Source Separation](audio-algorithms.md#audio-source-separation) and [Sound Event Detection](audio-algorithms.md#sound-event-detection).

Audio source separation is the task of pulling specific sound sources out of an audio mixture. Source separation techniques in audio processing can be classified into various methods, such as blind source separation, supervised source separation, and semi-supervised source separation, and these techniques use algorithms and signal processing methods to extract specific sound sources from an audio mixture. Note that usually these methods are for multi-channel audio; if we have a single-channel audio recording, this is more challenging.

Separation asks what the sources are; sound event detection asks when they happen. Sound Event Detection (SED) is the task of recognizing the sound events and their respective temporal start and end time in a recording. Sound events in real life do not always occur in isolation, but tend to considerably overlap with each other, and recognizing such overlapping sound events is referred to as polyphonic SED.

Query-based separation combines the two ideas by targeting one source. Query-based separation in audio could involve using specific sounds or queries to identify and extract particular sounds from an audio recording. For example, in a crowded audio environment, you might use query-based separation to extract the voice of a specific speaker from a mixture of voices. By providing a query related to the speaker's voice characteristics or specific phrases they are saying, algorithms can be designed to identify and separate that particular speaker's voice from the overall audio recording.
