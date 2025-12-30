# Speech Separation for Hearing-Impaired Children in the Classroom
###DATA WILL BE MADE AVAILABLE AFTER THE PAPER HAS BEEN PUBLISHED###

This repo is for our paper titled Speech Separation for Hearing-Impaired Children in the Classroom

Abstract:

Classroom environments pose significant challenges for hearing-impaired children, where background noise, simultaneous talkers, and reverberation degrade speech perception. Most deep learning-based speech separation algorithms are developed for adult voices in simplified conditions, neglecting both the higher spectral similarity of children's voices and the acoustic complexity of classrooms.
We address this gap using MIMO-TasNet, a compact, low-latency, multi-channel architecture suited for real-time processing in bilateral hearing aids or cochlear implants. We simulated naturalistic classroom conditions with moving talkers, both child–child and child–adult pairs across varying noise and distance settings. We compared three training strategies: adult speech only, classroom-specific data, and fine-tuning adult-trained models with limited classroom data.
Results show that binaural spatial cues enable adult-trained models to perform well in clean classroom conditions, even on overlapping child talkers. Classroom-specific training substantially improved separation quality, while fine-tuning with only half the data achieved superior performance, confirming data-efficient adaptation benefits. Training with diffuse babble noise enhanced robustness across conditions, and models preserved spatial awareness while generalizing to unseen talker–listener distances. Combining spatially aware architectures with targeted adaptation strategies can significantly improve speech accessibility for children in noisy classrooms, enabling practical on-device assistive technologies.
