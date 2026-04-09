# Background: Wake Word Detection for Beginners

This document explains the fundamentals of wake word detection for those new to machine learning. We focus on intuition and practical concepts rather than mathematical details.

## Setup checkpoint

Before following this guide, run the repo bootstrap once:

```bash
uv sync --group dev
source .venv/bin/activate
uv run pre-commit install
uv run wakeword-workbench --help
uv run wakeword-workbench validate examples/basic_config.yaml
```

If validation fails due to missing TTS backend, install `uv sync --extra kokoro` or `uv sync --extra piper`.

---

## 1. What is a Wake Word?

A **wake word** is a specific phrase that activates a voice assistant. You've probably used one: "Hey Siri" on Apple devices, "Alexa" on Amazon Echo, or "OK Google" on Android phones. These phrases tell the assistant to start listening for your actual request.

Wake words solve a privacy problem. Without them, voice assistants would need to record and send everything you say to the cloud for processing. That's impractical and invasive. Instead, the device runs a small, local program that listens only for the wake word. When it hears it, the device wakes up and starts recording your actual command.

This approach has three key benefits:

1. **Privacy**: The device only records after you say the wake word, not before
2. **Always-on**: The detector runs continuously in the background, waiting for your command
3. **Low power**: The wake word detector uses minimal battery because it's designed to be tiny and efficient

Production wake word systems run under extreme constraints. They need to respond in under 50 milliseconds, use less than 1 megabyte of memory, and consume less than 1 milliwatt of power. That's roughly the energy of a small LED indicator light. These constraints exist because the detector runs 24/7 on battery-powered devices like smart speakers and phones.

**How it works at a high level**: The device continuously processes audio through a small machine learning model. This model outputs a probability: "Is this the wake word?" If the probability exceeds a threshold, the device activates and starts recording for the main assistant.

Sources:
- Arun Baby, "Wake Word Detection" (2026): https://www.arunbaby.com/speech-tech/0040-wake-word-detection/
- Arun Baby, "Keyword Spotting" (2026): https://www.arunbaby.com/speech-tech/0009-keyword-spotting/

---

## 2. The Machine Learning Problem

Wake word detection is a **binary classification** problem. The model must decide: "Is this audio the wake word, or not?"

Think of it like a bouncer at a club. The bouncer checks each person's ID and decides: "Is this person on the guest list, or not?" The wake word detector checks each audio snippet and decides: "Is this the wake word, or not?"

But here's the challenge: the "not" category includes everything else. Other words, background noise, music, silence, coughing, sneezing. The model must correctly identify the wake word among millions of possible "not wake word" sounds.

**The similar-sounding problem**: Some phrases sound almost like the wake word. "Hey Siri" and "Hey Seriously" share similar sounds. "Alexa" and "Alex" are close. These near-misses are where models fail most often.

**The trade-offs**: You can't optimize for everything at once. There's a tension between three goals:

1. **Accuracy**: Correctly identifying the wake word
2. **Few false triggers**: Not activating when you didn't say the wake word
3. **Few missed detections**: Not missing when you did say the wake word

Improving one often hurts the others. A model that catches every "Hey Siri" might also trigger on "Hey Seriously." A model that never false-triggers might miss half your commands. Finding the right balance is the core challenge.

---

## 3. Training Data Basics

Machine learning models learn from examples. For wake word detection, you need two types of training data:

**Positive examples**: Recordings of people saying the wake word. If your wake word is "Hey Daisy," you need hundreds or thousands of recordings of different people saying "Hey Daisy" in different environments: quiet rooms, noisy streets, echoey bathrooms, cars, offices.

**Negative examples**: Recordings of everything that is NOT the wake word. This includes other words, background noise, music, silence, and crucially, phrases that sound similar to the wake word.

**Why variety matters**: A model trained only on clear recordings in quiet rooms will fail in the real world. Real users speak with different accents, at different speeds, in different environments. Your training data must cover this variety.

Think of it like studying for a test. If you only study the exact questions that will be on the test, you'll fail when the questions change slightly. But if you study many variations and examples, you'll recognize the underlying pattern. Wake word models need the same breadth of examples.

Sources:
- Arun Baby, "Wake Word Detection" (2026): https://www.arunbaby.com/speech-tech/0040-wake-word-detection/

---

## 4. Understanding Hard Negatives

**Hard negatives** are negative examples that are difficult for the model to distinguish from the positive class. They're "hard" because they're confusing, not because they're computationally expensive.

Imagine you're training a model to recognize apples. Easy negatives are things like cars, houses, and people. The model will quickly learn those aren't apples. Hard negatives are things like pears, red balls, or apple-shaped toys. These look similar to apples, so the model struggles to tell them apart.

**Why random audio isn't enough**: You might think, "I'll just use random audio clips as negatives." But random audio is mostly easy negatives. A model trained on random negatives will learn to distinguish "Hey Daisy" from silence, from music, from "The weather today is sunny." But it won't learn to distinguish "Hey Daisy" from "Hey Davy" or "Hey Maisy."

**A concrete example**: Suppose you're building a "Hey Daisy" wake word model. You train with positive examples of "Hey Daisy" and random negative examples (silence, music, other phrases). Your model achieves 99% accuracy on your test set. Great!

But then you deploy it, and users complain: "My speaker keeps activating when I say 'Hey Davy' or 'Hey Maisy'." Your model never saw these similar-sounding phrases during training, so it can't distinguish them. These are hard negatives.

**How hard negatives improve robustness**: By including hard negatives in your training data, you teach the model to distinguish the wake word from confusing alternatives. The model learns: "Hey Daisy" has a specific pattern. "Hey Davy" is similar but different. "Hey Maisy" is also similar but different. The model becomes more precise.

Research by Hou et al. (2020) showed that mining hard negatives significantly reduces false positives. Instead of random negatives, they systematically found examples that confused their model and added those to training. The result: a model that triggers less often on similar-sounding phrases.

Sources:
- Hou et al., "Mining Effective Negative Training Samples for Keyword Spotting" (IEEE ICASSP 2020): https://ieeexplore.ieee.org/document/9053009
- FutureBeeAI, "False Positives in Wake Word Detection" (2025): https://www.futurebeeai.com/knowledge-hub/false-positives-wake-word

---

## 5. Data Augmentation Explained

**Data augmentation** is the practice of artificially expanding your training data by applying transformations to existing examples. It's like taking a photo and creating variations: cropping it, adjusting brightness, adding noise, rotating it slightly. Each variation is a new training example.

**Why augmentation matters**: You can't record every possible variation of your wake word. You can't record in every possible room, with every possible background noise, at every possible volume level. Augmentation bridges this gap. It takes your limited recordings and creates variations that simulate real-world conditions.

**Key augmentation techniques for audio**:

1. **Noise injection**: Add background noise (traffic, crowds, wind) to clean recordings. This simulates real-world environments where background noise is common.

2. **Reverb**: Add reverberation to simulate different acoustic environments. A recording made in a small carpeted room sounds different from one made in a large bathroom or a car. Reverb augmentation teaches the model to recognize the wake word in these different spaces.

3. **Time stretching**: Speed up or slow down the audio slightly. This simulates different speaking rates. Some people speak quickly, others slowly. The model should recognize the wake word regardless.

4. **Pitch shifting**: Raise or lower the pitch slightly. This simulates different voice types. Children have higher-pitched voices than adults. Men typically have lower voices than women. Pitch augmentation helps the model generalize across these variations.

5. **Gain/volume changes**: Adjust the volume of recordings. Users speak at different volumes, and microphones have different sensitivities. The model should work whether you whisper or shout.

6. **Clipping/distortion**: Simulate poor microphone quality or transmission artifacts. Real-world audio isn't always pristine.

**How augmentation multiplies your dataset**: If you have 1,000 recordings of "Hey Daisy," you can apply 10 different augmentations to each one, creating 10,000 training examples from your original 1,000. This dramatically improves model robustness without requiring more human recordings.

Think of augmentation as training for different conditions. An athlete who only trains in perfect weather will struggle in rain or heat. An athlete who trains in varied conditions is prepared for anything. Augmentation gives your model that varied training.

Sources:
- Arun Baby, "Audio Augmentation Techniques" (2026): https://www.arunbaby.com/speech-tech/0018-audio-augmentation-techniques/

---

## 6. TTS for Wake Words

**Text-to-speech (TTS)** is technology that converts written text into spoken audio. You type "Hey Daisy," and the TTS system generates an audio file of someone saying "Hey Daisy."

**TTS as a data source for wake word training**: Instead of recording thousands of people saying your wake word, you can use TTS to generate synthetic speech. This offers several advantages:

**Pros of TTS-generated data**:

1. **Unlimited variety**: TTS can generate your wake word in thousands of different voices, accents, and speaking styles. You're not limited by who you can recruit for recordings.

2. **Fast iteration**: Need to test a new wake word? Generate TTS audio in minutes. No need to schedule recording sessions, recruit speakers, or process audio files.

3. **Privacy**: TTS doesn't require human recordings. This avoids privacy concerns and consent issues around storing voice data.

4. **Consistent quality**: TTS systems produce consistent, clean audio. No background noise, no microphone variations, no recording artifacts.

**Cons of TTS-generated data**:

1. **Domain gap**: TTS audio sounds different from real human speech. There's a subtle quality gap. Models trained only on TTS may not perform as well on real user audio.

2. **Synthetic artifacts**: TTS systems can introduce subtle artifacts that don't exist in natural speech. Models might learn to recognize these artifacts instead of the actual wake word.

3. **Limited emotional variation**: TTS systems typically produce neutral speech. Real users speak with emotion: excitement, frustration, tiredness. TTS may not capture this variation.

**Best practices**: Research by Google (2024) recommends mixing TTS-generated data with real human recordings. Use TTS for variety and volume, but include real recordings to ground the model in natural speech. This hybrid approach captures the benefits of both: the scale of synthetic data and the authenticity of real recordings.

Think of TTS like stock photos versus personal photos. Stock photos (TTS) give you variety and convenience, but personal photos (real recordings) capture authentic moments. The best photo albums include both.

Sources:
- Google, "Utilizing TTS Synthesized Data for Efficient Development of Keyword Spotting Model" (2024): https://arxiv.org/pdf/2407.18879

---

## 7. Evaluation Metrics

To build a good wake word model, you need to measure its performance. Three key metrics are used:

**FAR (False Acceptance Rate)**: The rate at which the model incorrectly activates when the wake word was NOT spoken. This is measured in false alarms per hour. For example, a FAR of 0.5 means the model falsely activates once every two hours on average.

FAR is critical for user experience. A model that false-triggers frequently is annoying. Imagine your smart speaker activating every few minutes while you're having a conversation. You'd quickly disable it. Low FAR is essential for user trust.

**FRR (False Rejection Rate)**: The rate at which the model fails to activate when the wake word WAS spoken. This is the miss rate. A FRR of 5% means the model misses 5% of your commands.

FRR affects usability. A model that frequently misses your commands is frustrating. You find yourself repeating "Hey Siri, Hey Siri, Hey Siri" until it finally responds. Low FRR is essential for user satisfaction.

**EER (Equal Error Rate)**: The point where FAR equals FRR. This is a single metric that balances both types of errors. Lower EER is better.

**The trade-off**: FAR and FRR are inversely related. You can lower one by raising the other. Here's why:

Your model outputs a probability: "How likely is this the wake word?" You set a threshold. If the probability exceeds the threshold, you activate. If you set a high threshold, you'll have fewer false activations (low FAR) but more missed commands (high FRR). If you set a low threshold, you'll catch more commands (low FRR) but have more false activations (high FAR).

Think of it like a spam filter. A strict filter catches all spam but might flag legitimate emails. A lenient filter lets all legitimate emails through but might miss some spam. You can't have both perfect filtering and perfect delivery. You choose a balance.

**Why FAR is typically more critical than FRR in wake word detection**: Users tolerate repeating a command more than they tolerate random activations. If your speaker misses one in twenty commands, you might repeat it. But if it activates randomly during conversations, you'll turn it off. This is why production systems often prioritize low FAR, even at the cost of higher FRR.

Sources:
- FutureBeeAI, "False Acceptance Rate in Wake Word Detection" (2025): https://www.futurebeeai.com/knowledge-hub/false-acceptance-rate-wake-word
- Recogtech, "FAR and FRR: Security Level versus Ease of Use" (2023): https://recogtech.com/en/insights-en/far-and-frr-security-level-versus-ease-of-use/

---

## 8. The Iterative Process

Building a wake word model is not a one-time task. It's an iterative cycle:

1. **Generate initial dataset**: Start with positive examples (wake word recordings) and negative examples (other audio). Include variety: different voices, accents, environments.

2. **Train initial model**: Use your dataset to train a machine learning model. This gives you a baseline.

3. **Evaluate**: Measure your model's FAR and FRR on a test set. Identify where it fails.

4. **Mine hard negatives**: Find examples that confuse your model. These are phrases that sound similar to your wake word but aren't. Add these to your negative examples. The built-in `negatives` config can also bias the generated dataset toward phonetic confusions or synthetic phrase negatives before mining.

5. **Retrain**: Train a new model with your expanded dataset (original positives, original negatives, and hard negatives).

6. **Repeat**: Continue the cycle of evaluation, hard negative mining, and retraining until your metrics meet your targets.

**Why iteration matters**: You can't predict all the ways your model will fail. You discover these failures through evaluation. Each round of hard negative mining addresses specific weaknesses. The model improves incrementally.

Think of it like debugging code. You write code, test it, find bugs, fix them, test again. Each cycle makes the code more robust. Wake word models improve the same way: train, evaluate, find weaknesses, address them, repeat.

---

## 9. Key Principles

After understanding the fundamentals, these principles guide effective wake word development:

**Data quality > model architecture**: A simple model trained on excellent data outperforms a complex model trained on poor data. Invest in your dataset: diverse recordings, comprehensive negatives, thorough augmentation. The model architecture matters less than the data it learns from.

**Negative coverage is critical**: Many developers focus on positive examples (recordings of the wake word). But negative examples are equally important. Your model must learn what the wake word is NOT. Hard negatives, in particular, prevent false triggers on similar-sounding phrases.

**Real-world audio > synthetic only**: TTS-generated data is useful for scale and variety, but real human recordings ground your model in natural speech. The best datasets combine both: TTS for volume and variety, real recordings for authenticity.

**Iteration beats perfection**: You won't get it right the first time. Accept this. Build a baseline, evaluate, find weaknesses, address them, repeat. Each iteration improves your model. The goal is continuous improvement, not immediate perfection.

**User experience drives metrics**: FAR and FRR are technical metrics, but they represent user experience. A low FAR means fewer annoying false activations. A low FRR means fewer frustrating missed commands. Always connect metrics back to the user. What will users experience?

**Context matters**: A wake word model that works in a quiet home might fail in a noisy car. Consider where your model will be deployed. Test in those environments. Augment for those conditions.

---

## Summary

Wake word detection is a specialized machine learning problem with unique constraints: always-on, low-power, real-time response. Success depends more on data quality than model sophistication. Key practices include:

- Training with diverse positive and negative examples
- Mining hard negatives to reduce false triggers
- Augmenting data to simulate real-world conditions
- Mixing TTS-generated and real human recordings
- Measuring FAR and FRR to balance user experience
- Iterating continuously to improve robustness

The goal is a model that activates when you say the wake word, doesn't activate when you don't, and works reliably in the environments where users actually use it.
