OpenNMT is an open source ecosystem for neural machine translation and neural sequence learning.
OpenNMT provides implementations in 2 popular deep learning frameworks:
Each implementation has its own set of unique features but shares similar goals:

Highly configurable model architectures and training procedures
Efficient model serving capabilities for use in real world applications
Extensions to allow other tasks such as text generation, tagging, summarization, image to text, and speech to text


In the following Program I used LSTM Model:
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can process not only single data points (such as images), but also entire sequences of data (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected handwriting recognition, speech recognition and anomaly detection in network traffic or IDSs (intrusion detection systems).

Used BLEU score for evaluating the model:

The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric for evaluating a generated sentence to a reference sentence.

A perfect match results in a score of 1.0, whereas a perfect mismatch results in a score of 0.0
