# ai-journey-2020
This repo contains solution to [AI Journey 2020](https://github.com/sberbank-ai/digital_peter_aij2020/blob/master/README.en.md) competition.

# Ideas
1. Transfer learning (ResNet-encoder, BERT-decoder)
2. Attention, different modifications of baseline
3. Attributes of images - is there a symbol (multilabel classification) => leveraging knowledge graph for decoding
4. Use DropConnect for RNN-part and NT-ASGD as optimizer for that as regularizations (AWD-LSTM for language modelling).

## Attributes for knowledge graph
* Easy to mix up letters «ѣ» и «е»
* "Ї" stand before vowels and "й" (including "е", "я", "Їo"). Petr had it in "иЇ" (right "Їи")
* Reform: «ӡ» -> «Ѕ» -> «ӡ», "I" ("Ї" before) for [и]

## Resources

[Sequence modeling with CTC | Distill](https://distill.pub/2017/ctc/)

[Handwriting recognition and language modeling with MXNet Gluon | Medium](https://medium.com/apache-mxnet/handwriting-ocr-handwriting-recognition-and-language-modeling-with-mxnet-gluon-4c7165788c67)

[Deslanting algorithm | Github](https://github.com/githubharald/DeslantImg)

[CTC Decoder | Github](https://github.com/parlance/ctcdecode)

[CTC Word Beam Search Decoding Algorithm | Github](https://github.com/githubharald/CTCWordBeamSearch)

[Handwritten Text Recognition with TensorFlow | Github](https://github.com/githubharald/SimpleHTR)

[Handwritten Text Recognition in Historical Documents](https://repositum.tuwien.at/retrieve/10807)

[Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.at/retrieve/1835)

[Trainable Spectrally Initializable Matrix Transformations in Convolutional Neural Networks | Arxiv 2019](https://arxiv.org/abs/1911.05045)
