# ai-journey-2020
This repo contains solution to [AI Journey 2020](https://github.com/sberbank-ai/digital_peter_aij2020/blob/master/README.en.md) competition.

# Ideas
1. Transfer learning (ResNet-encoder, BERT-decoder)
2. Attention, different modifications of baseline
3. Attributes of images - is there a symbol (multilabel classification) => leveraging knowledge graph for decoding

## Attributes for knowledge graph
* Easy to mix up letters «ѣ» и «е»
* "Ї" stand before vowels and "й" (including "е", "я", "Їo"). Petr had it in "иЇ" (right "Їи")
* Reform: «ӡ» -> «Ѕ» -> «ӡ», "I" ("Ї" before) for [и]

## Resources

https://distill.pub/2017/ctc/

https://medium.com/apache-mxnet/handwriting-ocr-handwriting-recognition-and-language-modeling-with-mxnet-gluon-4c7165788c67

https://github.com/githubharald/DeslantImg

https://github.com/parlance/ctcdecode

https://github.com/githubharald/CTCWordBeamSearch

https://github.com/githubharald/SimpleHTR

https://repositum.tuwien.at/retrieve/10807

https://arxiv.org/abs/1911.05045
