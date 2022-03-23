# 《ELECTRA is a Zero-Shot Learner, Too》

## Overview

Recently, for few-shot or even zero-shot learning, the new paradigm “*pre-train, prompt, and predict*” has achieved remarkable achievements compared with the “*pre-train, fine-tune*” paradigm. A series of small language models (e.g., BERT, ALBERT, RoBERTa) based on **Masked Language Model (MLM)** pre-training tasks became popular and widely used. However, another efficient and powerful pre-trained language model, ELECTRA, has probably been neglected. This paper attempts to accomplish several NLP tasks in the zero-shot scenario using a sample-efficient ELECTRA original pre-training task—**Replaced Token Detection (RTD)**. Through extensive experiments on 15 various NLP datasets, we find that ELECTRA performs surprisingly well as a zero-shot learner, which proves the ELECTRA model has more potential to be stimulated.
<img width="886" alt="image" src="https://user-images.githubusercontent.com/56249874/159676905-b0b62840-d2e2-449a-a095-b24485cf5f91.png">

## Acknowledgements
Our code is based on [Jianlin Su](https://github.com/bojone)'s [bert4keras](https://github.com/bojone/bert4keras) and [Sun Yi](https://github.com/sunyilgdx)'s [NSP-BERT](https://github.com/sunyilgdx/NSP-BERT/).
