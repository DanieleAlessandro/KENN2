# KENN2 (Knowledge Enhanced Neural Networks - v2)
KENN (Knowledge Enhanced Neural Networks) is a library for python 3 built on top of TensorFlow 2 that lets you modify neural networks models by providing logical knowledge in the form of a set of universally quantified FOL clauses. It does so by adding a new final layer, called **Knowledge Enhancer (KE)**, to the existing neural network. The KE changes the orginal predictions of the standard neural network enforcing the satisfaction of the knowledge. Additionally, it contains **clause weights**, learnable parameters which represent the strength of each clause.

This is an implementation of the model presented in our paper:
[Knowledge Enhanced Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-29908-8_43).

If you use this software for academic research, please, cite our work using the following BibTeX:
```
@InProceedings{10.1007/978-3-030-29908-8_43,
author="Daniele, Alessandro
and Serafini, Luciano",
editor="Nayak, Abhaya C.
and Sharma, Alok",
title="Knowledge Enhanced Neural Networks",
booktitle="PRICAI 2019: Trends in Artificial Intelligence",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="542--554",
isbn="978-3-030-29908-8"
}
```

**NB:** version 1.0 of KENN was released for python 2.7 and TensorFlow 1.x and it is available at [KENN v1.0](https://github.com/DanieleAlessandro/KENN). Notice that this version is not backward compatible and it implements additional features for working in the presence of relational domains.


## License
Copyright (c) 2019, Daniele Alessandro, Serafini Luciano
All rights reserved.

Licensed under the BSD 3-Clause License.
