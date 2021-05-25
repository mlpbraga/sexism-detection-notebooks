# Sexism Detection

This repository has the implementation of a sexism detection research [1], using a dataset built using the https://sexismo.vercel.app site.

The dataset can be found at [`/data/labeled-comments.csv`](https://github.com/mlpbraga/sexism-detection-notebooks/blob/main/data/labeled-comments.csv).

The file `sexism-detection.ipynb` contains all the feature selection, training and testing implementation. 

The file `baseline.ipynb` has the implementation of a hate speech detection work made by Davidson et al. [2] and the file `bert.ipynb` implements the [BERT model](https://github.com/google-research/bert), to compare results with the current work.

# Referencies

1 - Braga, M. L. P., Nakamura, F. G., and Nakamura, E. F. (2020).  Criação e caracterização de um corpus de discurso sexista em português. In Anais do IX Brazilian Workshop on Social Network Analysis and Mining, pages 97–107. SBC.
2 - Davidson, T., Warmsley, D., and Macy, M. (2017). Automated hate speech detection andthe problem of offensive language. Eleventh International AAAI Conference on Weband Social Media.
