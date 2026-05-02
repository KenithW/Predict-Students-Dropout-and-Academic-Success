# Student Dropout and Academic Success ML Coursework Project

This workspace contains a COMP4030-style machine learning project using the UCI Predict Students' Dropout and Academic Success dataset.

Research question:

> How well can machine learning models predict student dropout and academic success, and how much does early academic performance improve prediction compared with enrollment-only information?

Main files:

- `actg175_ml_coursework.ipynb`: single runnable and commented notebook for the student dropout project.
- `requirements.txt`: Python packages needed to run the notebook.
- `outputs/figures/`: generated plots for paper and slides/poster.

The official target is a three-class academic outcome:

- `Dropout`
- `Enrolled`
- `Graduate`

The project compares three prediction timings:

- `Enrollment only`: background, application, demographic, and socio-economic variables known around enrollment.
- `+ 1st semester`: enrollment variables plus first-semester academic performance.
- `+ 1st + 2nd semester`: all available features, including first- and second-semester academic performance.

The model comparison uses classroom methods including DummyClassifier, Logistic Regression, Decision Tree, KNN, SVM, Random Forest, and MLP. `GridSearchCV` is used on the training set with macro F1 as the tuning objective. Accuracy and weighted F1 are also reported, with special attention to per-class recall for `Dropout`.

Important modelling choice: semester performance variables are only used in the feature sets where the prediction timing occurs after those semester results are available. This avoids claiming that future academic performance is available at enrollment time.

Dataset citation:

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.
