# Binerals classfication task

Some notes how to run:

```bash
virtualenv venv

source venv/bin/activate

pip install -r requirements.txt

python -m spacy download en

python assessment.py -i ./articles
```

Some notes:
- python will need 5-6 Gb of space on hard drive for pytorch, spacy with GloVE embeddings
- for easier demonstartion I added link to colab to view results directly, however it's ok to run code localy, but you will need beefy GPU and use cuda with `-d cuda` argument
- Model isn't always reaching 90% precision mark, and shows some symptoms of overfitting. Realisticly I think precision is around 80-85%.
- Code uses old versions of pytorch and torchtext to syncro versions with current google colab versions (03-11-2020)


[Link to colab](https://colab.research.google.com/drive/1FQ4UKYbZiZv-HaKhPm73HBm0XQkwhovP?usp=sharing)
