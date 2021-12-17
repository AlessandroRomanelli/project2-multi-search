# Instructions

To extract data from `tensorflow` folder just run:
```
>   py extract-data.py tensorflow
```

The output will be stored in a file named `results.csv`

This will be the input for the next step, which runs the query `"Optimizer that implements the Adadelta algorithm"`.

```
>   py search-data.py
```

The results for the different kind of searches will be displayed in the terminal.

Finally, the last step computes precision and recall for each method, along with scatter plots for the `LSI` and `doc2vec` methods.

This utilizes the file `ground-truth-unique.txt` and the original `results.csv` to run the queries and evaluate the different methods.

```
>   py prec-recall.py
```

Precision and recall are listed in the terminal.