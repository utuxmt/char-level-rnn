# English Name Generator
---
A PyTorch implementation of character level language model.

The program will learn the different name patterns, and randomly generate new names.

The code is based on [char-rnn.pytorch repo](https://github.com/spro/char-rnn.pytorch). 
The mainly difference is that the input shape here is batch first, and input sequence padding is supported.

## Dataset
[US Baby Names](https://www.kaggle.com/kaggle/us-baby-names/data) from [Kaggle](https://www.kaggle.com/) which looks like:
```csv
Id,Name,Year,Gender,Count
1,Mary,1880,F,7065
2,Anna,1880,F,2604
3,Emma,1880,F,2003
4,Elizabeth,1880,F,1939
5,Minnie,1880,F,1746
...
```

## Demo
The current model is trained with my laptop with 3 epochs of all boys' name.
```python
$ python generate.py --starts_with ma --predict_len 3
>>> Names Generated:
>>>  maari
>>>  maaja
>>>  maeyk
>>>  maaty
>>>  maeko
```


