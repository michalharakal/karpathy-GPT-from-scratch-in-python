# Jupyter notebook with pipenv 

```bash
pipen --python 3.8 
pipenv install -d ipykernel
pipenv shell
```


This will bring up a terminal in your virtualenv like this:

```bash
(karpathy-GPT-from-scratch-in-python)  bash-4.4$
```

In that shell do:
```bash 
python -m ipykernel install --user --name=karpathy-GPT-from-scratch-in-python
```
