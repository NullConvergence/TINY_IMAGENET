# TINY_IMAGENET


This project is very similar to the [CIFAR](https://github.com/NullConvergence/CIFAR) project.


### Run:
```
$ python -m clean.train
```

#### Run with config 

```
$ python -m clean.train --config=<add-config-path>
```

#### Run with sh (chain experiments)

```
$ bash run.sh
```


#### Run with nohup
```
$ nohup bash run.sh &
```

and follow the updates with

```
$ tail -f nohup.out
```

or on W&B.

