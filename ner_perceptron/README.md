The header in `collins_perceptron.py` should be detailed enough.

```bash
$ tree -L 1
.
├── Makefile -- make data test run
├── README.md -- This file.
├── collins_perceptron.py -- Implements the structured perptron tagger.
├── setup.py -- Builds the Cython .so file.
├── test.py -- Tests the Python and Cython implementations of Viterbi.
└── viterbi.pyx -- Implements the Viterbi decoder in Cython. 50x speedup.
```

Usage:
- `make data` will download and unzip the Brown corpus if not present.
- `make test` will compile the Cython code, functional check the implementations and time the decoding.
- `make run` will launch the tagger script.
- `make clean` will delete all .pyc and .so files.

Dependencies (beyond the Python standard libraries):
```
plac==0.9.6
tqdm==4.23.4
ipdb==0.10.3
numpy==1.16.2
Cython==0.27.3
``` 
