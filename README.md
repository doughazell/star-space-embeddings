# GitHub README
Versions:
* Python 3.9
* scikit-learn 1.1.3

This repo is the code for:
* https://medium.com/@pavan.11.1987/embedding-all-things-with-starspace-and-keras-1df46e4db8f0

### MacOS Catalina (10.15.7) install
Install Python 3.9 (using 'pyenv' if you don't already use it to upgrade the system version, Python 2.7.16)
```
$ brew install pyenv
```

* Add 'pyenv' config to '.bashrc':
  
  eval "$(pyenv init --path)"
```
$ source .bashrc
```

* Set 'pyenv' version
```
$ pyenv install 3.9
$ pyenv global 3.9
```

Install TensorFlow + other necessary packages
```
$ pip install tensorflow
$ pip install scipy
$ pip install matplotlib
```

* https://pypi.org/project/sklearn/
  
  This repo implements the brownout strategy for deprecating the 'sklearn' package on PyPI.
  * use 'pip install scikit-learn' rather than 'pip install sklearn'

    (since 'sklearn' is the import name and 'scikit-learn' is the project name)
```
$ pip install scikit-learn
```

Clone 'https://github.com/pavan111987/star-space-embeddings'
```  
$ git clone https://github.com/pavan111987/star-space-embeddings
```

### Run 'star-space-embeddings'
```
$ cd star-space-embeddings
```

* Redirect 'stderr' to '/dev/null' to prevent 10mins of scroll lines from 'confusion_matrix(y_true, y_pred)' 

  having previously redirected 'stdout' for just that method:
```
$ time python star_space.py 2> /dev/null
```
