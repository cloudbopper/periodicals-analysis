periodicals-analysis
====================

Description
-----------

This repository implements the optical character recognition (OCR), natural-language processing (NLP) and topic modeling of historical organic periodicals.


Requirements
------------

* python 3.6+
* tesseract-ocr (https://github.com/tesseract-ocr/tesseract)
* imagemagick
* nltk
* scikit-learn
* jupyter
* pyLDAvis (for visualizing models)

I recommend using virtualenv/virtualenvwrapper to install/manage the python modules.



Install
-------

Use [Pip](https://pip.pypa.io/) to install directly from GitHub.

    pip install git+https://github.com/cloudbopper/perysis.git@master#egg=perysis

Add '-e' for an editable install.


Running
-------

Preprocessing OCR-ed text

    python -m perysis.preprocess -input_dir documents_raw -output_dir documents_processed


License
-------

periodicals-analysis is free, open source software, released under the MIT license. See `LICENSE` for details.


Contact
-------

[Akshay Sood](https://github.com/cloudbopper)
