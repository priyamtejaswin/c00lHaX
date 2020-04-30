![Alt Text](https://cdn190.picsart.com/230925670045202.gif)

Just playing with things and ideas.

If you've not gathered as much from the repo name and the gif, this code is not to be taken very seriously unless stated otherwise.

Here's a list of ***things*** i've covered so far:

**Genetic Algorithms for Text Generation**
- `monkey-typing-shakespeare.ipynb` for the notebook.
- Execute `python monkey_typing_shakespeare.py`. Enter a phrase. Be amazed. Or not. I don't care.

**Hidden Markov Models**
- Eveything's in `hmm.ipynb`. There's lot of stuff to render in there so [this nbviewer link](https://nbviewer.jupyter.org/github/priyamtejaswin/c00lHaX/blob/master/hmm.ipynb) might be better.
- This took a while to complete.
- - The intent was to (1) revist HMMs and (2) write a reference which I could keep coming back to.
- - Hence the madness with the notation, equations, and figures.
- Dervied solutions (in `Mathjax`) for all [three problems](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf): Evaluation, Decoding, Learning.
- Simple Python + Numpy implementations for all steps. Solved some toy problems from the Wiki pages.

**Decision Trees**
- A futile exercise in getting comfortable with Vim. Everything I wrote (codes, notes) was in Vim.
- Notes on CART, Pruning, etc in `notes/trees.md`.
- You can install and use [`pandoc`](https://pandoc.org/) to render this as a Latex PDF.
- - (From inside the `notes` folder, run: `pandoc trees.md -o trees.pdf`.
- Reference implementations in Python + Numpy in `dtrees` folder. Run the files individually to see colorful plots.

**Collins' Averaged Perceptron for NER tagging**
- *Boy* am I proud of this one. Just look the `ner_perceptron/` folder!!
- This is a stripped implementation of Collins' seminal [2002 paper](http://www.cs.columbia.edu/~mcollins/papers/tagperc.pdf).
- There's a `Makefile` that lists the fancy stuff that you can do: download extract the sample data, train the tagger, benchmark the Viterbi decoder, etc.
- Main motivation (apart from getting into structured prediction) was to figure out what [Cython](https://cython.org/) is all about.
- - I wrote a Cython implementation of the Viterbi decoder in `ner_perceptron/viterbi.pyx`.
- - This speeds up the computation by a factor of 6. You should check it out if you're writing a lot of custom code.
-- Cython is used heavily in [spaCy](https://spacy.io/). Its creator Matt Honnibal [says he just writes straight-up Cython](https://explosion.ai/blog/writing-c-in-cython) because *"The code is almost always exactly as fast as C/C++, because it really is just C/C++ with some syntactic sugar — but with Python "right there", should I need/want it."*

**Corner Detection and Matching**
- Wanted to explore some classifical CV stuff. Currently digging myself out of the SLAM rabbit hole.
- In `fast_corner`, you'll find notebooks implementing the [FAST corner detector](https://www.edwardrosten.com/work/rosten_2006_machine.pdf) and the [BRIEF descriptor](https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf).
- Run `python fast_corner/find_and_match.py -h`. 
- - This script accepts two images, applies FAST + BRIEF, does a brute force matching, and plots the matched features.
- - Uses `opencv` for FAST. I did implement it -- ID3 decision tree and all -- but `opencv` already has a pre-trained model.
- Papers (for reference) are tagged in the beginning of each notebook.
- Next stop, ORB.

**More corner detection**
- Implemented the [Harris Corner Detector](http://www.bmva.org/bmvc/1988/avc-88-023.pdf). Used in ORB.
- I made more of an effort in writing the notebook this time around -- `./harris_corner/harris_1988.ipynb`.
- There was a problem at work for quickly detecting barcodes. I put together something using low-level filters and `cv2` `Contour` functionality in about an hour -- `./harris_corner/barcode_corners.ipynb`. This is as hacky as it gets folks. Well, not as hacky as [this ...](https://www.youtube.com/watch?v=Bmz67ErIRa4)
