---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Introduction

Bioinformatics, as I see it, is the application of the tools of computer science (such as programming languages, algorithms, and databases) to address biological problems (for example, inferring the evolutionary relationship between a group of organisms based on fragments of their genomes, or understanding if or how the community of microorganisms that live in my gut changes if I modify my diet). Bioinformatics is a rapidly growing field, largely in response to the vast increase in the quantity of data that biologists now grapple with. Students from varied disciplines (e.g., biology, computer science, statistics, and biochemistry) and stages of their educational careers (undergraduate, graduate, or postdoctoral) are becoming interested in bioinformatics.

*An **I**ntroduction to **A**pplied **B**ioinformatics*, or **IAB**, is a bioinformatics text available at http://readIAB.org. It introduces readers to core concepts in bioinformatics in the context of their implementation and application to real-world problems and data. IAB makes extensive use of common Python libraries, such as [scikit-learn](https://scikit-learn.org/) and [scikit-bio](http://www.scikit-bio.org), which provide production-ready implementations of algorithms and data structures taught in the text. Readers therefore learn the concepts in the context of tools they can use to develop their own bioinformatics software and pipelines, enabling them to rapidly get started on their own projects. While some theory is discussed, the focus of IAB is on what readers need to know to be effective, practicing bioinformaticians.

My goal with IAB is to make getting started in bioinformatics as accessible as possible to students from varied backgrounds, and to get more people interested in this exciting field. I'm very interested in hearing from readers and instructors who are using IAB, so get in touch with feedback, corrections, or suggestions. The best way to get in touch is using the [IAB2 issue tracker](https://github.com/applied-bioinformatics/iab2/issues).

## Who should read IAB?

IAB is written for scientists, software developers, and students interested in understanding and applying bioinformatics methods, and ultimately in developing their own bioinformatics analysis pipelines or software.

IAB was initially developed for an undergraduate course at [Northern Arizona University](http://www.nau.edu) cross-listed in computer science and biology with no pre-requisites. It therefore assumes little background in biology or computer science, however some basic background is very helpful. For example, an understanding of the roles of and relationship between DNA and protein in a cell, and the ability to read and follow well-annotated Python 3 code, are both helpful (but not necessary) to get started.

## How to read IAB

This book can be read statically or as an interactive [Jupyter Book](https://jupyterbook.org/). The simplest way to read IAB interactively is with [Binder](https://mybinder.org/). When chapters are available to read interactively, you'll see a rocket ship icon toward the top-right of the page. The current page can be read interactively, allowing you to execute the following code block (making any modifications you choose to make). If you'd like to try it out, click the rocket ship icon, and then the Launch Binder box that pops up beneath it. 

Code blocks will be presented throughout the book. If you're reading a static version of the book, the output of running the code will be presented below the code block. If you're reading an interactive version of the book, you'll have to run the code cell to see the output. 

Here's an example Python 3 code cell that imports the scikit-bio library and prints some information to the screen. 

```{code-cell}
import skbio

print(skbio.title)
print(skbio.art)
```

## Conventions used in this text

### Text formats

I have tried to be consistent in how I emphasize text throughout the book. **Bolded text** indicates new terms that are being introduced; _italicized text_ is used for emphasizing important ideas; and `fixed-width text` is used to indicate literal values, for example names of variables that are being used in the text. 

### Special text blocks

Throughout the text, you'll sometimes see blocks of text that stand out as follows:

```{admonition} Tip: Learn programming a little at a time
:class: tip
Spending a few minutes coding every day is a great way to build your programming skills, empower your own research, and make you more competitive for future career opportunities. Python 3 and R are great languages to get started with for bioinformatics, but just about any language will be fine for learning. 
```

These are intended to share ideas related to what is being presented in the text. I'll use a few types of these special text blocks:

- Video: a video that may help you understand this content. 
- Tip: an idea that may help you in your learning.
- Note: a idea that is related to the content being presented, but is a bit off topic relative to the current discussion.
- Warning: a common error that is encountered or mistake that is made.
- Food for thought: a question or topic to think about related to the current content.
- Exercise: a suggestion for something to experiment with to help you understand the current content.
- Jargon: a term that is common among bioinformatics practitioners, but which may not be immediately clear to beginners. 

## Structure of this book

This book contains four sections that are intended to be read in order. The first section, _Biological Information_, provides a gentle introduction to information processing in biological systems and relates that to information processing in computer systems. The second section, _Pairwise sequence alignment_, introduces algorithms that are at the core of biological sequence analysis. The third section, _Sequence Homology Searching_ illustrates an application of global pairwise alignment (which you will have just learned about) in annotating biological sequences taxonomically. Here you'll work through the ideas at the core of sequence database searching, which you have used if you've ever done a BLAST search (for example). Finally, the fourth section, _Machine Learning in Bioinformatics_, introduces the concepts of unsupervised and supervised learning in the context of understanding biological sequences. 

I think that these topics provide a solid foundation in bioinformatics from which you can build - similar to what you might get from a one semester undergraduate or graduate course (and this is a great resource for such a course). There are many other topics that could be covered, such as phylogenetic reconstruction, which I may add in future editions of this text. For any given edition of IAB, starting with this second edition, you can always expect to find the content that is presented to be complete. However, like most good projects, I don't expect that I'll ever really be done with it. So if there are additional topics you'd like to see covered in this way, please let me know. 

Thanks for your interest in this text and in bioinformatics. Grab a cup of your favorite hot beverage, and let's get started! ☕