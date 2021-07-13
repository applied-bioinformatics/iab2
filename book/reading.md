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

# Reading this book

Part 1 of this book, _Microbiome Bioinformatics with QIIME 2_, is intended to be read sequentially. This section will teach you everything you need to know to get started using QIIME 2 for your own microbiome research, using example data sets along the way. As you learn to use the tools, you'll be introduced to underlying theory in microbiome bioinformatics. I also sprinkle in tips and tricks on how to get more out of QIIME 2, including how I use QIIME 2, so you'll be a QIIME 2 power user by the time you're done!  

If you'd like to learn more about the underlying algorithms used in microbiome research, some of those are covered in Part 2 chapters. The Part 2 chapters can be read sequentially but don't have to be. These chapters derive from my earlier online book, _An Introduction to Applied Bioinformatics_, but have been re-worked to focus on microbiome-relevant problems. If you're not interested in algorithmic details, but only on learning how to use QIIME 2, you can safely skip Part 2. Many of the algorithms discussed in these chapters are fundamentals of bioinformatics, so if you're interested in becoming a bioinformatician or bioinformatics software developer, you should read these.  

Part 3 is intended to be read sequentially. This will take you through different aspects of developing with QIIME 2. Most developers will be interested in creating plugins, so building, documenting, testing, and distributing QIIME 2 plugins is covered first. Building QIIME 2 plugins is a great way to get your bioinformatics tools in the hands of a lot of microbiome researchers. It comes with other benefits as well: QIIME 2's unique retrospective data provenance tracking system will be used whenever your plugins are run, ensuring reproducibility of the work and providing information that will help you provide technical support to users; your users will be able to access your functionality through any of QIIME 2's interfaces, which provide access to the same tools through interfaces geared toward users with different levels of computational sophistication; and QIIME 2 plugins are often published as stand-alone papers, so building QIIME 2 plugins can help you get the publications you may need to support your career. Part 3 also covers developing QIIME 2 interfaces. This content will help you integrate QIIME 2 as a component in another system you're developing, or allow you to implement your own ideas to make QIIME 2 more accessible. If you are considering wrapping QIIME 2 in another system, this content will give you tools to simplify that work through the QIIME 2 software development kit (SDK). QIIME 2 and the development team enthusiastically support your plugin and interface development efforts! Don't forget to get in touch on the forum with your QIIME 2 development questions.  

## Reading interactively with Binder

This book can be read statically or interactively. The simplest way to read interactively is with [Binder](https://mybinder.org/). When chapters are available to read interactively, you'll see a rocket ship icon toward the top-right of the page. The current page can be read interactively, allowing you to execute the following code block (making any modifications you choose to make). If you'd like to try it out, click the rocket ship icon, and then the Launch Binder box that pops up beneath it. 

Code blocks will be presented throughout the book. If you're reading a static version of the book, the output of running the code will be presented below the code block. If you're reading an interactive version of the book, you'll have to run the code cell to see the output. 

Here's an example Python 3 code cell that imports the scikit-bio library and prints some information to the screen. 

```{code-cell}
import skbio

print(skbio.title)
print(skbio.art)
```

## Conventions 

Text formats:

- **bolded text** indicates new terms that are being introduced. These will ultimately show up in the glossary as well. 
- _italicized text_ is used for emphasizing import ideas. 
- `fixed-width text` is used to indicate literal values, for example file names or parameters that are used by a command. 

## Special text blocks

Throughout the text, you'll sometimes see blocks of text that stand out as follows:

```{admonition} Tip: Learn programming a little at a time
:class: tip
Spending a few minutes coding every day is a great way to build your programming skills, empower your own research, and make you more competitive for future career opportunities. Python 3 and R are great languages to get started with for bioinformatics, but just about any language will be fine for learning. 
```

These are intended to share ideas related to what is being presented in the text. I'll use a few types of these special text blocks:

- Video: a video is available that may help you understand this content. 
- Tip: an idea that may help you in your learning.
- Warning: a common error that is encountered. 
- Note: a idea that is related to the content being presented, but is a bit off topic relative to the current discussion. 
- Food for thought: a question or topic to think about related to the current content
- Exercise: a suggestion for something to experiment to help you develop your understanding of the current content
- Jargon: discussed a term that is common among microbiome or bioinformatics practitioners, but which may not be immediately clear to beginners. 
- Attribution: used to indicate locations where ideas or content derived from open access resources, such as posts on the QIIME 2 Forum or from iterations of the QIIME 2 documentation.
