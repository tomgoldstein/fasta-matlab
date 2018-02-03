
FASTA (Fast Adaptive Shrinkage/Thresholding Algorithm) is an efficient, easy-to-use implementation of the Forward-Backward Splitting (FBS) method (also known as the proximal gradient method) for regularized optimization problems. Many variations on FBS are available in FASTA, including the popular accelerated variant FISTA (Beck and Teboulle ’09), the adaptive stepsize rule SpaRSA (Wright, Nowak, Figueiredo ’09), and other variants described in the review <a href="https://arxiv.org/abs/1411.3406">A Field Guide to Forward-Backward Splitting with a FASTA Implementation.</a>  Whether the problem you are solving is simple or complex, FASTA makes things easy by handling issues like stepsize selection, acceleration, and stopping conditions for you.  

## How to use FASTA?
Before using FASTA, please see the following:

> [The FASTA user's guide](/code/user_guide.pdf) <br>
> [A Field Guide to Forward-Backward Splitting with a FASTA Implementation](https://arxiv.org/abs/1411.3406) <br>

We also suggest seeing the [main FASTA webpage](https://www.cs.umd.edu/~tomg/projects/fasta/) for a more detailed
overview of FASTA, and of forward-backward optimization methods in general.  


## What can FASTA solve?
FASTA targets problems of the form

minimize f(Ax)+g(x)

where "A" is a linear operator, "f" is a differentiable function, and "g" is a “simple” (but possibly non-smooth) function. 
Problems of this form including sparse least-squares (basis-pursuit), lasso, total-variation denoising, matrix completion, 
and many more. The FASTA implementation is incredibly flexible; users can solve almost anything by providing their 
own "f," g," and "A." However, for users that want quick out-of-the-box solutions, simple customized solvers are 
provided for the following problems. See the FASTA user’s manual for details.

For a more extensive list of problems, and their mathematical formulations, see the
[main FASTA webpage](https://www.cs.umd.edu/~tomg/projects/fasta/).

## About the Authors
Fasta was developed by:

>   [Tom Goldstein](/) - University of Maryland  <br>
>   [Christoph Studer](http://www.csl.cornell.edu/~studer/) - Cornell University  <br>
>   [Richard Baraniuk](http://web.ece.rice.edu/richb/) - Rice University

## How to cite FASTA
If you find that FASTA has contributed to your published work, please include the following citations:

	@article{GoldsteinStuderBaraniuk:2014,
	  Author = {Goldstein, Tom and Studer, Christoph and Baraniuk, Richard},
	  Title = {A Field Guide to Forward-Backward Splitting with a {FASTA} Implementation},
	  year = {2014},
	  journal = {arXiv eprint},
	  volume = {abs/1411.3406},
	  url = {http://arxiv.org/abs/1411.3406},
	  ee = {http://arxiv.org/abs/1411.3406}
	}

	@misc{FASTA:2014,
	  Author = {Goldstein, Tom and Studer, Christoph and Baraniuk, Richard},
	  title = {{FASTA}:  A Generalized Implementation of Forward-Backward Splitting},
	  note = {http://arxiv.org/abs/1501.04979},
	  month = {January},
	  year = {2015}
	}
