## FoldARE
<pre>
A computatonal tool for the prediction and analysis of RNA 2D structure.
</pre>
## PREDICTION 
<pre>
2D prection is perfomed via a 2 step strategy: 
step 1 - from an RNA fasta: (i) predicts an ensemble, (ii) through stastical weighting creates a pseudo-SHAPE co-input 
step 2 - from the same RNA fasta, uses the pseudo-SHAPE co-input to predict the final 2D structure 
methods: V, R, L, E (for ViennaRNA, RNAstructure, LinearFold, EternaFold, respectively)

usage:
python foldare.py -s input_fasta -e method -p method 
	-s: the RNA sequence in fasta format
	-e: the choice of the method used to create the ensemble of step 1 
	-p: the choice of method used to predict (w/ pseudo-SHAPE co-input)
	optional arguments:
	--ens_n: max number of models in the ensemble 
	example:
     python foldare.py -s test.fasta -e E -p R --ens_n 75
</pre>
## ANALYSIS
<pre>
The analysis part of FoldARE is based on detailed comparisons of different ensembles, as created by the 4 methods (V, R, L, E) considered. 
FoldARE allows for three different analyses:
1. comparing all 4 methods, with the script compare_all.py   
2. comparing pairwise any combinations of the 4 methods, with compar_pair.py
3. assess the effect of RNA modifications (m6A), comparing (unmodified vs m6A) ensembles, with compare_m6A.python__

Usage:
python compare_all -s input_fasta 
	-s: the RNA sequence in fasta format
	optional arguments:
	-n = ensemble size (set the size of ensemble)
	--top_n = number of top models (for each method) to be considered for the aggregate scoring
    example: 
      python compare_all -s test.fasta -n 50 --top_n 10
 
python compare_pair -s input_fasta -e1 method -e2 method 
	-s: the RNA sequence in fasta format 
	-e1: the choice of the first method to compare 
	-e2: the choice of the second method to compare
	optional arguments:
	-n = ensemble size (set the size of ensemble)
	--top_n = number of top models (for each method) to be considered for the aggregate scoring
    example: 
      python compare_pair -s test.fasta -e1 L -e2 E -n 50 --top_n 10
	  
python compare_m6a.py -s input_fasta -m mods.txt 
	-s: the RNA sequence in fasta format 
	-m = input text file with specified positions to modify (format: one position per row. 
           e.g. if 3 positions, 3 rows each with one position)
	optional arguments:
	-n = ensemble size (set the size of ensemble)
	--top_n = number of top models (for each method) to be considered for the aggregate scoring
    example: 
      compare_m6a.py -s test.fasta -m mods.txt -n 50 --top_n 10  
</pre>
