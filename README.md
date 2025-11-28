# FoldARE: Folding and Analysis of RNA ensembles

FoldARE is a computatonal tool for the prediction and analysis of RNA secondary structure.  
The prediction step is based on a two-step strategy:  

 &nbsp;&nbsp;step 1 - from an RNA fasta:   
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a)  predicts an ensemble,   
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b)  through stastical weighting creates a pseudo-SHAPE co-input  
 &nbsp;&nbsp;step 2 - from the same RNA fasta, uses the pseudo-SHAPE co-input to predict the final 2D structure 
    
It builds on the combination of some available, well established, methods: V, R, L, E  (for ViennaRNA, RNAstructure, LinearFold, EternaFold).  Any of these methods can be used interchangeably for creating ensembles (step1) or predicting w/ the pseudo-SHAPE co-input (step2).  
In its default mode, it runs EternaFold for step1 and RNAstructure for step2.

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/FoldARE.git
cd FoldARE
```
note:  FoldARE pipeline requires command-line installations of the four external methods (V, R, L, E mentioned above);&nbsp; once installed, update the path accordingly, in the config.yaml file. &nbsp; For example, for LinearFold (if installed in /home/user/) edit the config.yaml :

LinearFold:  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;executable: "/home/user/LinearFold/linearfold"
executable: "/home/user/LinearFold/linearfold"  
  
similarly, edit the executable path for RNAstructure, EternaFold, ViennaRNA.  
Versions used for development:  
ViennaRNA (v2.7.0), &nbsp;RNAstructure (v6.5), &nbsp;LinearFold (commit c3ee9bd from 29.08.2022), &nbsp;EternaFold (commit 13d2487 from 17.07.2024)  

  
### requirements  
FoldARE was implemented in Python 3.10; Python libraries included: numpy, pandas, plotly, ruamel.yaml.  
The full micromamba environment is provided in dependencies.yaml ; to reconstruct the environment:  
```bash
micromamba env create -f dependencies.yaml
```

## PREDICTION 

```bash
python foldare.py -s input_fasta
```
arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">    -s: RNA sequence in FASTA format </pre>
optional arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">
	-e: choice of the method used to create the ensemble of step 1 
	-p: choice of method used to predict (w/ pseudo-SHAPE co-input)
	--ens_n: max number of models in the ensemble
</pre>
example:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;"> python foldare.py -s test.fasta -e E -p R --ens_n 75</pre>

## ANALYSIS  

The analysis section of FoldARE is based on detailed comparisons of different ensembles, created by (at least 2 of) the 4 methods (V, R, L, E).   
FoldARE allows for three different analyses:  
&nbsp;&nbsp;&nbsp;&nbsp;1. comparing all 4 methods, with &nbsp; compare_all.py     
&nbsp;&nbsp;&nbsp;&nbsp;2. comparing pairwise any combinations of the 4 methods, with &nbsp;compar_pair.py  
&nbsp;&nbsp;&nbsp;&nbsp;3. assessing the effect of (m6A) modifications on RNA ensembles, with &nbsp; compare_m6A.python
### compare all
```bash
python compare_all.py -s input_fasta
```
arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">    -s: RNA sequence in FASTA format </pre>
optional arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">
    -n: ensemble size (set the size of ensemble)
    --top_n: number of top models (for each method) to be considered for the aggregate scoring
</pre>
example:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;"> python compare_all.py -s test.fasta -n 50 --top_n 10 </pre>


### compare pair
```bash
python compare_pair.py -s input_fasta -e1 method -e2 method
```
arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">
   -s: RNA sequence in FASTA format 
   -e1: choice of the first method to compare 
   -e2: choice of the second method to compare
</pre>
optional arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">
   -n: ensemble size (set the size of ensemble)
   --top_n: number of top models (for each method) to be considered for the aggregate scoring
</pre>
example:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;"> python compare_pair.py -s test.fasta -e1 L -e2 E -n 50 --top_n 10 </pre>

### compare m6A
```bash
python compare_m6a.py -s input_fasta -m mods.txt 
```
arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">
   -s: RNA sequence in FASTA format 
</pre>
optional arguments:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;">
   -n = ensemble size (set the size of ensemble)
   --tool = tool used for m6A modification (options: V, R, both; default = "--tool both")
</pre>
example:
<pre style="font-family: Courier New; font-size: 85%; margin-left: 1em;"> python compare_m6a.py -s test.fasta -m mods.txt -n 50 --tool R </pre>




