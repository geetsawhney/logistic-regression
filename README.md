# logistic-regression

## Requirement
* Python 3
* Create a new virtual environment.
```python3 -m venv <name of environment>```
* Activate the virtual environment.
```source <name of environment>/bin/activate```
* Install packages as specified in requirements.txt.
```pip3 install -r requirements.txt```
* Optional: Deactivate the virtual environment, returning to your system's setup.
```deactivate```
in the directory where you have created the virtual environment.
* To run in train mode
```	python3 classify.py --mode train --algorithm algorithm_name --model-file model_file --data train_file```
* To run in test mode
```python3 classify.py --mode test --model-file model_file --data test_file --predictions-file predictions_file```

#### Example
```python3 classify.py --mode train --algorithm perceptron --model-file speech.perceptron.model --data speech.train```
To run the trained model on development data:
```python3 classify.py --mode test --model-file speech.perceptron.model --data speech.dev \--predictions-file speech.dev.predictions```





## Data
The data are provided in what is known as SVM-light format. Each line contains a single example:
```0 1:-0.2970 2:0.2092 5:0.3348 9:0.3892 25:0.7532 78:0.7280```
The first entry on the line is the label. The label can be an integer (0/1 for binary classification) or a real valued number (for regression.) The classification label of $-1$ indicates unlabeled. Subsequent entries on the line are features. The entry {\tt 25:0.7532} means that feature $25$ has value $0.7532$. Features are 1-indexed.
Model predictions are saved as one predicted label per line in the same order as the input data. The code that generates these predictions is provided in the library. The script {\tt compute\_accuracy.py} can be used to evaluate the accuracy of your predictions for classification:
```python3 compute_accuracy.py data_file predictions_file```

We consider several real world binary classification datasets taken from a range of applications. Each dataset is in the same format (described below) and contains a train, development, and test file. You will train your algorithm on the train file and use the development set to test that your algorithm works. The test file contains unlabeled examples that we will use to test your algorithm.

### Biology
Biological research produces large amounts of data to analyze. Applications of machine learning to biology include finding regions of DNA that encode for proteins, classification of gene expression data, and inferring regulatory networks from mRNA and proteomic data.
	
Our biology task of characterizing gene splice junction sequences comes from molecular biology, a field interested in the relationships of DNA, RNA, and proteins. Splice junctions are points on a sequence at which superfluous'' RNA is removed before the process of protein creation in higher organisms. Exons are nucleotide sequences that are retained after splicing while introns are spliced out. The goal of this prediction task is to recognize DNA sequences that contain boundaries between exons and introns. Sequences contain exon/intron (EI) boundaries, intron/exon (IE) boundaries, or do not contain splice examples.
For a binary task, you will classify sequences as either EI boundaries (label 1) or non-splice sequences (label 0). Each learning instance contains a 60 base pair sequence (ex. ACGT), with some ambiguous slots. Features encode which base pair occurs at each position of the sequence.

### Finance
Finance is a data rich field that employs numerous statistical methods for modeling and prediction, including the modeling of financial systems and portfolios.\footnote{For an overview of such applications, see the proceedings of the 2005 NIPS workshop on machine learning in finance. (http://www.icsi.berkeley.edu/~moody/MLFinance2005.htm) (http://www.icsi.berkeley.edu/~moody/MLFinance2005.htm)
Our financial task is to predict which Australian credit card applications should be accepted (label 1) or rejected (label 0). Each example represents a credit card application, where all values and attributes have been anonymized for confidentiality. Features are a mix of continuous and discrete attributes and discrete attributes have been binarized.
	
### NLP
Natural language processing studies the processing and understanding of human languages. Machine learning is widely used in NLP tasks, including document understanding, information extraction, machine translation and document classification.
Our NLP task is sentiment classification. Each example is a product review taken from Amazon kitchen appliance reviews. The review is either positive (label 1) or negative (label 0) towards the product. Reviews are represented as uni-gram and bi-grams; each one and two word phrase is extracted as a feature.
	
### Speech
Statistical speech processing has its roots in the 1980s and has been the focus of machine learning research for decades. The area deals with all aspects of processing speech signals, including speech transcription, speaker identification and speech information retrieval.
Our speech task is spoken letter identification. Each example comes from a speaker saying one of the twenty-six letters of English alphabet. Our goal is to predict which letter was spoken. The data was collected by asking 150 subjects to speak each letter of the alphabet twice.
Each spoken utterance is represented as a collection of 617 real valued attributes scaled to be between -1.0 and 1.0. Features include spectral coefficients; contour features, sonorant features, pre-sonorant features, and post-sonorant features. The binary task is to distinguish between the letter M (label 0) and N (label 1).
	
### Vision
Computer vision processes and analyzes images and videos and it is one of the fundamental areas of robotics. Machine learning applications include identifying objects in images, segmenting video and understanding scenes in film.
Our vision task is image segmentation. In image segmentation, an image is divided into segments are labeled according to content. The images in our data have been divided into 3x3 regions. Each example is a region and features include the centroids of parts of the image, pixels in a region, contrast, intensity, color, saturation and hue. The goal is to identify the primary element in the image as either a brickface, sky, foliage, cement, window, path or grass. In the binary task, you will distinguish segments of foliage (label 0) from grass (label 1).
	
### Synthetic Data
When developing algorithms it is often helpful to consider data with known properties. We typically create synthetic data for this purpose. To help test your algorithms, we are providing two synthetic datasets. These data are to help development.
	
### Easy
The easy data is labeled using a trivial classification function. Any reasonable learning algorithm should achieve near flawless accuracy. Each example is a 10 dimensional instance drawn from a multi-variate Gaussian distribution with 0 mean and a diagonal identity covariance matrix. Each example is labeled according to the presence one of 6 features; the remaining features are noise.
	
### Hard
Examples in this data are randomly labeled. Since there is no pattern, no learning algorithm should achieve accuracy significantly different from random guessing (50\%). Data is generated in an identical manner as Easy except there are 94 noisy features.
