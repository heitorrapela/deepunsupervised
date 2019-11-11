# Deep Unsupervised Learning

Deep Unsupervised Learning

## Dependencies

	Python 3.6
	CUDA 10.0
	Pytorch 1.3.1
	Tensorboard 2.0.0 (Dependends of Pytorch +1.2)
	Pandas
	Scipy
	Scikit-Learn
	Matplotlib


## Creating conda env

    conda create -n deepunsupervised python=3.6 -y
    source activate deepunsupervised (or conda activate deepunsupervised)
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch -y
    conda install pandas -y
    conda install scipy -y
    conda install scikit-learn -y
    conda install -c omnia munkres -y
    conda install matplotlib -y
    conda install tensorboard -y