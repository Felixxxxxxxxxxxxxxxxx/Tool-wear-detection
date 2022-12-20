#!/usr/local_rwth/bin/zsh
 
### #SBATCH directives need to be in the first part of the jobscript

 
#SBATCH --ntasks=1

#SBATCH -t 00:60:00
 
#SBATCH --mem-per-cpu=40000M   
 
#SBATCH --job-name="lwf"
 
#SBATCH --output=output.%J.log

#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:1


### your code goes here, the second part of the jobscript

source ~/.bashrc
conda activate tensorflow
python3 train_model_3.py
 
### !!! DON'T MIX THESE PARTS !!!