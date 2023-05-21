# Handwritten Chinese Character Guidance System



Here is a brief introduction of each folder in this repository.

- **PenGripPostureClassify**

This folder is the code implement of our <u>Pen Grip Posture Classification Module</u>. It includes the training and predicting code for the pen grip posture classification. The self-built  dataset and trained model can be downloaded in this [link](https://drive.google.com/drive/folders/1olMh_5u8aiRBAkjXN68F6oSOdx41EIs4?usp=share_link). 

- **PenStrokeOrderJudgement**

This folder is the code implement of our <u>Stroke Extraction Module</u> and <u>Stroke Order Judgement Module</u>. It includes the code for pen tip tracking, pen ballpoint detection, stroke extraction, stroke type identification, chineses character detection and stroke order comparson. The required environment can be installed through the requirements.txt. 
Add pysot to PYTHONPATH

```
export PYTHONPATH=/path/to/pysot:$PYTHONPATH
```

To run the code, execute the following code:

```
bash ./run_demo.sh
```

It can autonomously detect the if stroke writing order is correct when writing chinese characters on the paper. The result will be saved in ./demo/output

- **Chinese-OCR**

This folder contains the training and testing code for Chinese charater recognition. The train dataset can be downloaded from this [link](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)
