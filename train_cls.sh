if [ -f cls/cls_model_24.pth ] ; then
	python train_classification.py --model cls/cls_model_24.pth
else
	python train_classification.py
fi
