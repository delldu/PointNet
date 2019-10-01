if [ -f seg/seg_model_24.pth ] ; then
	python train_segmentation.py --model seg/seg_model_24.pth
else
	python train_segmentation.py
fi
