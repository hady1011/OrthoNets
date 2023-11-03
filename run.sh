#!/bin/bash

. ./settings.sh

experiment=$1
mode=${2:-"test_pretrained"}

mode_error_message="An incorrect mode (second input) was entered. Please enter 'test_pretrained', 'train' or 'test' as the second argument."

if [ $experiment = "orthonet_34_imagenet" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_34 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "pretrained_models/1_orthonet_34_imagenet.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_34 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "trained_models/1_orthonet_34_imagenet.pth.tar"
	elif [ $mode = "train" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_34 --b \
			$batch_size --dataset_location $imagenet_folder --numclasses 1000 \
			--model-save-name "1_orthonet_34_imagenet.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_50_imagenet" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_50 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "pretrained_models/2_orthonet_50_imagenet.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_50 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "trained_models/2_orthonet_50_imagenet.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_50 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000\
			--model-save-name "2_orthonet_50_imagenet.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_mod_50_imagenet" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "pretrained_models/3_orthonet_mod_50_imagenet.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "trained_models/3_orthonet_mod_50_imagenet.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000\
			--model-save-name "3_orthonet_mod_50_imagenet.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_mod_50_birds" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $birds_folder --numclasses 450 --evaluate True \
			--evaluate_model "pretrained_models/4_orthonet_mod_50_birds.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $birds_folder --numclasses 450 --evaluate True \
			--evaluate_model "trained_models/4_orthonet_mod_50_birds.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $birds_folder --numclasses 450\
			--model-save-name "4_orthonet_mod_50_birds.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_mod_50_places" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $places_folder --numclasses 365 --evaluate True \
			--evaluate_model "pretrained_models/5_orthonet_mod_50_places.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $places_folder --numclasses 365 --evaluate True \
			--evaluate_model "trained_models/5_orthonet_mod_50_places.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_50 \
			--b $batch_size --dataset_location $places_folder --numclasses 365\
			--model-save-name "5_orthonet_50_places.pth.tar"
	else
		echo $mode_error_message
	fi
elif [ $experiment = "fcanet_50_birds" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a fcanet_50 \
			--b $batch_size --dataset_location $birds_folder --numclasses 450 --evaluate True \
			--evaluate_model "pretrained_models/6_fcanet_50_birds.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a fcanet_50 \
			--b $batch_size --dataset_location $birds_folder --numclasses 450 --evaluate True \
			--evaluate_model "trained_models/6_fcanet_50_birds.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a fcanet_50 \
			--b $batch_size --dataset_location $birds_folder --numclasses 450\
			--model-save-name "6_fcanet_50_birds.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "fcanet_50_places" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a fcanet_50 \
			--b $batch_size --dataset_location $places_folder --numclasses 365 --evaluate True \
			--evaluate_model "pretrained_models/7_fcanet_50_places.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a fcanet_50 \
			--b $batch_size --dataset_location $places_folder --numclasses 365 --evaluate True \
			--evaluate_model "trained_models/7_fcanet_50_places.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a fcanet_50 \
			--b $batch_size --dataset_location $places_folder --numclasses 365\
			--model-save-name "7_fcanet_50_places.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_101_imagenet" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_101 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "pretrained_models/8_orthonet_101_imagenet.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_101 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "trained_models/8_orthonet_101_imagenet.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_101 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000\
			--model-save-name "8_orthonet_101_imagenet.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_mod_101_imagenet" ]; then
	if [ $mode = "test_pretrained" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_101 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "pretrained_models/9_orthonet_mod_101_imagenet.pth.tar"
	elif [ $mode = "test" ]; then
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_101 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000 --evaluate True \
			--evaluate_model "trained_models/9_orthonet_mod_101_imagenet.pth.tar"
	elif [ $mode = "train" ]; then 
		python -m torch.distributed.launch --nproc_per_node=$number_of_gpus main.py -a orthonet_mod_101 \
			--b $batch_size --dataset_location $imagenet_folder --numclasses 1000\
			--model-save-name "9_orthonet_101_imagenet.pth.tar"
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_mod_50_coco_faster_rcnn" ]; then
	if [ $mode = "test_pretrained" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_orthonet50_fpn_1x_coco.py' \
			'pretrained_models/10_orthonet_mod_50_coco_faster_rcnn.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "test" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_orthonet50_fpn_1x_coco.py' \
			'trained_models/10_orthonet_mod_50_coco_faster_rcnn/latest.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "train" ]; then
		./mmdetection/tools/dist_train.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_orthonet50_fpn_1x_coco.py' \
			$number_of_gpus --work-dir "trained_models/10_orthonet_mod_50_coco_faster_rcnn" \
			--dataset_location $coco_folder
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_mod_101_coco_faster_rcnn" ]; then
	if [ $mode = "test_pretrained" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_orthonet101_fpn_1x_coco.py' \
			'pretrained_models/11_orthonet_mod_101_coco_faster_rcnn.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "test" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_orthonet101_fpn_1x_coco.py' \
			'trained_models/11_orthonet_mod_101_coco_faster_rcnn/latest.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "train" ]; then
		./mmdetection/tools/dist_train.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_orthonet101_fpn_1x_coco.py' \
			$number_of_gpus --work-dir "trained_models/11_orthonet_mod_101_coco_faster_rcnn" \
			--dataset_location $coco_folder
	else
		echo $mode_error_message
	fi

elif [ $experiment = "fcanet_50_coco_mask_rcnn" ]; then
	if [ $mode = "test_pretrained" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_freqnet50_fpn_1x_coco_resnet_config.py' \
			'pretrained_models/12_fcanet_50_coco_mask_rcnn.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "test" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_freqnet50_fpn_1x_coco_resnet_config.py' \
			'trained_models/12_fcanet_50_coco_mask_rcnn/latest.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "train" ]; then
		./mmdetection/tools/dist_train.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_freqnet50_fpn_1x_coco_resnet_config.py' \
			$number_of_gpus --work-dir "trained_models/12_fcanet_50_coco_mask_rcnn" \
			--dataset_location $coco_folder
	else
		echo $mode_error_message
	fi

elif [ $experiment = "orthonet_mod_50_coco_mask_rcnn" ]; then
	if [ $mode = "test_pretrained" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_orthonet50_fpn_1x_coco_resnet_config.py' \
			'pretrained_models/13_orthonet_mod_50_coco_mask_rcnn.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "test" ]; then
		./mmdetection/tools/dist_test.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_orthonet50_fpn_1x_coco_resnet_config.py' \
			'trained_models/13_orthonet_mod_50_coco_mask_rcnn/latest.pth' $number_of_gpus --eval bbox \
			--dataset_location $coco_folder
	elif [ $mode = "train" ]; then
		./mmdetection/tools/dist_train.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_orthonet50_fpn_1x_coco_resnet_config.py' \
			$number_of_gpus --work-dir "trained_models/13_orthonet_mod_50_coco_mask_rcnn" \
			--dataset_location $coco_folder
	else
		echo $mode_error_message
	fi

else
	echo "Experiment name (option 1) not recognized. Please see README.md"
fi

exit 0
