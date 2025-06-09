#!/bin/bash
echo "[Info] Mode: $1"
mode=$1
shift

if [ "$mode" = "segmentation" ]
then 
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/ves_seg-S-GAN/config.yml --epoch 30 "$@" && \
    chmod -R 777 /var/OCTA500/labels && \
    echo "[Info] Segmentation completed and results saved to /var/OCTA500/labels."  

elif [ "$mode" = "generation" ]
then
#for generating vessel graphs, the first argument is the number of samples to generate
# if no argument is provided, it defaults to 100
# then it runs the test script to generate the images using the GAN model
# then it visualizes the generated vessel graphs and generates the labels
    num_samples=$1
    shift

    python /home/OCTA-seg/generate_vessel_graph.py --config_file /home/OCTA-seg/docker/vessel_graph_gen_docker_config.yml --num_samples 100
    echo "[Info] Vessel graphs generated."
    
    echo "[Info] Running GAN test script..."
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/GAN/config.yml --epoch 150
    echo "[Info] GAN test script completed."
  
    echo "[Info] Visualizing vessel graphs..."
    python /home/OCTA-seg/visualize_vessel_graphs.py --source_dir /var/generation/vessel_graphs --out_dir /var/generation/labels --resolution "640,640,16" --binarize 
    echo "[Info] Vessel graphs visualized."
    chmod -R 777 /var/generation

    #This will first generate the vessel graphs which will be saved in /var/generation/vessel_graphs,
    #then it will run the GAN test script to generate the images which incorporates adversarial training of the GAN model (adds realism to the generated images)
    #then it will visualize the generated vessel graphs and generate the labels which will be saved in /var/generation/labels.
    #The LABELS are the binary masks of the vessel graphs
elif [ "$mode" = "extraction" ]
then
# This will extract the vessel graphs from the segmented images
    echo "[Info] Extracting vessel graphs..."
    python /home/OCTA-seg/extract_vessel_graph.py --config_file /home/OCTA-seg/docker/vessel_graph_extraction_docker_config.yml  && \
    chmod -R 777 /var/OCTA500/vessel_graphs 
    echo "[Info] Vessel graphs extracted and saved to /var/OCTA500/vessel_graphs."

elif [ "$mode" = "train" ]
then
    echo "[Info] Training GAN model..."
    python /home/OCTA-seg/train.py --config_file /home/OCTA-seg/docker/trained_models/new_GAN/config_cycle_gan.yml --epoch 150 && \
    chmod -R 777 /var/trained_models/new_GAN/checkpoints && \
    echo "[Info] GAN model trained and checkpoints saved to /var/trained_models/new_GAN/checkpoints."

elif [ "$mode" = "transformation" ]
then
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/GAN/config.yml --epoch 150 "$@" && \
    chmod -R 777 /var/generation/images

elif [ "$mode" = "visualization" ]
then
    python /home/OCTA-seg/visualize_vessel_graphs.py --source_dir /var/generation/vessel_graphs --out_dir /var/generation/labels --resolution "640,640,16" --binarize && \
    chmod -R 777 /var/generation/labels

elif [ "$mode" = "3d_reconstruction" ]
then
    python /home/OCTA-seg/test.py --config_file /home/OCTA-seg/docker/trained_models/reconstruction_3d/config.yml --epoch 60 "$@" && \
    chmod -R 777 /var/reconstructed
else
    echo "Mode $mode does not exist. Choose segmentation, generation or translation."
    exit 1
fi

