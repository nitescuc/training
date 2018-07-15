#!/bin/bash

job_name=$1
if [ -z $job_name ] 
then
    echo 'Provide model name'
    exit 0
fi 
echo 'Creating training job '$1

aws sagemaker create-training-job \
    --training-job-name $job_name \
    --hyper-parameters '{ "sagemaker_region": "\"eu-west-1\"", "enhance": "true" }' \
    --algorithm-specification TrainingImage="263430657496.dkr.ecr.eu-west-1.amazonaws.com/robocars:1.4.1-gpu-py3",TrainingInputMode=File \
    --role-arn "arn:aws:iam::263430657496:role/service-role/AmazonSageMaker-ExecutionRole-20180512T173485" \
    --input-data-config '[{ "ChannelName": "train", "DataSource": { "S3DataSource": { "S3DataType": "S3Prefix", "S3Uri": "s3://robocars", "S3DataDistributionType": "FullyReplicated" }} }]' \
    --output-data-config S3OutputPath=s3://robocars \
    --resource-config InstanceType=ml.p2.xlarge,InstanceCount=1,VolumeSizeInGB=1 \
    --stopping-condition MaxRuntimeInSeconds=3600
