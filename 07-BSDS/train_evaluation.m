%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations

addpath benchmarks_modified
clear all;
close all;
clc;

%General 
%imgDir = 'BSR/BSDS500/data/images/train';
%gtDir = 'BSR/BSDS500/data/groundTruth/train';
nthresh = 99;

%Watershed
%inDir = 'train/watershed_cells/';
outDir_W = 'evaluation/train/watershed/';
%mkdir(outDir_W);
fprintf('***********************Watershed***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_W, nthresh);
%toc;
%plot_eval_individual(outDir_W)

%k-means
%inDir = 'train/kmeans_cells/';
outDir_K = 'evaluation/train/kmeans/';
%mkdir(outDir_K);
fprintf('***********************K-Means***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_K, nthresh);
%toc;
%plot_eval_individual(outDir_K)

%Plot watersheds and kmeans
plot_eval_train(outDir_W,outDir_K)
