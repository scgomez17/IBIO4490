%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations

addpath benchmarks_modified
clear all;
close all;
clc;

%General 
%imgDir = 'BSR/BSDS500/data/images/val';
%gtDir = 'BSR/BSDS500/data/groundTruth/val';
nthresh = 99;

%Watershed
%inDir = 'val/watershed_cells/';
outDir_W = 'evaluation/val/watershed/';
%mkdir(outDir_W);
fprintf('***********************Watershed***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_W, nthresh);
%toc;

%Watershed filter
%inDir = 'val/watershed_cells_filter/';
outDir_WF = 'evaluation/val/watershed_filter/';
%mkdir(outDir_WF);
fprintf('***********************Watershed_filter***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_WF, nthresh);
%toc;

%k-means
%inDir = 'val/kmeans_cells/';
outDir_K = 'evaluation/val/kmeans/';
%mkdir(outDir_K);
fprintf('***********************K-Means***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_K, nthresh);
%toc;

%k-means filter
%inDir = 'val/kmeans_cells_filter/';
outDir_KF = 'evaluation/val/kmeans_filter/';
%mkdir(outDir_KF);
fprintf('***********************K-Means_filter***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_KF, nthresh);
%toc;

%Plot val PR curve
plot_eval_val(outDir_W,outDir_WF,outDir_K,outDir_KF)
