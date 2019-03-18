%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations

addpath benchmarks_modified
clear all;
close all;
clc;

%General 
%imgDir = 'BSR/BSDS500/data/images/test';
%gtDir = 'BSR/BSDS500/data/groundTruth/test';
nthresh = 99;

%Watershed
%inDir = 'test/watershed_cells/';
outDir_W = 'evaluation/test/watershed_cells/';
%mkdir(outDir_W);
%fprintf('***********************Watershed***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_W, nthresh);
%toc;

%k-means with filter
%inDir = 'test/kmeans_cells_filter/';
outDir_K = 'evaluation/test/kmeans_cells_filter/';
%mkdir(outDir_K);
%fprintf('***********************K-Means_filter***********************')
%tic;
%allBench_fast(imgDir, gtDir, inDir, outDir_K, nthresh);
%toc;

%Pablo
outDir_Ucm2 = 'evaluation/test/ucm2/';

%Test
plot_eval_test(outDir_W,outDir_K,outDir_Ucm2)
