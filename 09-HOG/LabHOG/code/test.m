
%% Paths and parameters

data_path = '../data/'; %change if you want to work with a network copy
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

feature_params = struct('template_size', 36, 'hog_cell_size', 6);
w= load('w.mat'); w= w.w;
b= load('b.mat'); b=b.b;
%% Run detector on test set. 

[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params,1);

%% Test
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)