
%% Paths
data_path = '../data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set
%% Getting results 
clc;
test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
bboxes=[];
confidences= [];
image_ids= {};
faceDetector = vision.CascadeObjectDetector;
f=1;
for i = 1:length(test_scenes)
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    cur_bboxes= faceDetector(img);
    for j=1:size(cur_bboxes,1)
        bboxes(f,1:2)= cur_bboxes(j,1:2);
        bboxes(f,3)= cur_bboxes(j,1) + cur_bboxes(j,3);
        bboxes(f,4)= cur_bboxes(j,2) + cur_bboxes(j,4);
        image_ids(end+1,1)={test_scenes(i).name};
        confidences(end+1,1)=1;
        f=f+1;
    end
end



%% Test the results

[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)