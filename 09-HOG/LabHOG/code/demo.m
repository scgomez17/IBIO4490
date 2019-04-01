
%% Paths and parameters

data_path = '../data/'; %change if you want to work with a network copy
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

feature_params = struct('template_size', 36, 'hog_cell_size', 6);
w= load('w.mat'); w= w.w;
b= load('b.mat'); b=b.b;
%% Run detector on test set.

[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params,2);

%% Visualization

for i=1:size(bboxes,1)
    img= imread(fullfile(test_scn_path,image_ids{i}));
    if i==1
        name= image_ids{i};
        imshow (img)
        rectangle ('Position', [bboxes(i,1) bboxes(i,2) bboxes(i,3)-bboxes(i,1) bboxes(i,4)-bboxes(i,2)],...
            'EdgeColor','g','LineWidth',2)
    else
        name2= image_ids{i};
        if isequal(name,name2)
            rectangle ('Position', [bboxes(i,1) bboxes(i,2) bboxes(i,3)-bboxes(i,1) bboxes(i,4)-bboxes(i,2)],...
                'EdgeColor','g','LineWidth',2)
        else
            fprintf('Press any key')
            pause()
            imshow(img)
            rectangle ('Position', [bboxes(i,1) bboxes(i,2) bboxes(i,3)-bboxes(i,1) bboxes(i,4)-bboxes(i,2)],...
                'EdgeColor','g','LineWidth',2)
            name= name2;
        end
    end
end
