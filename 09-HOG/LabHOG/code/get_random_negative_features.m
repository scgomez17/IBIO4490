% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
num_inst=round(num_samples/num_images);
features_neg=[];
window_size=15; %Change this parameter if you want
desp= (window_size-1)/2;
f=1;
%Obtain num_samples random negatives from images.
for i=1:num_images
    img_path=strcat(non_face_scn_path,'/',image_files(i).name);
    for j=1:num_inst
        img=imread(img_path);
        img=single(rgb2gray(img));
%         x=randi(size(img,1)); %rows
%         y=randi(size(img,2)); %cols
%         
%         Xmin= x-desp;
%         Xmax= x+desp;
%         Ymin= y-desp;
%         Ymax= y+desp;
%         
%         %Conditions in order to take only the pixels of the image
%         if Xmin<1
%             Xmin=1;
%         end
%         if Xmax>size(img,1)
%             Xmax=size(img,1);
%         end
%         if Ymin<1
%             Ymin=1;
%         end
%         if Ymax>size(img,2)
%             Ymax=size(img,2);
%         end
        %img=img(Xmin:Xmax,Ymin:Ymax);
        
        x=randi([desp+1 size(img,1)-desp-1]); %rows
        y=randi([desp+1 size(img,2)-desp-1]); %rows
        img=img(x-desp:x+desp, y-desp:y+desp);
        img=imresize(img,[feature_params.template_size,feature_params.template_size]);
        Hog=vl_hog(img,feature_params.hog_cell_size);
        features_neg(f,:)=Hog(:);
        f=f+1;
    end
end

end

% placeholder to be deleted - THIS ONLY WORKS FOR THE INITIAL DEMO

