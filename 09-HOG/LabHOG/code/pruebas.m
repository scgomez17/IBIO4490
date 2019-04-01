
img= imread('cameraman.tif');
imshow (img)
hog= vl_hog(single(img),6);
img_hog= vl_hog('render',hog);
imshow (img_hog)

faceDetector = vision.CascadeObjectDetector;
bboxes = faceDetector(img);
IFaces = insertObjectAnnotation(img,'rectangle',bboxes,'Face')
imshow (IFaces)