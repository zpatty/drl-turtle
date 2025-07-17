close all
frameRight = imread("calib/_right camera_screenshot_18.06.2024.png");
frameLeft = imread("calib/_left camera_screenshot_18.06.2024.png");

[frameLeftRect, frameRightRect, reprojectionMatrix] = ...
    rectifyStereoImages(frameLeft, frameRight, stereoParams);

figure;
imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
title("Rectified Video Frames");

frameLeftGray  = im2gray(frameLeftRect);
frameRightGray = im2gray(frameRightRect);

B = 1/reprojectionMatrix(4,3);
f = reprojectionMatrix(3,4);
disparityMap = disparitySGM(frameLeftGray, frameRightGray);
figure;
dtest = f*B*1./disparityMap./1000;
imshow(dtest,[0 64]);
title("Disparity Map");
colormap jet
colorbar

points3D = reconstructScene(disparityMap, reprojectionMatrix);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
X = points3D(:, :, 1);
Y = points3D(:, :, 2);
Z = points3D(:, :, 3);

dists = sqrt(X.^2 + Y.^2 + Z.^2);
figure;
imshow(dists)
colorbar


ptCloud = pointCloud(points3D, Color=frameLeftRect);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], VerticalAxis="y", ...
    VerticalAxisDir="down");

% Visualize the point cloud
view(player3D, ptCloud);


%%
close all
myDir = "left/"; %gets directory
myFiles = dir(fullfile(myDir,'*.png'));
for k = 1:length(myFiles)
    images{k} = imread(strcat(myDir,myFiles(k).name));
    image_names{k} = convertStringsToChars(strcat(myDir,myFiles(k).name));
end
[imagePoints,boardSize] = detectCheckerboardPoints(image_names, 'HighDistortion', true);
squareSize = 22; % millimeters
worldPoints = generateCheckerboardPoints(boardSize,squareSize);
ims = imageDatastore(image_names);
I = readimage(ims,3); 
imageSize = [size(I,1) size(I,2)];
params = estimateFisheyeParameters(imagePoints,worldPoints,imageSize);

J1 = undistortFisheyeImage(I,params.Intrinsics);
figure
imshowpair(I,J1,'montage')
title('Original Image (left) vs. Corrected Image (right)')

J2 = undistortFisheyeImage(I,params.Intrinsics,'OutputView','same', 'ScaleFactor', 0.2);
figure
imshow(J2)
