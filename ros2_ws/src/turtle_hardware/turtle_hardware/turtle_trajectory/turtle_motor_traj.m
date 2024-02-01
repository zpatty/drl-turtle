clear
%% Straight
% Need to make sure that trajectory has the same starting and end point
% For any cyclical trajectory, the starting point is given 1 second to move
% from turtle's home position to that starting pointin the cycle
% the cycle trajectory (i.e 2nd and last element) must be the same
% up = 210;
% down = 150;
up = 240;
down = 110;
% up = 150;
% down = 90;
straight = [230 150 230];
% surface = [240 190 250];
turn = 30;
turn = 0;
% front fins
shoulder1 = [180 up down up];
shoulder2 = [180 straight];
shoulder3 = [180 240 140 240];
shoulder4 = 180 + (180 - shoulder1 - turn);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220 120];

% back fins 
rear_yaw = 180;
rear_pitch = 180;
shoulder7 = [180 rear_pitch rear_pitch rear_pitch];
shoulder8 = [180 rear_yaw rear_yaw rear_yaw];
shoulder9 = (180 - shoulder8);
shoulder10 = 180 - shoulder7;
% time points for a single cycle
timePoints = [0 2 3 4];
tshift = timePoints(2:end) + timePoints(2:end);
timePoints_cycle = [timePoints, tshift(2:end)];

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10] * pi/180;
wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]

tvec = 0:0.01:timePoints_cycle(end);
[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints_cycle,tvec);
% 
tvec = 0:0.01:timePoints(end);
qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));   
ddqd = ddqd(:, 1:length(tvec));
save('qd_straight.mat','qd');
save('dqd_straight.mat','dqd');
save('ddqd_straight.mat','ddqd');
save('tvec_straight.mat','tvec');

%% Turn with front flippers
up = 240;
down = 110;

straight = [230 150 230];
turn = 30;
% front fins
shoulder1 = [180 up down up];
shoulder2 = [180 straight];
shoulder3 = [180 240 140 240];
shoulder4 = 180 + (180 - shoulder1 - turn);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220 120];

% back fins 
rear_yaw = 180;
rear_pitch = 180;
shoulder7 = [180 rear_pitch rear_pitch rear_pitch];
shoulder8 = [180 rear_yaw rear_yaw rear_yaw];
shoulder9 = (180 - shoulder8);
shoulder10 = 180 - shoulder7;
% time points for a single cycle
timePoints = [0 2 3 4];
tshift = timePoints(2:end) + timePoints(2:end);
timePoints_cycle = [timePoints, tshift(2:end)];

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10] * pi/180;
wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]

tvec = 0:0.01:timePoints_cycle(end);
[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints_cycle,tvec);
% 
tvec = 0:0.01:timePoints(end);
qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));   
ddqd = ddqd(:, 1:length(tvec));
save('qd_turnrf.mat','qd');
save('dqd_turnrf.mat','dqd');
save('ddqd_turnrf.mat','ddqd');
save('tvec_turnrf.mat','tvec');

%% Dive Front Flippers
up = 210;
down = 150;

straight = [230 150 230];
% surface = [240 190 250];
% turn = 30;
turn = 0;
% front fins
shoulder1 = [180 up down up];
shoulder2 = [180 straight];
shoulder3 = [180 240 140 240];
shoulder4 = 180 + (180 - shoulder1 - turn);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220 120];

% back fins 
rear_yaw = 180;
rear_pitch = 180;
shoulder7 = [180 rear_pitch rear_pitch rear_pitch];
shoulder8 = [180 rear_yaw rear_yaw rear_yaw];
shoulder9 = (180 - shoulder8);
shoulder10 = 180 - shoulder7;
% time points for a single cycle
timePoints = [0 2 3 4];
tshift = timePoints(2:end) + timePoints(2:end);
timePoints_cycle = [timePoints, tshift(2:end)];

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10] * pi/180;
wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]

tvec = 0:0.01:timePoints_cycle(end);
[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints_cycle,tvec);
% 
tvec = 0:0.01:timePoints(end);
qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));   
ddqd = ddqd(:, 1:length(tvec));
save('qd_dive.mat','qd');
save('dqd_dive.mat','dqd');
save('ddqd_dive.mat','ddqd');
save('tvec_dive.mat','tvec');

%% Surface
% up = 240;
% down = 110;
up = 150;
down = 90;
% straight = [230 150 230];
surface = [240 190 250];
turn = 30;
turn = 0;
% front fins
shoulder1 = [180 up down up];
shoulder2 = [180 surface];
shoulder3 = [180 240 140 240];
shoulder4 = 180 + (180 - shoulder1 - turn);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220 120];

% back fins 
rear_yaw = 180;
rear_pitch = 180;
shoulder7 = [180 rear_pitch rear_pitch rear_pitch];
shoulder8 = [180 rear_yaw rear_yaw rear_yaw];
shoulder9 = (180 - shoulder8);
shoulder10 = 180 - shoulder7;
% time points for a single cycle
timePoints = [0 2 3 4];
tshift = timePoints(2:end) + timePoints(2:end);
timePoints_cycle = [timePoints, tshift(2:end)];

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10] * pi/180;
wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]

tvec = 0:0.01:timePoints_cycle(end);
[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints_cycle,tvec);
% 
tvec = 0:0.01:timePoints(end);
qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));   
ddqd = ddqd(:, 1:length(tvec));
save('qd_surface.mat','qd');
save('dqd_surface.mat','dqd');
save('ddqd_surface.mat','ddqd');
save('tvec_surface.mat','tvec');

%% Turn with rear flippers
up = 240;
down = 110;

straight = [230 150 230];
turn = 30;
% front fins
shoulder1 = [180 up down up];
shoulder2 = [180 straight];
shoulder3 = [180 240 140 240];
shoulder4 = 180 + (180 - shoulder1 - turn);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220 120];

% back fins 
rear_yaw = 180;
rear_pitch = 180;
shoulder7 = [180 rear_pitch rear_pitch rear_pitch];
shoulder8 = [180 rear_yaw rear_yaw rear_yaw];
shoulder9 = (180 - shoulder8);
shoulder10 = 180 - shoulder7;
% time points for a single cycle
timePoints = [0 2 3 4];
tshift = timePoints(2:end) + timePoints(2:end);
timePoints_cycle = [timePoints, tshift(2:end)];

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10] * pi/180;
wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]

tvec = 0:0.01:timePoints_cycle(end);
[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints_cycle,tvec);
% 
tvec = 0:0.01:timePoints(end);
qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));   
ddqd = ddqd(:, 1:length(tvec));
save('qd_turnrr.mat','qd');
save('dqd_turnrr.mat','dqd');
save('ddqd_turnrr.mat','ddqd');
save('tvec_turnrr.mat','tvec');