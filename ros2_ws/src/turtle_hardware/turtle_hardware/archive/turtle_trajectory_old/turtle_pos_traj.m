clear

%% Dive Kinda
theta = [0 0.1 0.2 0.3;
        0 0.1 0.2 0.3;
        0 -0.1 -0.2 -0.3];

shoulder1 = [180 160 110 180 200 180];
shoulder2 = [180 210 210 140 160 180];
shoulder3 = repmat(180,1,length(shoulder2));
shoulder4 = 180 + (180 - shoulder1);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = repmat(180,1,length(shoulder2));

rear_pitch = 180;
rear_yaw = 180;

shoulder7 = [180 rear_pitch rear_pitch rear_pitch rear_pitch rear_pitch];
shoulder8 = [180 rear_yaw rear_yaw rear_yaw rear_yaw rear_yaw];
shoulder9 = 180 + (180 - shoulder8);
shoulder10 = 180 + (180 - shoulder7);

timePoints = [0 0.5 1 2.5 3 3.5];


wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10]*pi/180;
% wayPoints(wayPoints < 0) = wayPoints(wayPoints < 0) + 360;
tvec = 0:0.01:timePoints(end);
N = length(tvec);
[qd,dqd,ddqd,pp] = cubicpolytraj(wayPoints,timePoints,tvec);

save('qd_p.mat','qd');
save('tvec_p.mat','tvec');